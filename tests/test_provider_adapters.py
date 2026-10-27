"""Provider adapters, tested with simulated clients. These prove the adapter's own handling, not any model's quality."""
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from contextrag.core.models import Context, ParentChunk
from contextrag.generation import llm as L
from contextrag.generation.llm import GenerationError, OllamaLLM

CTX = [Context(1, ParentChunk("p", "d", "t", "d.md", "El plazo es de cuatro años.", 0, 26), 1.0)]
SECRET_BODY = "BODY-SECRET-7731 /home/deploy/prompt.txt"


def block(text, kind="text"):
    return SimpleNamespace(type=kind, text=text)


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, 0

    def create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def anthropic_with(response=None, error=None):
    llm = object.__new__(L.AnthropicLLM)
    llm.model, llm.max_tokens, llm.prompt_cache, llm.last_usage = "m", 256, True, {}
    llm._c = SimpleNamespace(messages=FakeMessages(response, error))
    return llm


def reply(text_blocks, stop="end_turn"):
    return SimpleNamespace(content=text_blocks, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_a_normal_answer_is_returned_and_usage_recorded():
    llm = anthropic_with(reply([block("El plazo es de cuatro años. [1]")]))
    assert llm.generate("¿Plazo?", CTX) == "El plazo es de cuatro años. [1]"
    assert llm.last_usage["input_tokens"] == 10 and llm.last_usage["output_tokens"] == 5


def test_a_truncated_answer_is_refused_even_though_it_has_text():
    llm = anthropic_with(reply([block("El plazo es de cuatro")], stop="max_tokens"))
    with pytest.raises(GenerationError, match="TruncatedOutput"):
        llm.generate("¿Plazo?", CTX)


@pytest.mark.parametrize("content", [[], [block("", "text")], [block("x", "tool_use")], None])
def test_an_empty_or_non_text_response_is_an_error_not_a_refusal_to_answer(content):
    llm = anthropic_with(reply(content))
    with pytest.raises(GenerationError, match="EmptyResponse"):
        llm.generate("¿Plazo?", CTX)


def test_provider_errors_reach_the_caller_as_a_class_name_never_as_the_provider_message():
    from contextrag import factory

    class RateLimited(Exception):
        pass

    rag = factory.offline(llm=anthropic_with(error=RateLimited(SECRET_BODY)))
    rag.ingestor.ingest_text("El plazo es de cuatro años.", tenant="t", doc_id="d.md", source="d.md")
    with pytest.raises(GenerationError) as info:
        rag.ask("¿Plazo?", tenant="t")
    assert str(info.value) == "RateLimited"  # class only, from the pipeline's wrapper
    assert SECRET_BODY not in str(info.value)
    assert isinstance(info.value.__cause__, RateLimited)  # kept for internal debugging, never for a client


def test_an_oversized_request_is_refused_before_it_is_sent_and_the_size_is_reported_not_the_content():
    llm = anthropic_with(reply([block("x")]))
    huge = [Context(1, ParentChunk("p", "d", "t", "d.md", "palabra " * 40_000, 0, 1), 1.0)]
    with pytest.raises(GenerationError, match="InputTooLarge") as info:
        llm.generate("¿Plazo?", huge)
    assert llm._c.messages.calls == 0  # nothing left the process
    assert "palabra" not in str(info.value)


def test_the_sdk_client_is_built_with_an_explicit_timeout_and_bounded_retries(monkeypatch):
    seen = {}

    class FakeSDK:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    fake = SimpleNamespace(Anthropic=FakeSDK)
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake)
    L.AnthropicLLM("m", timeout_seconds=7, max_retries=1)
    assert seen == {"timeout": 7, "max_retries": 1}
    with pytest.raises(ValueError):
        L.AnthropicLLM("m", timeout_seconds=0)
    with pytest.raises(ValueError):
        L.AnthropicLLM("m", max_retries=9)


# ---- Ollama -------------------------------------------------------------------------------------------------

class FakeResponse(io.BytesIO):
    headers: dict[str, str] = {}  # a real response exposes the headers; none is declared here, so the size check reads the body
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ollama_returning(monkeypatch, body: bytes | None = None, error: Exception | None = None):
    seen = {}

    def fake_open(req, timeout):
        seen["timeout"] = timeout
        if error:
            raise error
        return FakeResponse(body)

    # OllamaLLM sends through an opener that refuses redirects; the real transport is covered in test_ollama_transport.py
    monkeypatch.setattr(L.urllib.request, "build_opener", lambda *handlers: SimpleNamespace(open=fake_open))
    return seen


def test_ollama_uses_its_configured_timeout_and_returns_the_text(monkeypatch):
    seen = ollama_returning(monkeypatch, json.dumps({"message": {"content": " El plazo es de cuatro años. [1] "}}).encode())
    assert OllamaLLM("m", timeout_seconds=9).generate("¿Plazo?", CTX) == "El plazo es de cuatro años. [1]"
    assert seen["timeout"] == 9


@pytest.mark.parametrize("body", [b"not json", b'{"message": {}}', b'{"other": 1}', b'{"message": {"content": ""}}'])
def test_malformed_or_empty_ollama_responses_are_errors(monkeypatch, body):
    ollama_returning(monkeypatch, body)
    with pytest.raises(GenerationError) as info:
        OllamaLLM("m").generate("¿Plazo?", CTX)
    assert str(info.value) in ("MalformedResponse", "EmptyResponse")


def test_ollama_transport_errors_do_not_expose_the_host_or_the_body(monkeypatch):
    ollama_returning(monkeypatch, error=urllib.error.URLError(f"connect refused http://internal-host:11434 {SECRET_BODY}"))
    with pytest.raises(GenerationError) as info:
        OllamaLLM("m", host="http://internal-host:11434").generate("¿Plazo?", CTX)
    assert str(info.value) == "URLError"
    assert "internal-host" not in str(info.value) and SECRET_BODY not in str(info.value)


def test_ollama_refuses_an_out_of_range_timeout():
    with pytest.raises(ValueError):
        OllamaLLM("m", timeout_seconds=0)
