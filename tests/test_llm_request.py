"""Anthropic request shape: the instructions are the cacheable prefix and per-request evidence stays outside it."""
from types import SimpleNamespace

from contextrag.core.models import Context, ParentChunk
from contextrag.generation.llm import SYSTEM_PROMPT, AnthropicLLM, anthropic_request


def ctx(text):
    p = ParentChunk(id="p", doc_id="d", tenant="t", source="d.md", text=text, start=0, end=len(text))
    return Context(index=1, parent=p, score=1.0)


def test_cache_control_marks_only_the_stable_system_prefix():
    req = anthropic_request("m", 256, "¿pregunta?", [ctx("Los registros se conservan 7 años.")], None,
                            prompt_cache=True)
    assert req["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    user = req["messages"][0]["content"]
    assert "Los registros se conservan 7 años." in user and "Los registros" not in SYSTEM_PROMPT


def test_without_caching_the_system_is_a_plain_string():
    req = anthropic_request("m", 256, "q", [ctx("x")], None, prompt_cache=False)
    assert req["system"] == SYSTEM_PROMPT and req["temperature"] == 0


def test_generate_sends_the_cached_request_and_records_usage_from_the_response():
    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=" respuesta [1] ")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=3, cache_read_input_tokens=800,
                                      cache_creation_input_tokens=0))

    llm = AnthropicLLM.__new__(AnthropicLLM)  # no SDK client: only the request and response handling are tested
    llm._c = SimpleNamespace(messages=FakeMessages())
    llm.model, llm.max_tokens, llm.prompt_cache, llm.last_usage = "m", 256, True, {}
    assert llm.generate("q", [ctx("x")]) == "respuesta [1]"
    assert calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert llm.last_usage["cache_read_input_tokens"] == 800
