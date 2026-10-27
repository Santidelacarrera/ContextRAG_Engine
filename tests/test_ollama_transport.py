"""OllamaLLM over a real local HTTP server: redirects are refused, schemes and URL parts are validated.

Nothing here reaches a model; the server only answers with the canned bodies each test sets up."""
import http.server
import json
import threading

import pytest

from contextrag.core.models import Context, ParentChunk
from contextrag.generation.llm import GenerationError, OllamaLLM

CTX = [Context(1, ParentChunk("p", "d", "t", "d.md", "El plazo es de cuatro años.", 0, 26), 1.0)]
ANSWER = {"message": {"content": "El plazo es de cuatro años. [1]"}}


class _Server:
    """A throwaway server on 127.0.0.1. ``routes`` maps a path to (status, headers, body)."""

    def __init__(self, routes):
        self.hits = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                outer.hits.append(self.path)
                status, headers, body = routes.get(self.path, (404, {}, b""))
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep test output quiet
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_a_local_ollama_answer_is_returned():
    with _Server({"/api/chat": (200, {"Content-Type": "application/json"}, json.dumps(ANSWER).encode())}) as s:
        assert OllamaLLM("m", host=s.url).generate("¿Plazo?", CTX) == "El plazo es de cuatro años. [1]"
        assert s.hits == ["/api/chat"]


def test_a_redirect_is_refused_and_its_target_is_never_contacted():
    # A configured host that answers 302 must not be followed: the target here is another path on the same server,
    # so a hit on it would prove the redirect was followed.
    routes = {"/api/chat": (302, {"Location": "/forbidden-target"}, b""),
              "/forbidden-target": (200, {"Content-Type": "application/json"}, json.dumps(ANSWER).encode())}
    with _Server(routes) as s:
        with pytest.raises(GenerationError) as info:
            OllamaLLM("m", host=s.url).generate("¿Plazo?", CTX)
        assert str(info.value) == "HTTPError"
        assert s.hits == ["/api/chat"]


@pytest.mark.parametrize("host", [
    "file:///etc/passwd",
    "ftp://example.org",
    "gopher://example.org",
    "http://",
    "localhost:11434",
    "http://user:secret@example.org",
    "http://example.org/?x=1",
    "http://example.org/#frag",
    "http://exa mple.org",
    "",
    None,
])
def test_hosts_outside_the_contract_are_refused_at_construction(host):
    with pytest.raises(ValueError):
        OllamaLLM("m", host=host)


def test_a_path_prefix_is_kept_and_a_trailing_slash_is_removed():
    llm = OllamaLLM("m", host="https://ollama.internal.example/prefix/")
    assert llm.host == "https://ollama.internal.example/prefix"


def test_an_unreachable_host_reports_the_class_only():
    with _Server({}) as s:
        port = s.httpd.server_address[1]
    # the server is closed now: the connection is refused
    with pytest.raises(GenerationError) as info:
        OllamaLLM("m", host=f"http://127.0.0.1:{port}", timeout_seconds=2).generate("¿Plazo?", CTX)
    assert str(info.value) == "URLError"
    assert str(port) not in str(info.value)


# ---- destination guard: what is contacted, and how many times the name is resolved ----------------------------

import socket  # noqa: E402  (grouped with the guard tests on purpose)

from contextrag.generation import llm as L  # noqa: E402


def _fake_resolution(monkeypatch, name, *answers):
    """Successive answers for ``name`` only; other lookups (socket.create_connection resolves IP literals too) use the
    real resolver. Every lookup of ``name`` is recorded."""
    real = socket.getaddrinfo
    calls = []
    queue = list(answers)

    def fake(host, port, *a, **kw):
        if host != name:
            return real(host, port, *a, **kw)
        calls.append(host)
        ip = queue.pop(0) if len(queue) > 1 else queue[0]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return calls


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_a_metadata_address_is_refused_before_any_connection(monkeypatch, scheme):
    attempts = []
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: attempts.append(a))
    with pytest.raises(GenerationError) as info:
        OllamaLLM("m", host=f"{scheme}://169.254.169.254:11434").generate("¿Plazo?", CTX)
    assert str(info.value) == "URLError"
    assert attempts == []  # nothing was opened


def test_a_name_that_resolves_to_link_local_is_refused(monkeypatch):
    attempts = []
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: attempts.append(a))
    _fake_resolution(monkeypatch, "ollama.internal.example", "169.254.10.20")
    with pytest.raises(GenerationError):
        OllamaLLM("m", host="http://ollama.internal.example:11434").generate("¿Plazo?", CTX)
    assert attempts == []


def test_the_name_is_resolved_once_and_the_checked_address_is_the_one_contacted(monkeypatch):
    # A rebinding server would answer loopback first and metadata second. One resolution, one connection: the second
    # answer is never requested, so it cannot be used.
    # The server is started before the resolver is replaced: HTTPServer resolves its own name when it binds.
    with _Server({"/api/chat": (200, {"Content-Type": "application/json"}, json.dumps(ANSWER).encode())}) as s:
        port = s.httpd.server_address[1]
        calls = _fake_resolution(monkeypatch, "rebind.example", "127.0.0.1", "169.254.169.254")
        assert OllamaLLM("m", host=f"http://rebind.example:{port}").generate("¿Plazo?", CTX) == "El plazo es de cuatro años. [1]"
        assert calls == ["rebind.example"]
        assert s.hits == ["/api/chat"]


def test_an_environment_proxy_is_not_used_for_ollama(monkeypatch):
    # With a proxy configured in the environment the request would leave through it, where the address check cannot see.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    with _Server({"/api/chat": (200, {"Content-Type": "application/json"}, json.dumps(ANSWER).encode())}) as s:
        assert OllamaLLM("m", host=s.url).generate("¿Plazo?", CTX) == "El plazo es de cuatro años. [1]"
        assert s.hits == ["/api/chat"]  # reached directly, not through the (dead) proxy


def test_an_oversized_answer_is_refused_without_being_buffered_whole():
    big = b'{"message": {"content": "' + b"a" * (L.MAX_RESPONSE_BYTES + 10) + b'"}}'
    with _Server({"/api/chat": (200, {"Content-Type": "application/json"}, big)}) as s:
        with pytest.raises(GenerationError) as info:
            OllamaLLM("m", host=s.url).generate("¿Plazo?", CTX)
    assert str(info.value) == "ResponseTooLarge"


def test_a_private_network_host_is_still_allowed_by_design():
    # Loopback is the normal local deployment and must keep working; only link-local and metadata are refused.
    with _Server({"/api/chat": (200, {"Content-Type": "application/json"}, json.dumps(ANSWER).encode())}) as s:
        assert "127.0.0.1" in L._checked_addresses("127.0.0.1", s.httpd.server_address[1])
