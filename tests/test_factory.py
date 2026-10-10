import pytest

from contextrag import Settings, Status, factory
from contextrag.core.cache import InMemoryCache
from contextrag.retrieval.vector_store import InMemoryVectorStore
from conftest import POLICY

ENV_VARS = ["EMBEDDER", "RERANKER", "NLI", "LLM", "QDRANT_URL", "POSTGRES_DSN", "REDIS_URL", "MASTER_KEY",
            "CONTEXTRAG_TOP_K", "CONTEXTRAG_VERIFICATION_THRESHOLD"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)


def test_settings_from_env_parses_types(monkeypatch):
    monkeypatch.setenv("CONTEXTRAG_TOP_K", "7")
    monkeypatch.setenv("CONTEXTRAG_VERIFICATION_THRESHOLD", "0.99")
    s = Settings.from_env()
    assert s.top_k == 7 and s.verification_threshold == 0.99 and s.parent_tokens == 1024


def test_from_env_defaults_are_self_contained():
    rag = factory.from_env()
    assert isinstance(rag.index.store, InMemoryVectorStore) and isinstance(rag.cache, InMemoryCache)
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    assert rag.ask("¿Cuánto se conservan los registros de auditoría?", tenant="acme").status is Status.VERIFIED


def test_from_env_with_master_key_uses_encrypted_qdrant(monkeypatch):
    pytest.importorskip("qdrant_client")
    pytest.importorskip("cryptography")
    monkeypatch.setenv("MASTER_KEY", "0123456789abcdef-master")
    monkeypatch.setenv("CONTEXTRAG_VERIFICATION_THRESHOLD", "0.9")
    rag = factory.from_env()
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    store = rag.index.store
    assert store.cipher and all("auditoría" not in p.payload["text"] for p in store.client.scroll(store.col, limit=50)[0])
    a = rag.ask("¿Cuánto se conservan los registros de auditoría?", tenant="acme")
    assert a.status is Status.VERIFIED and rag.cfg.verification_threshold == 0.9


def test_ingest_path_walks_directories_and_skips_hidden_and_binary(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\nLa contraseña caduca cada 90 días.")
    (tmp_path / "src.py").write_text("def f():\n    return 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.txt").write_text("secret")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    rag = factory.offline()
    done = rag.ingestor.ingest_path(tmp_path, tenant="t")
    assert sorted(d["doc_id"] for d in done) == ["docs/a.md", "src.py"]
    assert rag.ask("¿cuándo caduca la contraseña?", tenant="t").status is Status.VERIFIED
    [single] = rag.ingestor.ingest_path(tmp_path / "src.py", tenant="t")
    assert single["doc_id"] == "src.py" and single["kind"] == "code"


def test_cli_ask_and_eval(tmp_path, capsys):
    import json

    from contextrag.__main__ import main

    (tmp_path / "p.md").write_text("# Retención\n\nLos registros de auditoría se conservan durante 7 años.")
    assert main(["ask", "¿Cuánto se conservan los registros de auditoría?", "--tenant", "t", "--docs", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "verified" and "7 años" in out["claims"][0]["evidence"]["text"]
    assert main(["ask", "¿Quién ganó el mundial?", "--tenant", "t", "--docs", str(tmp_path)]) == 2  # fallback => exit 2
    capsys.readouterr()
    assert main(["eval", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["verifier"]["false_support_rate"] == 0 and rep["rag"]["unsafe_rate"] == 0
    assert main(["eval"]) == 0 and "false_support_rate=0.0%" in capsys.readouterr().out


def test_cli_eval_exits_nonzero_on_unsafe_custom_cases(tmp_path, capsys):
    import json

    from contextrag.__main__ import main

    bad = tmp_path / "v.json"  # a claim the heuristic model (wrongly) supports => harness must fail the run
    bad.write_text(json.dumps([{"id": "x", "label": "unsupported", "category": "paraphrase",
                                "premise": "Las claves se rotan cada 12 meses.", "claim": "Las claves se rotan cada 12 meses."}]))
    assert main(["eval", "--verifier-cases", str(bad), "--json"]) == 1
