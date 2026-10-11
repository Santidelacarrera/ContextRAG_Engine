"""Exercises the real-model code paths (sentence-transformers / transformers / torch) using *tiny models built
on the fly with random weights*. This verifies integration (pair tokenisation, the entailment label index,
batching, truncation, normalisation, wiring through ``factory.from_env``) -- NOT model quality. Skipped when the
ML extras are not installed."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from contextrag import Status, factory
from contextrag.core.embeddings import SentenceTransformerEmbedder
from contextrag.retrieval.reranker import CrossEncoderReranker
from contextrag.verification.nli_verifier import DebertaNLI, verify_attribution
from conftest import POLICY

WORDS = ("los registros de auditoría se conservan durante 7 años el cielo es azul las claves rotan cada 12 meses "
         "datos en reposo cifran con aes 256 . , ¿ ? [ ] 1 2 3").split()


def _tokenizer(path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS))
    (path / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True).save_pretrained(path)
    return len(vocab)


def _classifier(path, labels, bias=None, seed=0):
    torch.manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    n_vocab = _tokenizer(path)
    cfg = BertConfig(vocab_size=n_vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=512, num_labels=len(labels),
                     id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    m = BertForSequenceClassification(cfg)
    if bias is not None:  # make the model's verdict deterministic regardless of the input
        with torch.no_grad():
            m.classifier.weight.zero_()
            m.classifier.bias.copy_(torch.tensor(bias, dtype=torch.float))
    m.save_pretrained(path)
    return str(path)


NLI_LABELS = ["contradiction", "entailment", "neutral"]  # entailment deliberately NOT at index 0 or 2 of a 3-way tie


@pytest.fixture(scope="module")
def nli_yes(tmp_path_factory):
    return _classifier(tmp_path_factory.mktemp("nli_yes"), NLI_LABELS, bias=[-8.0, 8.0, -8.0])


@pytest.fixture(scope="module")
def nli_no(tmp_path_factory):
    return _classifier(tmp_path_factory.mktemp("nli_no"), NLI_LABELS, bias=[8.0, -8.0, 8.0])


@pytest.fixture(scope="module")
def nli_random(tmp_path_factory):
    return _classifier(tmp_path_factory.mktemp("nli_rand"), NLI_LABELS)


@pytest.fixture(scope="module")
def rerank_dir(tmp_path_factory):
    return _classifier(tmp_path_factory.mktemp("rr"), ["LABEL_0"])


@pytest.fixture(scope="module")
def embed_dir(tmp_path_factory):
    from sentence_transformers import SentenceTransformer, models

    p = tmp_path_factory.mktemp("emb")
    _classifier(p / "base", ["LABEL_0"])
    st = SentenceTransformer(modules=[models.Transformer(str(p / "base")), models.Pooling(32, "mean"), models.Normalize()])
    st.save(str(p / "st"))
    return str(p / "st")


# ---- DebertaNLI ---------------------------------------------------------------------------

def test_nli_reads_the_probability_of_the_entailment_label_not_a_fixed_index(nli_yes, nli_no):
    pairs = [("los registros se conservan 7 años .", "los registros se conservan 7 años .")]
    assert DebertaNLI(nli_yes).entailment(pairs)[0] > 0.99
    assert DebertaNLI(nli_no).entailment(pairs)[0] < 0.01


def test_nli_batching_is_consistent_and_handles_long_inputs(nli_random):
    nli = DebertaNLI(nli_random, batch_size=3)
    pairs = [(" ".join(["los registros se conservan"] * k), "el cielo es azul") for k in (1, 2, 5, 9, 30, 400)]
    batched = nli.entailment(pairs)  # 400 repetitions >> 512 tokens: must be truncated, not crash
    singles = [nli.entailment([p])[0] for p in pairs]
    assert len(batched) == 6 and all(0.0 <= x <= 1.0 for x in batched)
    assert all(abs(a - b) < 1e-4 for a, b in zip(batched, singles))  # padding does not change results
    assert nli.entailment([]) == []


def test_verify_attribution_and_verifier_with_a_transformer_nli(nli_yes, nli_no):
    chunks = ["Los registros se conservan durante 7 años. Las claves rotan cada 12 meses."]
    assert verify_attribution("Los registros se conservan durante 7 años.", chunks, DebertaNLI(nli_yes)) > 0.99
    assert verify_attribution("Los registros se conservan durante 7 años.", chunks, DebertaNLI(nli_no)) < 0.01


# ---- reranker / embedder -------------------------------------------------------------------

def test_cross_encoder_reranker_scores_each_passage(rerank_dir):
    rr = CrossEncoderReranker(rerank_dir)
    passages = ["los registros de auditoría", "el cielo es azul", "las claves rotan cada 12 meses"]
    s = rr.score("registros", passages)
    assert len(s) == 3 and all(isinstance(x, float) for x in s) and s == rr.score("registros", passages)
    assert rr.min_score is None and CrossEncoderReranker(rerank_dir, min_score=0.5).min_score == 0.5


def test_sentence_transformer_embedder_is_normalised_and_deterministic(embed_dir):
    e = SentenceTransformerEmbedder(embed_dir)
    v = e.embed(["los registros de auditoría", "el cielo es azul"])
    assert e.dim == 32 and len(v) == 2 and len(v[0]) == 32
    for row in v:
        assert abs(sum(x * x for x in row) - 1.0) < 1e-4
    assert e.embed(["el cielo es azul"])[0] == pytest.approx(v[1], abs=1e-5)
    assert e.embed(["los registros", "los registros"])[0] == pytest.approx(e.embed(["los registros"])[0], abs=1e-5)


# ---- end to end through env wiring ---------------------------------------------------------

def _env(monkeypatch, embed_dir, rerank_dir, nli_dir, **extra):
    for k, v in {"EMBEDDER": "bge", "EMBED_MODEL": embed_dir, "RERANKER": "cross-encoder", "RERANK_MODEL": rerank_dir,
                 "NLI": "deberta", "NLI_MODEL": nli_dir, **extra}.items():
        monkeypatch.setenv(k, v)


def test_full_pipeline_with_transformer_components_and_qdrant(monkeypatch, tmp_path, embed_dir, rerank_dir, nli_yes):
    pytest.importorskip("qdrant_client")
    pytest.importorskip("cryptography")
    _env(monkeypatch, embed_dir, rerank_dir, nli_yes, MASTER_KEY="0123456789abcdef-master")
    rag = factory.from_env()
    assert type(rag.verifier.nli).__name__ == "DebertaNLI" and type(rag.embedder).__name__ == "CachedEmbedder"
    assert type(rag.index.store).__name__ == "QdrantVectorStore" and rag.index.store.cipher
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    a = rag.ask("¿Cuánto tiempo se conservan los registros de auditoría?", tenant="acme")
    # random embedder/reranker => retrieval quality is arbitrary, but the wiring must run end to end and any
    # produced answer must carry citations and evidence offsets that point into the original document
    assert a.status in (Status.VERIFIED, Status.FALLBACK)
    if a.status is Status.VERIFIED:
        ev = a.claims[0].evidence
        assert POLICY[ev.start : ev.end] == ev.text
    assert rag.health()["index"] is True


def test_a_contradicting_nli_forces_the_fallback_even_with_a_perfect_retriever(monkeypatch, embed_dir, rerank_dir, nli_no):
    _env(monkeypatch, embed_dir, rerank_dir, nli_no)
    rag = factory.from_env()
    rag.reranker = rag.retriever.reranker = factory.LexicalReranker()  # make retrieval deterministic
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    a = rag.ask("¿Cuánto tiempo se conservan los registros de auditoría?", tenant="acme")
    assert a.sources and a.status is Status.FALLBACK and a.attempts == 2 and a.confidence < 0.01


def test_an_entailing_nli_lets_the_verified_path_through(monkeypatch, embed_dir, rerank_dir, nli_yes):
    _env(monkeypatch, embed_dir, rerank_dir, nli_yes)
    rag = factory.from_env()
    rag.reranker = rag.retriever.reranker = factory.LexicalReranker()
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    a = rag.ask("¿Cuánto tiempo se conservan los registros de auditoría?", tenant="acme")
    assert a.status is Status.VERIFIED and "7 años" in a.text
