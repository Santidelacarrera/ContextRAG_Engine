"""A realistic Obsidian-style vault: frontmatter, wikilinks, embeds, tags, callouts, tables, code, accents in
names, hidden config directories. Checks the vault indexes and answers with verifiable coordinates."""
import pytest

from contextrag import Settings, Status, factory
from contextrag.core.config import Settings as CoreSettings  # noqa: F401  (same class; documents the import path)
from contextrag.core.text import split_sentences
from contextrag.verification.nli_verifier import NLIVerifier, parse_claims, verify_attribution

VAULT = {
    "Seguridad/Política de contraseñas.md": """---
title: Política de contraseñas
tags: [seguridad, politica]
aliases: [passwords]
---
# Política de contraseñas

Las contraseñas caducan cada 90 días. Ver [[Gestión de claves|claves]] y ![[diagrama.png]].

> [!warning] Importante
> Las contraseñas no pueden reutilizarse durante 12 meses.

## Excepciones #seguridad

El equipo de guardia puede usar claves temporales de 24 horas.
""",
    "Seguridad/Gestión de claves.md": """# Gestión de claves

| Tipo | Rotación |
|------|----------|
| API  | 30 días  |
| TLS  | 365 días |

```python
ROTATION_DAYS = 30
```

Las claves de API se rotan cada 30 días.
""",
    "Proyectos/Nexus.md": "# Nexus\n\nEl laboratorio Nexus Security & AI Lab desarrolla ContextRAG Engine.\n",
    ".obsidian/workspace.json": '{"secret": "no debe indexarse"}',
    ".trash/borrada.md": "Nota borrada que no debe aparecer.",
}


@pytest.fixture
def vault(tmp_path):
    for name, text in VAULT.items():
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    rag = factory.offline()
    done = rag.ingestor.ingest_path(tmp_path, tenant="nexus")
    return rag, tmp_path, done


def test_vault_indexes_notes_and_skips_hidden_config(vault):
    rag, _, done = vault
    assert sorted(d["doc_id"] for d in done) == ["Proyectos/Nexus.md", "Seguridad/Gestión de claves.md",
                                                 "Seguridad/Política de contraseñas.md"]
    assert all(d["kind"] == "markdown" and d["children"] >= 1 for d in done)
    assert rag.ask("¿qué dice la nota borrada?", tenant="nexus").status is Status.FALLBACK


@pytest.mark.parametrize("question,expected", [
    ("¿Cada cuánto caducan las contraseñas?", "90 días"),
    ("¿Cada cuánto se rotan las claves de API?", "30 días"),
    ("¿Quién desarrolla ContextRAG Engine?", "Nexus"),
])
def test_questions_over_the_vault_are_verified_with_exact_coordinates(vault, question, expected):
    rag, root, _ = vault
    a = rag.ask(question, tenant="nexus")
    assert a.status is Status.VERIFIED and expected in a.text
    ev = a.claims[0].evidence
    original = (root / ev.doc_id).read_text(encoding="utf-8")
    assert original[ev.start : ev.end] == ev.text  # offsets point into the real note on disk


def test_sections_and_frontmatter_are_handled(vault):
    rag, _, _ = vault
    a = rag.ask("¿Qué claves temporales puede usar el equipo de guardia?", tenant="nexus")
    assert a.status is Status.VERIFIED and a.sources[0].parent.doc_id == "Seguridad/Política de contraseñas.md"
    assert "Excepciones" in a.sources[0].parent.section


def test_reingesting_an_unchanged_vault_is_a_noop_and_edits_are_picked_up(vault):
    rag, root, _ = vault
    assert all(d["unchanged"] for d in rag.ingestor.ingest_path(root, tenant="nexus"))
    note = root / "Seguridad/Política de contraseñas.md"
    note.write_text(note.read_text(encoding="utf-8").replace("cada 90 días", "cada 60 días"), encoding="utf-8")
    changed = [d["doc_id"] for d in rag.ingestor.ingest_path(root, tenant="nexus") if not d["unchanged"]]
    assert changed == ["Seguridad/Política de contraseñas.md"]
    assert "60 días" in rag.ask("¿Cada cuánto caducan las contraseñas?", tenant="nexus").text


# ---- abbreviations ---------------------------------------------------------------------------

def test_abbreviations_do_not_split_sentences():
    s = [t for t, _, _ in split_sentences("El Dr. Pérez aprobó el plan, p. ej. el de 2024. Luego se fue.")]
    assert s == ["El Dr. Pérez aprobó el plan, p. ej. el de 2024.", "Luego se fue."]
    s = [t for t, _, _ in split_sentences("Use e.g. TLS 1.3 or similar. Done.")]
    assert s == ["Use e.g. TLS 1.3 or similar.", "Done."]
    assert [c for c, _ in parse_claims("Dr. Smith approved the plan, e.g. the 2024 one. [1] It ended.")] == [
        "Dr. Smith approved the plan, e.g. the 2024 one.", "It ended."]
    offs = split_sentences("Dr. A llegó. B se fue.", base=10)
    text = "Dr. A llegó. B se fue."
    assert all(text[s - 10 : e - 10] == t for t, s, e in offs)


def test_verbatim_claims_with_abbreviations_are_supported_by_the_heuristic_nli():
    premise = "El Dr. Pérez aprobó el plan e.g. el de 2024."
    assert verify_attribution(premise, [premise]) == 1.0
    from conftest import POLICY  # noqa: F401
    from contextrag.core.models import Context, ParentChunk

    c = Context(1, ParentChunk("p", "d", "t", "d", premise, 0, len(premise)))
    [v] = NLIVerifier().verify(premise + " [1]", [c])
    assert v.supported and v.evidence.text == premise


# ---- decision gate configuration -------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, float("nan")])
def test_a_threshold_that_would_disable_verification_is_refused(bad):
    with pytest.raises(ValueError):
        Settings(verification_threshold=bad)
    with pytest.raises(ValueError):
        NLIVerifier(threshold=bad)


def test_threshold_boundaries_are_inclusive_at_exactly_the_configured_value():
    class Fixed:
        def __init__(self, v):
            self.v = v

        def entailment(self, pairs):
            return [self.v] * len(pairs)

    from contextrag.core.models import Context, ParentChunk

    c = Context(1, ParentChunk("p", "d", "t", "d", "Alpha es rojo.", 0, 14))
    for score, ok in ((0.95, True), (0.9499999, False), (1.0, True), (0.0, False)):
        assert NLIVerifier(Fixed(score), 0.95).verify("Alpha es rojo. [1]", [c])[0].supported is ok
    with pytest.raises(ValueError):
        Settings(max_attempts=0)
    assert Settings().verification_threshold == 0.95


def test_pipeline_applies_the_gate_end_to_end_at_the_boundary():
    class Fixed:
        def __init__(self, v):
            self.v = v

        def entailment(self, pairs):
            return [self.v] * len(pairs)

    for score, expected in ((0.95, Status.VERIFIED), (0.9499, Status.FALLBACK)):
        rag = factory.offline()
        rag.verifier.nli = Fixed(score)
        rag.ingestor.ingest_text("Los registros se conservan 7 años.", tenant="t", doc_id="d")
        a = rag.ask("¿cuánto se conservan los registros?", tenant="t")
        assert a.status is expected
        if expected is Status.FALLBACK:
            assert a.confidence < 0.95 and "95%" in a.text
