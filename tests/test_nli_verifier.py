import pytest

from contextrag.core.models import Context, ParentChunk
from contextrag.verification.nli_verifier import (
    LexicalEntailment,
    NLIVerifier,
    fallback_message,
    parse_claims,
    verify_attribution,
)

CHUNKS = [
    "Los registros de auditoría se conservan durante 7 años. Los registros de acceso se eliminan a los 90 días.",
    "Todos los datos en reposo se cifran con AES-256. Las claves se rotan cada 12 meses.",
]


def ctx(text, index=1, start=100, page=None):
    return Context(index, ParentChunk(f"p{index}", "doc.md", "t", "doc.md", text, start, start + len(text), page=page))


# ---- verify_attribution ------------------------------------------------------------------

def test_fully_supported_statement_scores_one():
    assert verify_attribution("Los registros de auditoría se conservan durante 7 años.", CHUNKS) == 1.0


def test_statement_spanning_two_chunks_needs_each_claim_supported_somewhere():
    s = "Los datos en reposo se cifran con AES-256. Los registros de acceso se eliminan a los 90 días."
    assert verify_attribution(s, CHUNKS) == 1.0


def test_one_unsupported_claim_sinks_the_whole_statement():
    s = "Los datos en reposo se cifran con AES-256. El servidor está en Marte."
    assert verify_attribution(s, CHUNKS) < 0.95


def test_wrong_number_scores_zero():
    assert verify_attribution("Los registros de auditoría se conservan durante 10 años.", CHUNKS) == 0.0


def test_negation_flip_scores_zero():
    assert verify_attribution("Los registros de auditoría no se conservan durante 7 años.", CHUNKS) == 0.0


def test_unrelated_statement_scores_low_and_degenerate_inputs_are_zero():
    assert verify_attribution("La capital de Francia es París.", CHUNKS) < 0.5
    assert verify_attribution("", CHUNKS) == 0.0
    assert verify_attribution("algo", []) == 0.0
    assert verify_attribution("algo", [""]) == 0.0


def test_citation_markers_are_ignored_by_standalone_score():
    assert verify_attribution("Los datos en reposo se cifran con AES-256. [2]", CHUNKS) == 1.0


def test_score_uses_the_supplied_nli_model():
    class Fixed:
        def __init__(self, v):
            self.v = v

        def entailment(self, pairs):
            return [self.v] * len(pairs)

    assert verify_attribution("cualquier cosa.", CHUNKS, Fixed(0.97)) == 0.97
    assert verify_attribution("cualquier cosa.", CHUNKS, Fixed(0.4)) == 0.4


def test_lexical_entailment_basics():
    nli = LexicalEntailment()
    assert nli.entailment([("las claves se rotan cada 12 meses", "claves rotan 12 meses")]) == [1.0]
    assert nli.entailment([("las claves se rotan cada 12 meses", "claves rotan 6 meses")]) == [0.0]
    assert nli.entailment([("a", "")]) == [0.0]


# ---- decision gate message ---------------------------------------------------------------

def test_fallback_message_matches_spec():
    assert fallback_message() == ("No se puede verificar la respuesta con un 95% de certidumbre "
                                  "basada en las fuentes proporcionadas")
    assert "99%" in fallback_message(0.99)


# ---- claim parsing -----------------------------------------------------------------------

@pytest.mark.parametrize("answer,expected", [
    ("A es B. [1] C es D [2][3].", [("A es B.", [1]), ("C es D.", [2, 3])]),
    ("A es B [1].", [("A es B.", [1])]),
    ("Sin cita.", [("Sin cita.", [])]),
    ("Uno. [1]\nDos. [2]", [("Uno.", [1]), ("Dos.", [2])]),
    ("", []),
    ("[1]", []),
])
def test_parse_claims(answer, expected):
    assert parse_claims(answer) == expected


# ---- NLIVerifier (cited, with coordinates) -----------------------------------------------

def test_supported_claim_reports_exact_evidence_coordinates():
    text = CHUNKS[0]
    c = ctx(text, start=250, page=3)
    [v] = NLIVerifier().verify("Los registros de acceso se eliminan a los 90 días. [1]", [c])
    assert v.supported and v.score == 1.0
    ev = v.evidence
    assert ev.text == "Los registros de acceso se eliminan a los 90 días."
    assert (ev.doc_id, ev.page) == ("doc.md", 3)
    assert text[ev.start - 250 : ev.end - 250] == ev.text  # absolute offsets map back to the source


def test_claim_must_be_supported_by_the_passage_it_cites():
    contexts = [ctx(CHUNKS[0], 1), ctx(CHUNKS[1], 2)]
    [v] = NLIVerifier().verify("Los datos en reposo se cifran con AES-256. [1]", contexts)  # true, but cited wrongly
    assert not v.supported and v.evidence is None
    [ok] = NLIVerifier().verify("Los datos en reposo se cifran con AES-256. [2]", contexts)
    assert ok.supported


def test_missing_and_unknown_citations_are_rejected():
    c = ctx(CHUNKS[0])
    no_cite, bad_cite = NLIVerifier().verify("Los registros se conservan durante 7 años. Otra cosa. [9]", [c])
    assert (no_cite.supported, no_cite.reason) == (False, "no citation")
    assert not bad_cite.supported and "unknown citation" in bad_cite.reason


def test_threshold_is_inclusive_and_configurable():
    class Fixed:
        def __init__(self, v):
            self.v = v

        def entailment(self, pairs):
            return [self.v] * len(pairs)

    c = ctx(CHUNKS[0])
    assert NLIVerifier(Fixed(0.95), 0.95).verify("x y. [1]", [c])[0].supported
    assert not NLIVerifier(Fixed(0.9499), 0.95).verify("x y. [1]", [c])[0].supported
    assert NLIVerifier(Fixed(0.8), 0.7).verify("x y. [1]", [c])[0].supported


def test_multi_sentence_evidence_window():
    c = ctx("El sistema usa Postgres. Guarda los logs de auditoría ahí.")
    [v] = NLIVerifier().verify("El sistema usa Postgres y guarda los logs de auditoría. [1]", [c])
    assert v.supported and v.evidence.text.count(".") == 2


def test_deberta_wrapper_is_importable_and_requires_extras():
    from contextrag.verification.nli_verifier import DebertaNLI

    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    assert DebertaNLI  # model weights are not downloaded in CI


def test_evidence_is_the_window_most_similar_to_the_claim_even_when_nli_scores_saturate():
    class Saturated:  # a model that is ~certain about everything: every window "entails" the claim
        def entailment(self, pairs):
            return [0.9999] * len(pairs)

    text = "Retención de datos. Los registros de auditoría se conservan durante 7 años. Fin."
    [v] = NLIVerifier(Saturated()).verify("Los registros de auditoría se conservan durante 7 años. [1]", [ctx(text)])
    assert v.supported and v.evidence.text == "Los registros de auditoría se conservan durante 7 años."


def test_evidence_choice_never_changes_the_support_decision():
    class Graded:  # high score only for windows containing "siete"; everything else just below the threshold
        def entailment(self, pairs):
            return [0.99 if "siete" in p else 0.9 for p, _ in pairs]

    c = ctx("Hay siete cosas. Otra frase. Y una más.")
    [ok] = NLIVerifier(Graded()).verify("Hay siete cosas. [1]", [c])
    assert ok.supported and "siete" in ok.evidence.text and ok.score == 0.99
    [bad] = NLIVerifier(Graded()).verify("Otra frase. [1]", [ctx("Otra frase. Y una más.")])
    assert not bad.supported and bad.evidence is None and bad.score == 0.9
