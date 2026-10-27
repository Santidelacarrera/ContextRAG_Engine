"""Sentence splitting around company-form suffixes, which are everywhere in Spanish legal text."""
import pytest

from contextrag.core.text import split_sentences


def sentences(text):
    return [s for s, _, _ in split_sentences(text)]


@pytest.mark.parametrize("text, expected", [
    ("El capital social de Industrias Valdivia S.A. es de 898 millones de pesos. Fue constituida en 1990.",
     ["El capital social de Industrias Valdivia S.A. es de 898 millones de pesos.", "Fue constituida en 1990."]),
    ("Comercial Pehuén Ltda. tiene su domicilio en Temuco.", ["Comercial Pehuén Ltda. tiene su domicilio en Temuco."]),
    ("Transportes Austral SpA. 2026 fue un buen año.", ["Transportes Austral SpA. 2026 fue un buen año."]),
    ("Contrató a Alfa S.A. El plazo vence en mayo.", ["Contrató a Alfa S.A.", "El plazo vence en mayo."]),
    ("La sociedad es Beta Ltda. \"El domicilio\" es Talca.", ["La sociedad es Beta Ltda.", "\"El domicilio\" es Talca."]),
    ("El plazo es de cuatro años. El plazo corre desde la notificación.",
     ["El plazo es de cuatro años.", "El plazo corre desde la notificación."]),
    ("Dr. Pérez firmó. Luego se retiró.", ["Dr. Pérez firmó.", "Luego se retiró."]),
])
def test_company_suffixes_do_not_split_a_sentence_that_continues(text, expected):
    assert sentences(text) == expected


def test_offsets_still_point_at_the_original_text():
    text = "El capital de Alfa S.A. es alto. Otra frase."
    for sentence, start, end in split_sentences(text):
        assert text[start:end] == sentence


# ---- the same rule in claim parsing and in the NLI analysis ------------------------------------------

def test_claim_parsing_keeps_a_company_name_inside_its_claim():
    from contextrag.verification.nli_verifier import parse_claims

    claims = parse_claims("Industrias Valdivia S.A. tiene su domicilio en Valparaíso. [1]")
    assert claims == [("Industrias Valdivia S.A. tiene su domicilio en Valparaíso.", [1])]
    two = parse_claims("Contrató a Alfa S.A. El plazo vence en mayo. [2]")
    assert [c for c, _ in two] == ["Contrató a Alfa S.A.", "El plazo vence en mayo."]


def test_a_claim_with_a_company_suffix_is_supported_by_its_own_sentence():
    from contextrag.verification.nli_verifier import LexicalEntailment, verify_attribution

    premise = "Industrias Valdivia S.A. tiene su domicilio en Valparaíso."
    nli = LexicalEntailment()
    assert verify_attribution("Industrias Valdivia S.A. tiene su domicilio en Valparaíso.", [premise], nli) == 1.0
    wrong = verify_attribution("Industrias Valdivia S.A. tiene su domicilio en Santiago.", [premise], nli)
    assert wrong < 0.95  # a partial lexical overlap, below the verification threshold, so it is rejected


def test_protect_corporate_suffixes_rewrites_only_what_continues():
    from contextrag.core.text import protect_corporate_suffixes

    assert protect_corporate_suffixes("Valdivia S.A. es grande. Ltda. de 5") == "Valdivia SA es grande. Ltda de 5"
    assert protect_corporate_suffixes("Contrató a Alfa S.A. El plazo") == "Contrató a Alfa S.A. El plazo"
    assert protect_corporate_suffixes("fin de la frase.") == "fin de la frase."


# ---- a sentence is always fully supported by itself --------------------------------------------------

SENTENCES = [
    "Sociedad Reservada S.A. (en adelante, «SR») tiene su domicilio en Osorno.",
    "Industrias Valdivia S.A. (también conocido como IVSA) es una sociedad anónima cerrada.",
    "El contrato, firmado el 5 de mayo de 2026, vence en junio.",
    "Comercial Pehuén Ltda. (la \"Compañía\") pagará 120 millones de pesos.",
    "El Juzgado Tercero de Letras fijó la audiencia el 9 de junio de 2026.",
    "Transportes Austral SpA. 2026 fue un buen año para la empresa.",
    "El plazo de prescripción de la acción es de cuatro años contados desde la notificación.",
    "Según el artículo 5 (inciso 2), la parte demandada debe comparecer.",
]


@pytest.mark.parametrize("sentence", SENTENCES)
def test_an_exact_sentence_is_fully_supported_by_itself(sentence):
    from contextrag.verification.nli_verifier import LexicalEntailment, verify_attribution

    assert verify_attribution(sentence, [sentence], LexicalEntailment()) == 1.0
