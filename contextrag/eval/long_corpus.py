"""Deterministic synthetic corpus of long documents, for comparing chunking and budget settings.

Each document is long filler, with facts about several entities. Facts share the attribute vocabulary with their
distractors (same attribute, different entity, nearly the same wording), so a retriever has to match the entity,
not just the topic. Questions name one entity and one attribute; the reference is the value.

This measures lexical retrieval behaviour on templated text. It says nothing about real documents, where the
parameters that matter (vocabulary, structure, answer length) differ.
"""
from __future__ import annotations

import random

ATTRIBUTES = {
    "plazo de retención": ["90 días", "2 años", "7 años", "18 meses", "30 días"],
    "responsable": ["Ana Ruiz", "Luis Mena", "Eva Soto", "Marta Pino", "Hugo Vera"],
    "presupuesto anual": ["120 mil euros", "45 mil euros", "300 mil euros", "78 mil euros", "9 mil euros"],
}
ENTITIES = ["Atlas", "Boreal", "Cénit", "Delta", "Éter", "Fénix", "Gaia", "Hélice", "Índigo", "Jade"]
FILLER = (
    "La organización revisa periódicamente los procedimientos internos y documenta cada cambio en el registro común. "
    "Los equipos coordinan reuniones trimestrales para evaluar avances, riesgos y prioridades del calendario. "
    "Cada informe incluye métricas de uso, incidencias abiertas y acciones correctivas acordadas con los responsables. "
    "El acceso a los sistemas se concede según el rol y se revisa cuando cambian las funciones de la persona."
)


def build_long_corpus(n_docs: int = 6, paragraphs: int = 60, facts_per_doc: int = 3, seed: int = 7):
    """Return ``{"documents": [...], "cases": [...]}`` in the same shape as ``rag_cases.json``."""
    # seeded on purpose (reproducible evaluation data, not security-sensitive)
    rng = random.Random(seed)  # nosec B311
    documents, cases = [], []
    for d in range(n_docs):
        doc_id = f"informe-{d:02d}.md"
        chosen = rng.sample(ENTITIES, facts_per_doc)
        blocks = [f"# Informe {d:02d}"]
        facts = []
        for name in chosen:
            attr = rng.choice(list(ATTRIBUTES))
            value = rng.choice(ATTRIBUTES[attr])
            facts.append((name, attr, value))
        fact_at = {rng.randrange(paragraphs): f for f in facts}
        for p in range(paragraphs):
            blocks.append(f"## Sección {p}\n\n{FILLER}")
            if p in fact_at:
                name, attr, value = fact_at[p]
                blocks.append(f"El {attr} del Proyecto {name} es {value}, según la última revisión aprobada.")
            # a distractor: same attribute, another entity, different value, near-identical wording
            other = rng.choice([e for e in ENTITIES if e not in {f[0] for f in facts}])
            attr_d = rng.choice(list(ATTRIBUTES))
            blocks.append(f"El {attr_d} del Proyecto {other} es {rng.choice(ATTRIBUTES[attr_d])}, "
                          "según una revisión anterior.")
        documents.append({"doc_id": doc_id, "text": "\n\n".join(blocks)})
        for i, (name, attr, value) in enumerate(facts):
            cases.append({"id": f"{d:02d}-{i}", "question": f"¿Cuál es el {attr} del Proyecto {name}?",
                          "answerable": True, "expect": value, "doc": doc_id})
    return {"documents": documents, "cases": cases}
