"""Synthetic corpus for entity handling (version 1): names, documented aliases, homonyms, decoys and unknown entities.

Everything is fictional. Expected outcomes:

* ``answer``: the documents support an answer; the expected value and the source document are known.
* ``abstain``: no document supports an answer (unknown entity, a near-name decoy, or an entity the caller may not see).
* ``clarify``: the name is documented for two different entities, so the question needs a clarification.

Kinds: ``canonical`` (the documented name), ``alias_other_doc`` (alias declared in a document other than the one holding
the fact), ``alias_same_doc``, ``unknown``, ``decoy`` (a name similar to a real one, which must not be merged with it),
``decoy_control`` (the real name next to the decoys), ``ambiguous``, ``restricted_denied`` and ``restricted_allowed``.

Kinds starting with ``limit_`` are cases the design is known not to cover. They are reported separately and never mixed
into the headline rates, so the headline cannot look better than the system is:

* ``limit_alias_form``: an alias written in a form the declaration patterns do not read.
* ``limit_homonym``: one name for two entities that no document declares as such.
* ``limit_attribute``: the entity exists but the attribute asked about is not documented.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

VERSION = 1
SEED = 23
DATA_PATH = Path(__file__).parent / "data" / "entity_bench_v1.json"

ENTITIES = [("Industrias Valdivia S.A.", "IVSA"), ("Comercial Pehuén Ltda.", "Pehuén"),
            ("Agrícola Los Robles S.A.", "Los Robles"), ("Transportes Austral SpA", "TASA"),
            ("Inmobiliaria Cordillera S.A.", "Cordillera"), ("Servicios Mapocho Ltda.", "Mapocho"),
            ("Pesquera Caleta Azul S.A.", "Caleta Azul"), ("Constructora Andes Sur SpA", "CAS")]
CITIES = ["Valparaíso", "Concepción", "Temuco", "La Serena", "Rancagua", "Talca", "Antofagasta", "Osorno"]
UNKNOWN = ["Metalúrgica Quillota Ltda.", "Panificadora Ñuñoa S.A.", "Textiles Arica SpA", "Vitivinícola Colchagua S.A.",
           "Aserraderos Chiloé Ltda.", "Minera Copiapó SpA", "Astilleros Talcahuano S.A.", "Cervecería Valdés Ltda."]
DECOYS = ["Halvorsen Corp", "Halvorsen Inc", "Halverson Group", "Halvorsen Grupo", "Halvorsen Group Chile",
          "Halvorsen Holdings"]


def build_entity_bench(seed: int = SEED) -> dict:
    # seeded on purpose (reproducible evaluation data, not security-sensitive)
    rng = random.Random(seed)  # nosec B311
    documents: list[dict] = []
    facts: dict[str, dict] = {}
    for i, (canon, alias) in enumerate(ENTITIES):
        city, amount = CITIES[i], f"{rng.randrange(100, 990)} millones"
        facts[canon] = {"city": city, "amount": amount, "doc": f"ent_{i}.md"}
        body = (f"# {canon}\n\n{canon} tiene su domicilio en {city}.\n\n"
                f"El capital social de {canon} es de {amount} de pesos.\n")
        if i >= 4:  # the alias is declared in the document that holds the facts
            body += f"\n{canon} (también conocido como {alias}) es una sociedad anónima cerrada.\n"
        documents.append({"doc_id": f"ent_{i}.md", "text": body, "acl": []})
        if i < 4:  # the alias is declared in another document
            documents.append({"doc_id": f"contrato_{i}.md", "acl": [], "text": (
                f"# Contrato marco {i}\n\nEn el contrato marco de servicios, {canon} (en adelante, «{alias}») se obliga a "
                "prestar los servicios descritos en el anexo técnico dentro de los plazos pactados.\n")})
    for name, city, budget in (("Proyecto Atlas Norte", "Temuco", "120 millones"), ("Proyecto Atlas Sur", "Puerto Montt", "340 millones")):
        key = name.split()[-1].lower()
        documents.append({"doc_id": f"atlas_{key}.md", "acl": [], "text": (
            f"# {name}\n\n{name} (también conocido como Atlas) tiene su sede en {city}.\n\n"
            f"El presupuesto del {name} es de {budget} de pesos.\n")})
    documents.append({"doc_id": "halvorsen.md", "acl": [], "text": (
        "# Halvorsen Group\n\nHalvorsen Group tiene su domicilio en Valdivia.\n\n"
        "El capital social de Halvorsen Group es de 900 millones de pesos.\n")})
    documents.append({"doc_id": "astilleros.md", "acl": [], "text": (
        "# Astilleros Magallanes Ltda.\n\nAstilleros Magallanes Ltda. tiene su domicilio en Punta Arenas.\n")})
    documents.append({"doc_id": "prensa.md", "acl": [], "text": (
        "# Nota de prensa\n\nLa Naviera, nombre con que todos conocen a Astilleros Magallanes Ltda., anunció una "
        "ampliación de sus astilleros durante el próximo año.\n")})
    for suffix, city, line in (("a", "Arica", "transportes marítimos"), ("b", "Iquique", "seguros generales")):
        documents.append({"doc_id": f"mercurio_{suffix}.md", "acl": [], "text": (
            f"# Mercurio S.A.\n\nMercurio S.A. se dedica a {line}. Mercurio S.A. tiene su domicilio en {city}.\n")})
    documents.append({"doc_id": "reservado.md", "acl": ["caso-x"], "text": (
        "# Sociedad Reservada S.A.\n\nSociedad Reservada S.A. (en adelante, «SR») tiene su domicilio en Osorno.\n\n"
        "El capital social de Sociedad Reservada S.A. es de 50 millones de pesos.\n")})

    q: list[dict] = []

    def add(kind, question, expected, expect=None, doc=None, groups=()):
        q.append({"id": f"{kind[:3]}{len(q) + 1:02d}", "kind": kind, "question": question, "expected": expected,
                  "expect": expect, "relevant_doc": doc, "groups": list(groups)})

    for canon, _ in ENTITIES:
        f = facts[canon]
        add("canonical", f"¿Cuál es el domicilio de {canon}?", "answer", f["city"], f["doc"])
    for canon, alias in ENTITIES[:4]:
        f = facts[canon]
        add("alias_other_doc", f"¿Cuál es el capital social de {alias}?", "answer", f["amount"], f["doc"])
    for canon, alias in ENTITIES[4:]:
        f = facts[canon]
        add("alias_same_doc", f"¿Dónde tiene su domicilio {alias}?", "answer", f["city"], f["doc"])
    for name in UNKNOWN:
        add("unknown", f"¿Cuál es el domicilio de {name}?", "abstain")
    for name in DECOYS:
        add("decoy", f"¿Cuál es el capital social de {name}?", "abstain")
    add("decoy_control", "¿Cuál es el capital social de Halvorsen Group?", "answer", "900 millones", "halvorsen.md")
    add("decoy_control", "¿Dónde tiene su domicilio Halvorsen Group?", "answer", "Valdivia", "halvorsen.md")
    for question in ("¿Dónde tiene su sede Atlas?", "¿Cuál es el presupuesto de Atlas?", "¿Qué presupuesto tiene Atlas?",
                     "¿En qué ciudad está Atlas?", "¿Cuánto cuesta Atlas?"):
        add("ambiguous", question, "clarify")
    for question in ("¿Dónde tiene su domicilio SR?", "¿Cuál es el capital social de Sociedad Reservada S.A.?",
                     "¿Cuál es el domicilio de SR?"):
        add("restricted_denied", question, "abstain")
    add("restricted_allowed", "¿Dónde tiene su domicilio SR?", "answer", "Osorno", "reservado.md", groups=("caso-x",))
    add("restricted_allowed", "¿Cuál es el capital social de Sociedad Reservada S.A.?", "answer", "50 millones",
        "reservado.md", groups=("caso-x",))
    add("limit_alias_form", "¿Dónde tiene su domicilio la Naviera?", "answer", "Punta Arenas", "astilleros.md")
    add("limit_homonym", "¿Dónde tiene su domicilio Mercurio S.A.?", "clarify")
    add("limit_attribute", "¿Cuál es el RUT de Industrias Valdivia S.A.?", "abstain")
    add("limit_attribute", "¿Quién es el gerente general de Comercial Pehuén Ltda.?", "abstain")
    return {"version": VERSION, "seed": seed, "synthetic": True, "documents": documents, "queries": q}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = json.dumps(build_entity_bench(), indent=1, ensure_ascii=False) + "\n"
    if args.write:
        DATA_PATH.write_text(text, encoding="utf-8")
        return 0
    if args.check:
        same = DATA_PATH.exists() and DATA_PATH.read_text(encoding="utf-8") == text
        if not same:
            print("entity_bench_v1.json differs from build_entity_bench(); regenerate with --write", file=sys.stderr)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
