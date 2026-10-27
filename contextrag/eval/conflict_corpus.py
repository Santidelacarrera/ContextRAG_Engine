"""Labelled pairs of statements for conflict detection (version 1). Synthetic; no real document.

Each case is two statements about one question, with the label a careful reader would give:

* ``contradiction``: they cannot both be true (a different date, amount, status, party, or a negation of the same fact).
* ``version``: the same fact at two dates (an old and a new value). Treated as a conflict that needs review, because the
  system cannot know which version is in force.
* ``compatible``: the same fact worded differently, or one statement adds detail. NOT a conflict.
* ``complementary``: different facts about the same subject. NOT a conflict.
* ``homonym``: the same name for two different entities. NOT a conflict; the system should ask which is meant.
* ``insufficient``: one statement says nothing about the question. NOT a conflict, and nothing can be concluded.

Labels are assigned by the author, not by the detector under test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERSION = 1
DATA_PATH = Path(__file__).parent / "data" / "conflict_bench_v1.json"
CONFLICT_LABELS = ("contradiction", "version")
NON_CONFLICT_LABELS = ("compatible", "complementary", "homonym", "insufficient")


def build_conflict_bench() -> dict:
    cases: list[dict] = []

    def add(label, question, a, b, note=""):
        cases.append({"id": f"{label[:4]}{len(cases) + 1:02d}", "label": label, "question": question,
                      "a": a, "b": b, "note": note})

    q_hearing = "¿Cuándo es la audiencia preliminar del expediente Alfa?"
    add("contradiction", q_hearing, "La audiencia preliminar del expediente Alfa se fijó para el 5 de mayo de 2026.",
        "La audiencia preliminar del expediente Alfa se fijó para el 9 de junio de 2026.", "date")
    add("contradiction", "¿Cuál es el capital social de Industrias Valdivia S.A.?",
        "El capital social de Industrias Valdivia S.A. es de 120 millones de pesos.",
        "El capital social de Industrias Valdivia S.A. es de 150 millones de pesos.", "amount")
    add("contradiction", "¿Cuál es el monto de la indemnización del caso Beta?",
        "La indemnización del caso Beta asciende a 80 millones de pesos.",
        "La indemnización del caso Beta asciende a 95 millones de pesos.", "amount")
    add("contradiction", "¿Quién es el demandante del expediente Gamma?",
        "El demandante del expediente Gamma es Pedro Soto.", "El demandante del expediente Gamma es Juan Pérez.", "party")
    add("contradiction", "¿Fue notificada la sentencia del expediente Delta?",
        "La sentencia del expediente Delta fue notificada a la demandada.",
        "La sentencia del expediente Delta no fue notificada a la demandada.", "negation")
    add("contradiction", "¿Está vigente el contrato Epsilon?",
        "El contrato Epsilon se encuentra vigente al día de hoy.",
        "El contrato Epsilon se encuentra terminado al día de hoy.", "status")
    add("contradiction", "¿Cuál es el plazo de cierre del proyecto Zeta?",
        "El plazo de cierre del proyecto Zeta vence el 1 de marzo de 2027.",
        "El plazo de cierre del proyecto Zeta vence el 15 de marzo de 2027.", "date")
    add("contradiction", "¿Cuántas cuotas tiene el acuerdo Eta?",
        "El acuerdo Eta se paga en 6 cuotas mensuales.", "El acuerdo Eta se paga en 12 cuotas mensuales.", "number")
    add("contradiction", "¿Quién firmó el contrato Theta?",
        "El contrato Theta fue firmado por la notario Marta Rojas.",
        "El contrato Theta no fue firmado por la notario Marta Rojas.", "negation")

    add("version", "¿Cuál es el plazo de prescripción de la acción del reglamento interno?",
        "Versión 2019: el plazo de prescripción de la acción es de tres años.",
        "Versión 2024: el plazo de prescripción de la acción es de cinco años.", "old vs new figure")
    add("version", "¿Cuál es la tarifa de honorarios del servicio Iota?",
        "Tarifa vigente hasta 2023: 90 UF por hora del servicio Iota.",
        "Tarifa vigente desde 2024: 110 UF por hora del servicio Iota.", "old vs new figure")

    add("compatible", q_hearing,
        "La audiencia preliminar del expediente Alfa se fijó para el 5 de mayo de 2026.",
        "El expediente Alfa tiene su audiencia preliminar fijada el 05/05/2026.", "same date, other format and wording")
    add("compatible", "¿Cuál es el capital social de Industrias Valdivia S.A.?",
        "El capital social de Industrias Valdivia S.A. es de 120 millones de pesos.",
        "El capital social de Industrias Valdivia S.A. es de 120.000.000 pesos.", "same amount, other notation")
    add("compatible", q_hearing,
        "La audiencia preliminar del expediente Alfa se fijó para el 5 de mayo de 2026.",
        "La audiencia preliminar del expediente Alfa se fijó para el 5 de mayo de 2026 a las 10:00 horas.",
        "same date with extra detail")
    add("compatible", "¿Está vigente el contrato Kappa?",
        "El contrato Kappa se encuentra vigente al día de hoy.",
        "A la fecha, el contrato Kappa está vigente.", "same status, reworded")

    add("complementary", "¿Qué fechas tiene el contrato del expediente Lambda?",
        "El contrato del expediente Lambda se firmó el 5 de mayo de 2026.",
        "El contrato del expediente Lambda vence el 5 de mayo de 2029.", "signing and expiry: different facts")
    add("complementary", "¿Cuál es el domicilio de la sociedad Mu?",
        "La sociedad Mu tiene su domicilio en Valparaíso.",
        "La sociedad Mu tiene como gerente general a Ana Díaz.", "address and manager")
    add("complementary", "¿Cuánto cobra el servicio Nu?",
        "El servicio Nu cobra 90 UF por hora.", "El servicio Nu incluye soporte telefónico de lunes a viernes.",
        "price and scope")

    add("homonym", "¿Dónde tiene su domicilio Halvorsen?",
        "Halvorsen Group tiene su domicilio en Valdivia.",
        "Halvorsen Corp tiene su domicilio en Temuco.", "two entities, same first word: not a conflict")
    add("homonym", "¿Cuál es el capital social de Atlas?",
        "Proyecto Atlas Norte tiene un capital social de 120 millones de pesos.",
        "Proyecto Atlas Sur tiene un capital social de 340 millones de pesos.", "a name shared by two projects")

    add("insufficient", q_hearing,
        "La audiencia preliminar del expediente Alfa se fijó para el 5 de mayo de 2026.",
        "El Juzgado Tercero de Letras tiene su sede en Temuco.", "one statement is silent on the question")
    add("insufficient", "¿Cuál es el capital social de Industrias Valdivia S.A.?",
        "El capital social de Industrias Valdivia S.A. es de 120 millones de pesos.",
        "Industrias Valdivia S.A. se dedica a la fabricación de muebles.", "one statement is silent on the figure")
    return {"version": VERSION, "synthetic": True, "cases": cases}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = json.dumps(build_conflict_bench(), indent=1, ensure_ascii=False) + "\n"
    if args.write:
        DATA_PATH.write_text(text, encoding="utf-8")
        return 0
    if args.check:
        same = DATA_PATH.exists() and DATA_PATH.read_text(encoding="utf-8") == text
        if not same:
            print("conflict_bench_v1.json differs from build_conflict_bench(); regenerate with --write", file=sys.stderr)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
