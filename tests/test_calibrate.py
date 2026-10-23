"""Calibration: selection policies, insufficient-data flags, CLI exit codes, and no writes beyond the requested paths."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from contextrag.eval import calibrate as C
from contextrag.eval.context_metrics import ContextReport
from contextrag.eval.long_corpus import build_long_corpus

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CASES = ROOT / "contextrag" / "eval" / "data" / "verifier_cases.json"


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "contextrag", *args], cwd=cwd or ROOT, capture_output=True,
                          text=True, timeout=600)


# ---- chunk grid -------------------------------------------------------------------------------------

def test_the_chunk_policy_is_recall_first_so_a_lost_fact_cannot_be_traded_for_ranking(monkeypatch):
    fake = {(1024, 256): (1.0, 0.76), (1024, 400): (0.94, 0.72), (2000, 256): (0.94, 0.84), (2000, 400): (0.94, 0.875)}

    def fake_eval(make_rag, data, tenant="x"):
        make_rag()  # the real function builds a store; the stub only needs the configuration it was given
        return ContextReport(total=18, mean_recall=fake[CURRENT["cfg"]][0],
                             mean_average_precision=fake[CURRENT["cfg"]][1], per_case={})

    CURRENT = {}
    import contextrag.factory as factory

    real_offline = factory.offline

    def spying_offline(cfg=None, llm=None):
        CURRENT["cfg"] = (cfg.parent_tokens, cfg.child_tokens)
        return real_offline(cfg)

    monkeypatch.setattr(factory, "offline", spying_offline)
    monkeypatch.setattr(C, "evaluate_context", fake_eval)
    rows, section = C.chunk_grid({"documents": [], "cases": []})
    assert len(rows) == 4
    # 2000/400 has the best AP but loses recall, so the recall-first policy keeps 1024/256.
    assert section["recommended"] == {"parent_tokens": 1024, "child_tokens": 256}


def test_the_chunk_grid_on_the_synthetic_corpus_reproduces_the_regression_baseline():
    import json as _json

    baseline = _json.loads((ROOT / "contextrag" / "eval" / "data" / "context_baseline.json").read_text())
    rows, section = C.chunk_grid(build_long_corpus(seed=7))
    default = next(r for r in rows if (r.parent_tokens, r.child_tokens) == (1024, 256))
    assert default.mean_recall == pytest.approx(baseline["metrics"]["mean_recall"])
    assert default.mean_average_precision == pytest.approx(baseline["metrics"]["mean_average_precision"], abs=1e-4)
    assert section["recommended"] == {"parent_tokens": 1024, "child_tokens": 256}


# ---- thresholds --------------------------------------------------------------------------------------

def test_precision_and_recall_at_a_threshold_are_counted_exactly():
    scored = [(0.9, True), (0.8, False), (0.4, True), (0.2, False)]
    assert C._precision_recall(scored, 0.5) == (1, 1, 1)  # tp, fp, fn
    assert C._precision_recall(scored, 0.0) == (2, 2, 0)


def test_a_small_labelled_set_is_flagged_insufficient_not_presented_as_a_threshold():
    labels = json.loads(FIXTURE_CASES.read_text(encoding="utf-8"))
    section = C.nli_threshold(labels)
    assert section.status == "insufficient_data"
    assert section.data["labelled"] == {"supported": 13, "unsupported": 43}
    assert "lexical" in section.data["nli_model"]


def test_nli_without_labels_is_not_run_rather_than_guessed():
    assert C.nli_threshold([]).status == "not_run"


def test_rerank_calibration_reports_its_counts_and_flags_the_small_sample():
    data = build_long_corpus(seed=7)
    section = C.rerank_threshold(data)
    assert section.status == "insufficient_data"  # a handful of relevant candidates cannot support a floor
    assert section.data["relevant"] + section.data["irrelevant"] == section.data["candidates"]
    assert all(0.0 <= r["precision"] <= 1.0 and 0.0 <= r["recall"] <= 1.0 for r in section.data["rows"])


def test_a_target_precision_outside_the_unit_interval_is_refused():
    with pytest.raises(C.CalibrationError, match="target_precision"):
        C.rerank_threshold(build_long_corpus(seed=7), target_precision=0.0)


def test_cases_without_a_labelled_document_are_refused_for_rerank_calibration():
    data = {"documents": [{"doc_id": "d", "text": "uno dos tres"}],
            "cases": [{"id": "c", "question": "uno", "answerable": True, "expect": "uno"}]}
    with pytest.raises(C.CalibrationError, match="labelled document"):
        C.rerank_threshold(data)


# ---- report and proposal -------------------------------------------------------------------------------

def test_the_report_states_its_evidence_level_seed_and_what_it_does_not_show():
    data = build_long_corpus(seed=7)
    report = C.calibrate(data, verifier_cases=None, data_kind="synthetic", data_path=None, seed=7)
    assert report["data"]["kind"] == "synthetic" and report["data"]["seed"] == 7
    assert "not evidence about production documents" in report["claims"]
    assert report["nli_threshold"]["status"] == "not_run"


def test_an_unknown_data_kind_is_refused():
    with pytest.raises(C.CalibrationError, match="data_kind"):
        C.calibrate(build_long_corpus(seed=7), verifier_cases=None, data_kind="production", data_path=None, seed=7)


def test_the_proposal_carries_candidates_and_their_statuses(tmp_path):
    report = C.calibrate(build_long_corpus(seed=7), verifier_cases=None, data_kind="synthetic", data_path=None, seed=7)
    out = tmp_path / "proposal.json"
    C.write_proposal(report, out)
    proposal = json.loads(out.read_text())
    assert proposal["NOTE"].startswith("candidates for review")
    assert proposal["statuses"]["rerank"] == "insufficient_data"
    assert proposal["RERANK_MIN_SCORE"] is None


# ---- CLI: exit codes and what is written -------------------------------------------------------------

def test_calibrate_needs_exactly_one_data_source(tmp_path):
    r = cli("calibrate", "--out", str(tmp_path / "r.json"))
    assert r.returncode == 2 and "exactly one" in r.stderr
    r = cli("calibrate", "--synthetic", "--cases", str(FIXTURE_CASES), "--out", str(tmp_path / "r.json"))
    assert r.returncode == 2


def test_a_missing_cases_file_is_a_fatal_error_with_no_output(tmp_path):
    out = tmp_path / "r.json"
    r = cli("calibrate", "--cases", str(tmp_path / "nope.json"), "--out", str(out))
    assert r.returncode == 2 and not out.exists()


def test_the_synthetic_run_writes_only_the_requested_files(tmp_path):
    out, proposal = tmp_path / "report.json", tmp_path / "proposal.json"
    before = {p.name for p in ROOT.iterdir()}
    r = cli("calibrate", "--synthetic", "--verifier-cases", str(FIXTURE_CASES), "--out", str(out),
            "--proposal", str(proposal))
    assert r.returncode == 0, r.stderr
    assert {p.name for p in ROOT.iterdir()} == before  # no stray files in the repository
    report = json.loads(out.read_text())
    assert report["data"]["kind"] == "synthetic"
    assert report["nli_threshold"]["status"] == "insufficient_data"
    assert proposal.exists()


def test_without_the_proposal_flag_no_candidate_file_is_written(tmp_path):
    out = tmp_path / "report.json"
    assert cli("calibrate", "--synthetic", "--out", str(out)).returncode == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_the_bundled_fixture_is_labelled_as_a_fixture_not_as_supplied_data(tmp_path):
    out = tmp_path / "report.json"
    r = cli("calibrate", "--cases", str(ROOT / "contextrag" / "eval" / "data" / "rag_cases.json"), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["data"]["kind"] == "fixture"


# ---- RAGAS CLI -------------------------------------------------------------------------------------------

def test_the_ragas_command_refuses_to_run_without_a_configured_judge(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "contextrag", "ragas", "--cases", str(FIXTURE_CASES),
                        "--out", str(tmp_path / "r.json")], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 2 and "RAGAS_PROVIDER" in r.stderr
    assert not (tmp_path / "r.json").exists()
