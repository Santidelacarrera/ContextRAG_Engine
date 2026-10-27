"""The load benchmark itself: it runs, it is honest about what it is, and a load test finds no errors or mixed-up data."""
import json

import pytest

from contextrag.__main__ import main
from contextrag.eval.load_bench import CAVEAT, run_load


@pytest.fixture(scope="module")
def report():
    return run_load(levels=(1, 4), requests=120, seed=7, mixed_ingest_every=10)


def test_every_request_completes_without_errors_at_every_level(report):
    for row in report["read_only"] + report["mixed_read_write"]["levels"]:
        assert row["outcomes"]["error"] == 0
        assert sum(row["outcomes"].values()) == row["requests"] - (row["ingests"]["n"] if row["ingests"] else 0)
    assert all(row["ingests"]["failed"] == 0 for row in report["mixed_read_write"]["levels"])


def test_the_report_does_not_claim_capacity_and_lists_what_it_did_not_measure(report):
    assert report["caveat"] == CAVEAT and "not the capacity of a deployment" in report["caveat"]
    assert "not evidence about production models" in report["caveat"]
    assert any("HTTP" in item for item in report["not_measured"]) and any("real models" in item for item in report["not_measured"])


def test_percentiles_follow_the_sample_size_rule(report):
    lat = report["read_only"][0]["latency"]
    assert lat["n"] >= 100 and lat["p99_ms"] is not None  # 120 samples: enough to report p99 (still a small sample)
    small = run_load(levels=(1,), requests=30, mixed_ingest_every=0)
    assert small["read_only"][0]["latency"]["p99_ms"] is None  # 30 samples: refused rather than reported
    assert small["mixed_read_write"] is None


def test_the_query_plan_is_reproducible_from_the_seed():
    a = run_load(levels=(1,), requests=40, seed=3, mixed_ingest_every=0)
    b = run_load(levels=(1,), requests=40, seed=3, mixed_ingest_every=0)
    assert a["read_only"][0]["outcomes"] == b["read_only"][0]["outcomes"]  # same questions, same verdicts
    assert a["dataset"] == b["dataset"]


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        run_load(levels=(0,), requests=10)
    with pytest.raises(ValueError):
        run_load(levels=(), requests=10)
    with pytest.raises(ValueError):
        run_load(levels=(1,), requests=0)


def test_the_cli_writes_a_report_and_rejects_bad_input(tmp_path, capsys):
    out = tmp_path / "load.json"
    assert main(["loadtest", "--levels", "1", "--requests", "30", "--mixed-ingest-every", "0", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["read_only"][0]["concurrency"] == 1
    assert "not the capacity of a deployment" in capsys.readouterr().out
    assert main(["loadtest", "--levels", "0", "--requests", "10"]) == 2
