"""The release gate: one candidate fails, one passes, a threshold edit flips the outcome."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import gate

ROOT = Path(__file__).resolve().parents[2]
BASE = {"recall_at_5": 0.90, "faithfulness": 0.88, "citation_validity": 1.0,
        "latency_p95": 12.0, "abstention_correctness": 1.0}


def write(path: Path, metrics: dict[str, float]) -> Path:
    path.write_text(json.dumps({"meta": {"collection": "x"}, "metrics": metrics}))
    return path


def cfg(tmp: Path, **thresholds: object) -> Path:
    path = tmp / "gate.yaml"
    path.write_text(yaml.safe_dump({"thresholds": thresholds}))
    return path


def run(tmp: Path, cand: dict[str, float], config: Path, base: dict[str, float] | None = BASE) -> int:
    results = tmp / "results"
    results.mkdir(exist_ok=True)
    write(results / "run1.json", cand)
    args = ["--results", str(results), "--config", str(config), "--no-decision-file"]
    if base is not None:
        args += ["--baseline", str(write(tmp / "baseline.json", base))]
    else:
        args += ["--baseline", str(tmp / "missing.json"), "--allow-missing-baseline"]
    return gate.main(args)


def test_passing_candidate(tmp_path: Path) -> None:
    good = dict(BASE, recall_at_5=0.89)                      # drop of one point
    assert run(tmp_path, good, ROOT / "config" / "gate.yaml") == 0


def test_candidate_failing_on_recall_drop(tmp_path: Path) -> None:
    bad = dict(BASE, recall_at_5=0.80)                       # drop of ten points
    assert run(tmp_path, bad, ROOT / "config" / "gate.yaml") == 1


def test_each_rule_can_fail_alone(tmp_path: Path) -> None:
    config = ROOT / "config" / "gate.yaml"
    for field, value in [("faithfulness", 0.5), ("citation_validity", 0.9),
                         ("latency_p95", 99.0), ("abstention_correctness", 0.8)]:
        assert run(tmp_path, dict(BASE, **{field: value}), config) == 1, field


def test_changing_the_threshold_changes_the_outcome(tmp_path: Path) -> None:
    cand = dict(BASE, faithfulness=0.82)
    assert run(tmp_path, cand, cfg(tmp_path, min_faithfulness=0.80)) == 0
    assert run(tmp_path, cand, cfg(tmp_path, min_faithfulness=0.90)) == 1


def test_percent_values_are_normalised() -> None:
    m = gate.metrics_of({"metrics": {"recall_at_5": 92.0, "latency_p95_ms": 1500}})
    assert m["recall_at_5"] == pytest.approx(0.92) and m["latency_p95"] == pytest.approx(1.5)


def test_missing_baseline_is_an_error_unless_allowed(tmp_path: Path) -> None:
    results = tmp_path / "r"
    results.mkdir()
    write(results / "a.json", BASE)
    config = str(ROOT / "config" / "gate.yaml")
    assert gate.main(["--results", str(results), "--config", config,
                      "--baseline", str(tmp_path / "nope.json")]) == 2


def test_promote_copies_the_result_only_on_pass(tmp_path: Path) -> None:
    results = tmp_path / "r"
    results.mkdir()
    write(results / "a.json", BASE)
    target = tmp_path / "baseline.json"
    code = gate.main(["--results", str(results), "--config", str(ROOT / "config" / "gate.yaml"),
                      "--baseline", str(target), "--allow-missing-baseline",
                      "--promote-to", str(target), "--no-decision-file"])
    assert code == 0 and target.is_file()


def test_decision_file_is_written_next_to_the_results(tmp_path: Path) -> None:
    results = tmp_path / "r"
    results.mkdir()
    write(results / "a.json", BASE)
    gate.main(["--results", str(results), "--config", str(ROOT / "config" / "gate.yaml"),
               "--baseline", str(tmp_path / "x.json"), "--allow-missing-baseline"])
    assert list(results.glob("gate-*.json"))


def test_configmap_matches_config_file() -> None:
    """k8s/base/gate-thresholds.yaml must carry exactly the values of config/gate.yaml."""
    file_cfg = yaml.safe_load((ROOT / "config" / "gate.yaml").read_text())
    cm = yaml.safe_load((ROOT / "k8s" / "base" / "gate-thresholds.yaml").read_text())
    assert yaml.safe_load(cm["data"]["gate.yaml"]) == file_cfg
