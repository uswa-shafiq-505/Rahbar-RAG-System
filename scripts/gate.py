"""Release gate (Assignment 03, part C2).

Reads the newest evaluation result file, applies the thresholds from config/gate.yaml (a
ConfigMap on the cluster) and exits 0 (PASS) or 1 (FAIL). Exit code 2 means the gate could
not run (missing file, bad config): that is also a failure for the Job.

    python scripts/gate.py --results /data/results --baseline /data/results/baseline.json \\
        --config /config/gate.yaml

Result file shape (see docs/RAGOPS-CONTRACTS.md): {"meta": {...}, "metrics": {name: value}},
or the metrics at the top level. Ratios are fractions (0..1); a value above 1 for a ratio
metric is read as a percentage and divided by 100. Latencies are seconds (latency_p95), or
milliseconds under latency_p95_ms.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

RATIO_METRICS = {
    "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5", "faithfulness",
    "abstention_correctness", "citation_validity",
}
SKIP_NAMES = {"baseline.json"}


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def metrics_of(data: dict[str, Any]) -> dict[str, float]:
    raw = data.get("metrics", data)
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        v = float(value)
        if key in RATIO_METRICS and v > 1.0:
            v /= 100.0
        out[key] = v
    if "latency_p95" not in out and "latency_p95_ms" in out:
        out["latency_p95"] = out["latency_p95_ms"] / 1000.0
    return out


def newest_result(results: Path) -> Path:
    if results.is_file():
        return results
    files = [p for p in results.glob("*.json")
             if p.name not in SKIP_NAMES and not p.name.startswith("gate-")]
    if not files:
        raise FileNotFoundError(f"no result files in {results}")
    return max(files, key=lambda p: p.stat().st_mtime)


def evaluate(cand: dict[str, float], base: dict[str, float] | None,
             thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one row per rule: {rule, value, limit, ok, note}."""
    rows: list[dict[str, Any]] = []

    def add(rule: str, value: float | None, limit: str, ok: bool, note: str = "") -> None:
        rows.append({"rule": rule, "value": value, "limit": limit, "ok": ok, "note": note})

    if "max_recall_at_5_drop" in thresholds:
        drop = float(thresholds["max_recall_at_5_drop"])
        value = cand.get("recall_at_5")
        if value is None:
            add("recall_at_5 drop", None, f"<= {drop:.3f}", False, "recall_at_5 missing")
        elif base is None or "recall_at_5" not in base:
            add("recall_at_5 drop", value, f"<= {drop:.3f}", True, "no baseline: skipped")
        else:
            actual = base["recall_at_5"] - value
            add("recall_at_5 drop", actual, f"<= {drop:.3f}", actual <= drop + 1e-12,
                f"baseline {base['recall_at_5']:.3f}, candidate {value:.3f}")
    if "min_faithfulness" in thresholds:
        low = float(thresholds["min_faithfulness"])
        value = cand.get("faithfulness")
        add("faithfulness", value, f">= {low:.3f}", value is not None and value >= low)
    if "min_citation_validity" in thresholds:
        low = float(thresholds["min_citation_validity"])
        value = cand.get("citation_validity")
        add("citation_validity", value, f">= {low:.3f}", value is not None and value >= low)
    if "max_latency_p95_seconds" in thresholds:
        high = float(thresholds["max_latency_p95_seconds"])
        value = cand.get("latency_p95")
        add("latency p95 (s)", value, f"<= {high:.1f}", value is not None and value <= high)
    if "min_abstention_correctness" in thresholds:
        spec = thresholds["min_abstention_correctness"]
        value = cand.get("abstention_correctness")
        if spec == "baseline":
            if base is None or "abstention_correctness" not in base:
                add("abstention vs baseline", value, ">= baseline", True,
                    "no baseline: skipped")
            else:
                floor = base["abstention_correctness"]
                add("abstention vs baseline", value, f">= {floor:.3f}",
                    value is not None and value >= floor - 1e-12)
        else:
            low = float(spec)
            add("abstention correctness", value, f">= {low:.3f}",
                value is not None and value >= low)
    return rows


def render(rows: list[dict[str, Any]], passed: bool, cand_path: str) -> str:
    lines = [f"candidate: {cand_path}", "",
             f"{'rule':<26}{'value':>10}  {'limit':<14}{'verdict':<8}note",
             "-" * 78]
    for r in rows:
        v = "n/a" if r["value"] is None else f"{r['value']:.3f}"
        lines.append(f"{r['rule']:<26}{v:>10}  {r['limit']:<14}"
                     f"{'ok' if r['ok'] else 'FAIL':<8}{r['note']}")
    lines += ["-" * 78, f"GATE {'PASSED' if passed else 'FAILED'}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="result file, or a directory (newest wins)")
    ap.add_argument("--baseline", default="", help="baseline result file")
    ap.add_argument("--config", default="config/gate.yaml")
    ap.add_argument("--allow-missing-baseline", action="store_true")
    ap.add_argument("--promote-to", default="", help="copy the result here when the gate passes")
    ap.add_argument("--no-decision-file", action="store_true")
    args = ap.parse_args(argv)
    try:
        with open(args.config, encoding="utf-8") as fh:
            thresholds = (yaml.safe_load(fh) or {}).get("thresholds") or {}
        cand_path = newest_result(Path(args.results))
        cand = metrics_of(load_json(cand_path))
        base: dict[str, float] | None = None
        if args.baseline and Path(args.baseline).is_file():
            base = metrics_of(load_json(Path(args.baseline)))
        elif args.baseline and not args.allow_missing_baseline:
            print(f"gate: baseline {args.baseline} not found "
                  f"(use --allow-missing-baseline for the first run)", file=sys.stderr)
            return 2
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"gate: cannot run: {exc}", file=sys.stderr)
        return 2
    if not thresholds:
        print("gate: no thresholds in the config", file=sys.stderr)
        return 2

    rows = evaluate(cand, base, thresholds)
    passed = all(r["ok"] for r in rows)
    print(render(rows, passed, str(cand_path)))
    if not args.no_decision_file and cand_path.parent.is_dir():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        try:
            (cand_path.parent / f"gate-{stamp}.json").write_text(json.dumps(
                {"candidate": str(cand_path), "passed": passed, "rows": rows,
                 "thresholds": thresholds}, indent=2), encoding="utf-8")
        except OSError:
            pass
    if passed and args.promote_to:
        shutil.copyfile(cand_path, args.promote_to)
        print(f"promoted {cand_path.name} to {args.promote_to}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
