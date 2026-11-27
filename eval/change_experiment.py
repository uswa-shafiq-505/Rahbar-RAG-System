"""Change experiment runner: baseline vs candidate across 3 eval levels (Assignment 03 Part C1).

Compares baseline (`rahbar`) and candidate (`rahbar_candidate`) across:
  - Module: embedding_gap (from eval.module_embedding)
  - Component: recall_at_3, recall_at_5, mrr, ndcg_at_5 (from eval.component_retrieval)
  - End-to-end: faithfulness, abstention_correctness, citation_validity,
                latency_p50, latency_p95 (from eval.e2e_answers / eval.ragops.suite)

Applies release gate thresholds from config/gate.yaml via scripts/gate.py and outputs
a timestamped JSON report under eval/results/change_experiment_<UTC>.json.

Run from repository root:
    python -m eval.change_experiment --change chunk_size
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from eval.component_retrieval import evaluate_retrieval, make_ollama_embed_fn
from eval.e2e_answers import evaluate_e2e
from eval.module_embedding import evaluate_embeddings
from eval.ragops.suite import evaluate_api
from scripts.gate import evaluate as gate_evaluate
from scripts.gate import load_json as gate_load_json
from scripts.gate import metrics_of as gate_metrics_of

DEFAULT_GATE_CONFIG = Path("config/gate.yaml")
DEFAULT_GOLDEN_PATH = Path("eval/golden.jsonl")
DEFAULT_PAIRS_PATH = Path("eval/embedding_pairs.yaml")
DEFAULT_OUT_DIR = Path("eval/results")

MODULE_KEYS = ("embedding_gap",)
COMPONENT_KEYS = ("recall_at_3", "recall_at_5", "mrr", "ndcg_at_5")
E2E_KEYS = (
    "faithfulness",
    "abstention_correctness",
    "citation_validity",
    "latency_p50",
    "latency_p95",
)

CHANGE_DESCRIPTIONS = {
    "chunk_size": ("chunk_size=400", "chunk_size=300"),
    "top_k": ("top_k=5", "top_k=8"),
    "embedding_model": ("embed_model=nomic-embed-text", "embed_model=candidate"),
}


def compute_level_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
    keys: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Compute {baseline, candidate, delta} dicts for a single evaluation level."""
    base_clean = {k: round(float(baseline.get(k, 0.0)), 6) for k in keys}
    cand_clean = {k: round(float(candidate.get(k, 0.0)), 6) for k in keys}
    delta = {k: round(cand_clean[k] - base_clean[k], 6) for k in keys}
    return {"baseline": base_clean, "candidate": cand_clean, "delta": delta}


def load_gate_thresholds(gate_config_path: Path = DEFAULT_GATE_CONFIG) -> dict[str, Any]:
    """Read thresholds mapping from config/gate.yaml."""
    raw = yaml.safe_load(gate_config_path.read_text(encoding="utf-8")) or {}
    thresholds = raw.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        raise ValueError(f"{gate_config_path}: expected dict under 'thresholds'")
    return dict(thresholds)


def _to_gate_metrics(
    component: dict[str, float],
    end_to_end: dict[str, float],
    latency_in_ms: bool = True,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for k in COMPONENT_KEYS:
        if k in component:
            merged[k] = float(component[k])
    for k in ("faithfulness", "abstention_correctness", "citation_validity", "latency_p50"):
        if k in end_to_end:
            merged[k] = float(end_to_end[k])
    if "latency_p95" in end_to_end:
        lat_p95 = float(end_to_end["latency_p95"])
        merged["latency_p95"] = (lat_p95 / 1000.0) if latency_in_ms else lat_p95
    return merged


def compare_all_levels(
    change_type: str,
    baseline_metrics: dict[str, dict[str, float]],
    candidate_metrics: dict[str, dict[str, float]],
    gate_config_path: Path = DEFAULT_GATE_CONFIG,
    baseline_collection: str = "rahbar",
    candidate_collection: str = "rahbar_candidate",
    latency_in_ms: bool = True,
) -> dict[str, Any]:
    """Compare baseline vs candidate across all 3 levels and apply the release gate."""
    base_label, cand_label = CHANGE_DESCRIPTIONS.get(change_type, ("baseline", "candidate"))

    mod_cmp = compute_level_delta(
        baseline_metrics.get("module", {}),
        candidate_metrics.get("module", {}),
        MODULE_KEYS,
    )
    comp_cmp = compute_level_delta(
        baseline_metrics.get("component", {}),
        candidate_metrics.get("component", {}),
        COMPONENT_KEYS,
    )
    e2e_cmp = compute_level_delta(
        baseline_metrics.get("end_to_end", {}),
        candidate_metrics.get("end_to_end", {}),
        E2E_KEYS,
    )

    thresholds = load_gate_thresholds(gate_config_path)
    base_flat = _to_gate_metrics(
        comp_cmp["baseline"], e2e_cmp["baseline"], latency_in_ms=latency_in_ms
    )
    cand_flat = _to_gate_metrics(
        comp_cmp["candidate"], e2e_cmp["candidate"], latency_in_ms=latency_in_ms
    )

    gate_rows = gate_evaluate(cand_flat, base_flat, thresholds)
    failures = [
        f"{r['rule']}: value={r['value']} limit={r['limit']}"
        for r in gate_rows
        if not r["ok"]
    ]
    passed = len(failures) == 0

    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "change_type": change_type,
        "baseline": {"collection": baseline_collection, "label": base_label},
        "candidate": {"collection": candidate_collection, "label": cand_label},
        "metrics": {
            "module": mod_cmp,
            "component": comp_cmp,
            "end_to_end": e2e_cmp,
        },
        "verdict": "RELEASE" if passed else "REJECT",
        "gate_result": {"pass": passed, "failures": failures},
    }


def _collect_live_metrics(
    *,
    collection: str,
    api_url: str,
    golden_path: Path,
    pairs_path: Path,
    chroma_client: Any,
    embed_fn: Any,
    embed_model: str,
    ollama_url: str,
    use_suite: bool,
    tmp_out: Path,
) -> dict[str, dict[str, float]]:
    mod_report = evaluate_embeddings(
        pairs_path=pairs_path,
        embed_fn=embed_fn,
        embed_model=embed_model,
        ollama_url=ollama_url,
        n_timing=10,
        check_subprocess=False,
    )
    comp_report = evaluate_retrieval(
        golden_path=golden_path,
        chroma_client=chroma_client,
        collection_name=collection,
        embed_fn=embed_fn,
        embed_model=embed_model,
    )
    if use_suite:
        suite_path = evaluate_api(api_url=api_url, collection=collection, out_dir=tmp_out)
        flat = gate_metrics_of(gate_load_json(suite_path))
        e2e_metrics = {
            "faithfulness": float(flat.get("faithfulness", 0.0)),
            "abstention_correctness": float(flat.get("abstention_correctness", 0.0)),
            "citation_validity": float(flat.get("citation_validity", 0.0)),
            "latency_p50": float(flat.get("latency_p50", 0.0)) * 1000.0,
            "latency_p95": float(flat.get("latency_p95", 0.0)) * 1000.0,
        }
    else:
        e2e_report = evaluate_e2e(
            golden_path=golden_path,
            api_url=api_url,
            ollama_url=ollama_url,
            skip_judge=True,
        )
        e2e_metrics = {
            "faithfulness": float(e2e_report["faithfulness"]),
            "abstention_correctness": float(e2e_report["abstention_correctness"]),
            "citation_validity": float(e2e_report["citation_validity"]),
            "latency_p50": float(e2e_report["latency_p50"]),
            "latency_p95": float(e2e_report["latency_p95"]),
        }

    return {
        "module": {"embedding_gap": float(mod_report["gap"])},
        "component": {k: float(comp_report["metrics"][k]) for k in COMPONENT_KEYS},
        "end_to_end": e2e_metrics,
    }


def write_results(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Create out_dir if needed and write timestamped change_experiment_<UTC>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"change_experiment_{ts_slug}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


def format_comparison_table(report: dict[str, Any]) -> str:
    """Render a 3-level comparison + delta table and release verdict."""
    lines = [
        f"=== Change Experiment: {report['change_type']} ===",
        f"Baseline  : {report['baseline']['collection']} ({report['baseline']['label']})",
        f"Candidate : {report['candidate']['collection']} ({report['candidate']['label']})",
        "",
        f"{'Level / Metric':<30} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}",
        f"{'-'*30} {'-'*10} {'-'*10} {'-'*10}",
    ]
    for level in ("module", "component", "end_to_end"):
        blk = report["metrics"][level]
        for key in blk["baseline"]:
            b_val = blk["baseline"][key]
            c_val = blk["candidate"][key]
            d_val = blk["delta"][key]
            label = f"{level}.{key}"
            lines.append(f"{label:<30} {b_val:>10.4f} {c_val:>10.4f} {d_val:>+10.4f}")
    lines.append("")
    lines.append(f"VERDICT : {report['verdict']}")
    if report["gate_result"]["failures"]:
        for fail in report["gate_result"]["failures"]:
            lines.append(f"  FAIL  : {fail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Part C1 baseline vs candidate change experiment.")
    parser.add_argument(
        "--change",
        choices=["chunk_size", "top_k", "embedding_model"],
        default="chunk_size",
    )
    parser.add_argument("--baseline-collection", default="rahbar")
    parser.add_argument("--candidate-collection", default="rahbar_candidate")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument(
        "--candidate-api-url",
        default=os.environ.get("CANDIDATE_API_URL", "http://localhost:8001"),
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chroma-host", default=os.environ.get("CHROMA_HOST", "localhost"))
    parser.add_argument("--chroma-port", type=int, default=int(os.environ.get("CHROMA_PORT", "8000")))
    parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--use-suite", action="store_true")
    args = parser.parse_args(argv)

    import chromadb

    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    embed_fn = make_ollama_embed_fn(args.embed_model, args.ollama_url)

    base_metrics = _collect_live_metrics(
        collection=args.baseline_collection,
        api_url=args.api_url,
        golden_path=args.golden,
        pairs_path=args.pairs,
        chroma_client=client,
        embed_fn=embed_fn,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        use_suite=args.use_suite,
        tmp_out=args.out,
    )
    cand_metrics = _collect_live_metrics(
        collection=args.candidate_collection,
        api_url=args.candidate_api_url,
        golden_path=args.golden,
        pairs_path=args.pairs,
        chroma_client=client,
        embed_fn=embed_fn,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        use_suite=args.use_suite,
        tmp_out=args.out,
    )

    report = compare_all_levels(
        change_type=args.change,
        baseline_metrics=base_metrics,
        candidate_metrics=cand_metrics,
        gate_config_path=args.gate_config,
        baseline_collection=args.baseline_collection,
        candidate_collection=args.candidate_collection,
    )
    out_path = write_results(report, out_dir=args.out)
    print(format_comparison_table(report))
    print(f"Report written : {out_path}")
    return 0 if report["gate_result"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
