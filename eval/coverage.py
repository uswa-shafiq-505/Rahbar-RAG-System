"""Query coverage evaluation against the golden set (Assignment 03 Part E2).

Embeds live queries (e.g. eval/live_queries.jsonl) and golden questions
(eval/golden.jsonl) in batches of 20, finds each live query's maximum cosine
similarity to any golden question, reports coverage overall and by domain,
groups uncovered queries by topic, and exits 0 iff overall_coverage >= 0.85.

Run from repository root:
    python -m eval.coverage --live eval/live_queries.jsonl --golden eval/golden.jsonl --out eval/results/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

EmbedFn = Callable[[str], list[float]]
BatchEmbedFn = Callable[[list[str]], list[list[float]]]

DEFAULT_LIVE_PATH = Path("eval/live_queries.jsonl")
DEFAULT_GOLDEN_PATH = Path("eval/golden.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")
DEFAULT_THRESHOLD = 0.75
DEFAULT_MIN_COVERAGE = 0.85
BATCH_SIZE = 20

_DUP_QUERY_KEY_RE = re.compile(r'"query":\s*"query":\s*')


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0.0 else vec


def make_ollama_batch_embed_fn(
    embed_model: str,
    ollama_url: str,
    timeout: float = 60.0,
) -> BatchEmbedFn:
    """Batch embed via Ollama /api/embed (falling back to /api/embeddings per text)."""
    base = ollama_url.rstrip("/")

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = requests.post(
                f"{base}/api/embed",
                json={"model": embed_model, "input": texts},
                timeout=timeout,
            )
            if resp.status_code != 404:
                resp.raise_for_status()
                data = resp.json()
                embs = data.get("embeddings")
                if isinstance(embs, list) and len(embs) == len(texts):
                    return [_normalize([float(x) for x in v]) for v in embs]
        except requests.HTTPError:
            raise
        except Exception:  # noqa: BLE001 - fall back to single-prompt endpoint
            pass

        vectors: list[list[float]] = []
        for text in texts:
            r = requests.post(
                f"{base}/api/embeddings",
                json={"model": embed_model, "prompt": text},
                timeout=timeout,
            )
            r.raise_for_status()
            vec = [float(x) for x in r.json()["embedding"]]
            vectors.append(_normalize(vec))
        return vectors

    return _embed_batch


def batch_embed(
    texts: list[str],
    embed_fn: EmbedFn | None = None,
    batch_embed_fn: BatchEmbedFn | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of texts in chunks of `batch_size` (default 20)."""
    if not texts:
        return []
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        if batch_embed_fn is not None:
            out.extend(batch_embed_fn(chunk))
        elif embed_fn is not None:
            out.extend(embed_fn(t) for t in chunk)
        else:
            raise ValueError("Either embed_fn or batch_embed_fn must be provided")
    return out


def _load_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, tolerating accidental duplicate '"query": "query":' key typos."""
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            fixed = _DUP_QUERY_KEY_RE.sub('"query": ', stripped)
            rows.append(json.loads(fixed))
    return rows


def _classify_domain(row: dict[str, Any]) -> str:
    raw = str(row.get("domain") or row.get("group") or "").strip().lower()
    if raw in {"in", "in_domain", "in-domain"}:
        return "in"
    if raw in {"out", "out_domain", "out_of_domain", "out-of-domain", "uncovered"}:
        return "out"
    return "in"


def _topic_for_query(row: dict[str, Any], query_text: str) -> str:
    manual = str(row.get("topic") or "").strip()
    if manual:
        return manual
    words = query_text.split()
    return " ".join(words[:3]).lower() if words else "unknown"


def evaluate_coverage(
    live_path: Path,
    golden_path: Path,
    embed_fn: EmbedFn | None = None,
    batch_embed_fn: BatchEmbedFn | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    embed_model: str | None = None,
    ollama_url: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Compute golden-set coverage over live queries."""
    model_name = embed_model if embed_model is not None else os.environ.get("EMBED_MODEL", "nomic-embed-text")
    url = ollama_url if ollama_url is not None else os.environ.get("OLLAMA_URL", "http://localhost:11434")

    if embed_fn is None and batch_embed_fn is None:
        batch_embed_fn = make_ollama_batch_embed_fn(model_name, url)

    live_rows = _load_jsonl_tolerant(live_path)
    golden_rows = _load_jsonl_tolerant(golden_path)

    live_texts = [str(r.get("query") or r.get("question") or "") for r in live_rows]
    golden_texts = [str(r.get("question") or r.get("query") or "") for r in golden_rows]

    live_vecs = batch_embed(live_texts, embed_fn=embed_fn, batch_embed_fn=batch_embed_fn, batch_size=batch_size)
    golden_vecs = batch_embed(golden_texts, embed_fn=embed_fn, batch_embed_fn=batch_embed_fn, batch_size=batch_size)

    golden_ids = [str(r.get("id") or f"g{i}") for i, r in enumerate(golden_rows)]

    per_live_query: list[dict[str, Any]] = []
    uncovered: list[dict[str, Any]] = []
    topic_counts: Counter[str] = Counter()

    in_total, in_cov = 0, 0
    out_total, out_cov = 0, 0

    for row, qtext, lvec in zip(live_rows, live_texts, live_vecs):
        qid = str(row.get("id") or "")
        domain = _classify_domain(row)

        best_gid = ""
        best_sim = -1.0
        for gid, gvec in zip(golden_ids, golden_vecs):
            sim = cosine_similarity(lvec, gvec)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid
        if not golden_vecs:
            best_sim = 0.0

        covered = best_sim >= threshold
        best_sim_rounded = round(best_sim, 6)

        if domain == "in":
            in_total += 1
            if covered:
                in_cov += 1
        else:
            out_total += 1
            if covered:
                out_cov += 1

        per_live_query.append(
            {
                "id": qid,
                "query": qtext,
                "best_match_golden_id": best_gid,
                "best_sim": best_sim_rounded,
                "covered": covered,
            }
        )

        if not covered:
            topic = _topic_for_query(row, qtext)
            topic_counts[topic] += 1
            uncovered.append(
                {
                    "id": qid,
                    "query": qtext,
                    "best_sim": best_sim_rounded,
                    "suggested_topic": topic,
                }
            )

    n_live = len(live_rows)
    total_cov = in_cov + out_cov
    overall_coverage = round(total_cov / n_live, 4) if n_live else 0.0
    in_domain_coverage = round(in_cov / in_total, 4) if in_total else 0.0
    out_domain_coverage = round(out_cov / out_total, 4) if out_total else 0.0

    uncovered_topics = [
        {"topic": topic, "count": count}
        for topic, count in sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embed_model": model_name,
        "n_live": n_live,
        "n_golden": len(golden_rows),
        "threshold": threshold,
        "overall_coverage": overall_coverage,
        "in_domain_coverage": in_domain_coverage,
        "out_domain_coverage": out_domain_coverage,
        "uncovered": uncovered,
        "uncovered_topics": uncovered_topics,
        "per_live_query": per_live_query,
    }


def write_results(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Create out_dir if needed and write timestamped coverage_<UTC>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"coverage_{ts_slug}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


def format_summary_table(report: dict[str, Any]) -> str:
    """Format a human-readable summary table for stdout."""
    lines = [
        "=== Query Coverage Evaluation (Part E2) ===",
        f"Model               : {report['embed_model']}",
        f"Live / Golden       : {report['n_live']} / {report['n_golden']} (sim threshold={report['threshold']})",
        f"Overall coverage    : {report['overall_coverage']:.2%}",
        f"In-domain coverage  : {report['in_domain_coverage']:.2%}",
        f"Out-domain coverage : {report['out_domain_coverage']:.2%}",
        f"Uncovered queries   : {len(report['uncovered'])}",
    ]
    if report["uncovered_topics"]:
        lines.append("Uncovered topics:")
        for item in report["uncovered_topics"]:
            lines.append(f"  - {item['topic']:<24} {item['count']}")
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    embed_fn: EmbedFn | None = None,
    batch_embed_fn: BatchEmbedFn | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Part E2 live-query coverage evaluation.")
    parser.add_argument("--live", type=Path, default=DEFAULT_LIVE_PATH)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--no-exit-code", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_coverage(
        live_path=args.live,
        golden_path=args.golden,
        embed_fn=embed_fn,
        batch_embed_fn=batch_embed_fn,
        threshold=args.threshold,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
    )
    out_path = write_results(report, out_dir=args.out)
    print(format_summary_table(report))
    print(f"Report written      : {out_path}")

    if args.no_exit_code:
        return 0
    return 0 if report["overall_coverage"] >= DEFAULT_MIN_COVERAGE else 1


if __name__ == "__main__":
    sys.exit(main())
