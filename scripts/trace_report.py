"""Summarise the trace log over a time window (Assignment 03, part D3).

    python scripts/trace_report.py --dir traces --since 24h
    python scripts/trace_report.py --dir traces --since 2026-11-20T00:00:00Z --json report.json

Reports: queries, top-1 retrieval similarity distribution, abstention rate, cache hit rate,
p50/p95 latency per stage, and the number of injection attempts flagged by a guardrail.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

FLAG_ACTIONS = {"flagged", "blocked", "removed", "stripped", "sanitized", "rejected"}


def parse_when(text: str, now: datetime) -> datetime:
    m = re.fullmatch(r"(\d+)([smhd])", text.strip())
    if m:
        unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[m.group(2)]
        return now - timedelta(**{unit: int(m.group(1))})
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def load(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "trace_id" in rec:
                    yield rec


def in_window(rec: dict[str, Any], start: datetime, end: datetime) -> bool:
    try:
        ts = datetime.fromisoformat(str(rec["ts"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return start <= ts <= end


def injections_in(rec: dict[str, Any]) -> int:
    count = 0
    for g in rec.get("guardrail_actions") or []:
        if not isinstance(g, dict):
            continue
        text = " ".join(str(g.get(k, "")).lower() for k in ("rail", "type", "action", "detail"))
        if "inject" in text and (str(g.get("action", "")).lower() in FLAG_ACTIONS
                                 or g.get("flagged") is True):
            count += 1
    return count


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    top1: list[float] = []
    stages: dict[str, list[float]] = {}
    cache = Counter(str(r.get("cache", "unknown")).lower() for r in records)
    abstained = sum(1 for r in records if r.get("abstained") is True)
    injections = sum(injections_in(r) for r in records)
    for r in records:
        scores: list[float] = [float(x["score"]) for x in (r.get("retrieved") or [])
                               if isinstance(x, dict) and isinstance(x.get("score"), (int, float))]
        if scores:
            top1.append(max(scores))
        for stage, ms in (r.get("latency_ms") or {}).items():
            if isinstance(ms, (int, float)):
                stages.setdefault(stage, []).append(float(ms))
    lookups = cache.get("hit", 0) + cache.get("miss", 0)
    buckets = Counter(min(int(s * 10), 9) for s in top1)
    return {
        "queries": n,
        "abstention_rate": (abstained / n) if n else None,
        "cache_hit_rate": (cache.get("hit", 0) / lookups) if lookups else None,
        "cache": dict(cache),
        "injection_attempts_flagged": injections,
        "top1_similarity": {
            "count": len(top1),
            "mean": (sum(top1) / len(top1)) if top1 else None,
            "p10": percentile(top1, 0.10), "median": percentile(top1, 0.50),
            "p90": percentile(top1, 0.90),
            "histogram": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": buckets.get(b, 0)
                          for b in range(10)},
        },
        "latency_ms": {s: {"p50": percentile(v, 0.50), "p95": percentile(v, 0.95),
                           "count": len(v)} for s, v in sorted(stages.items())},
        "collections": dict(Counter(str(r.get("collection", "?")) for r in records)),
        "embed_models": dict(Counter(str(r.get("embed_model", "?")) for r in records)),
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(summary: dict[str, Any], start: datetime, end: datetime) -> str:
    t = summary["top1_similarity"]
    lines = [f"window: {start.isoformat(timespec='seconds')} .. {end.isoformat(timespec='seconds')}",
             f"queries:                      {summary['queries']}",
             f"abstention rate:              {fmt(summary['abstention_rate'])}",
             f"cache hit rate:               {fmt(summary['cache_hit_rate'])}   {summary['cache']}",
             f"injection attempts flagged:   {summary['injection_attempts_flagged']}",
             f"top-1 similarity:             mean {fmt(t['mean'])}  p10 {fmt(t['p10'])}  "
             f"median {fmt(t['median'])}  p90 {fmt(t['p90'])}",
             "top-1 similarity histogram:"]
    for bucket, count in t["histogram"].items():
        lines.append(f"  {bucket}  {'#' * min(count, 50)} {count}")
    lines.append("latency per stage (ms):")
    for stage, v in summary["latency_ms"].items():
        lines.append(f"  {stage:<12} p50 {fmt(v['p50'], 1):>9}   p95 {fmt(v['p95'], 1):>9}   "
                     f"n={v['count']}")
    lines.append(f"collections: {summary['collections']}   models: {summary['embed_models']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="traces")
    ap.add_argument("--since", default="24h", help="30m, 6h, 7d or an ISO timestamp")
    ap.add_argument("--until", default="", help="ISO timestamp (default: now)")
    ap.add_argument("--json", default="", help="also write the summary to this file")
    args = ap.parse_args(argv)
    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"trace_report: {directory} is not a directory", file=sys.stderr)
        return 2
    now = datetime.now(UTC)
    start = parse_when(args.since, now)
    end = parse_when(args.until, now) if args.until else now
    records = [r for r in load(directory) if in_window(r, start, end)]
    summary = summarise(records)
    print(render(summary, start, end))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
