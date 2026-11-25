"""Trace tool (Assignment 03, part D2): print what produced one answer.

    python scripts/trace.py <trace_id> --dir traces/             # files copied from the volume
    kubectl logs -l app=api -n rahbar --tail=-1 | python scripts/trace.py <trace_id> --stdin

A prefix of the id is enough when it is unique. The exact context and prompt are printed
when the trace stored them (docs/RAGOPS-CONTRACTS.md); otherwise the tool says what is
missing instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def iter_records(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "trace_id" in rec:
            yield rec


def find(records: Iterable[dict[str, Any]], wanted: str) -> list[dict[str, Any]]:
    records = list(records)          # a generator would be used up by the exact-match pass
    exact = [r for r in records if r["trace_id"] == wanted]
    if exact:
        return exact[:1]
    seen: dict[str, dict[str, Any]] = {}
    for r in records:
        if str(r["trace_id"]).startswith(wanted):
            seen[r["trace_id"]] = r
    return list(seen.values())


def records_from_dir(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            yield from iter_records(fh)


def fmt_score(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def render(rec: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add(f"TRACE {rec.get('trace_id')}   {rec.get('ts', '?')}   pod {rec.get('pod', '?')}")
    add(f"collection: {rec.get('collection', '?')}   embed model: {rec.get('embed_model', '?')}"
        f"   index version: {rec.get('index_version', '?')}")
    add(f"cache: {rec.get('cache', '?')}   abstained: {rec.get('abstained', '?')}")
    add("")
    add("QUESTION")
    add(f"  {rec.get('query')}")
    if rec.get("rewritten_query"):
        add(f"  rewritten: {rec['rewritten_query']}")
    add("")
    add(f"FILTERS APPLIED: {json.dumps(rec.get('filters', {}), ensure_ascii=False)}")
    add("RETRIEVED")
    retrieved = rec.get("retrieved") or []
    for i, r in enumerate(retrieved, 1):
        add(f"  {i}. {r.get('chunk_id')}  doc={r.get('doc_id', r.get('source', '?'))}"
            f"  version={r.get('doc_version', '?')}  score={fmt_score(r.get('score'))}")
    if not retrieved:
        add("  (nothing retrieved)")
    if rec.get("reranked_order"):
        add(f"RERANKED ORDER: {rec['reranked_order']}")
    add("")
    context = rec.get("context") or []
    add("CONTEXT SENT TO THE MODEL")
    if context:
        for c in context:
            if isinstance(c, dict):
                add(f"  [{c.get('chunk_id')}] {c.get('text')}")
            else:
                add(f"  {c}")
    else:
        add("  (this trace did not store the context text)")
    add("")
    add("PROMPT")
    if rec.get("prompt"):
        add(rec["prompt"])
    elif rec.get("prompt_hash"):
        add(f"  (only the hash was stored: {rec['prompt_hash']}; the prompt cannot be rebuilt "
            f"without the full prompt or the context above)")
    else:
        add("  (no prompt stored)")
    add("")
    model = rec.get("model")
    add(f"MODEL AND PARAMETERS: {json.dumps(model, ensure_ascii=False) if model else '?'}")
    add("RESPONSE")
    add(f"  {rec.get('response')}")
    add(f"CITATIONS: {json.dumps(rec.get('citations', []), ensure_ascii=False)}")
    add(f"GUARDRAIL ACTIONS: {json.dumps(rec.get('guardrail_actions', []), ensure_ascii=False)}")
    add(f"LATENCY (ms): {json.dumps(rec.get('latency_ms', {}))}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace_id")
    ap.add_argument("--dir", default="traces", help="directory of traces-*.jsonl files")
    ap.add_argument("--file", default="", help="a single .jsonl file")
    ap.add_argument("--stdin", action="store_true", help="read trace lines from stdin")
    args = ap.parse_args(argv)
    if args.stdin:
        records: Iterable[dict[str, Any]] = iter_records(sys.stdin)
    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            records = list(iter_records(fh))
    else:
        directory = Path(args.dir)
        if not directory.is_dir():
            print(f"trace: {directory} is not a directory", file=sys.stderr)
            return 2
        records = records_from_dir(directory)
    matches = find(records, args.trace_id)
    if not matches:
        print(f"trace: no trace with id {args.trace_id}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print("trace: the prefix is ambiguous:", ", ".join(m["trace_id"] for m in matches),
              file=sys.stderr)
        return 1
    print(render(matches[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
