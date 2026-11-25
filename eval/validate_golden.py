"""Validator for eval/golden.jsonl (Assignment 03 Part B1).

Run from repo root:
    python -m eval.validate_golden
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = {
    "id",
    "question",
    "type",
    "difficulty",
    "ground_truth",
    "expected_answer",
    "must_abstain",
    "label",
    "reviewed",
    "notes",
}

DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"


def validate_golden(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    malformed = False
    warnings: list[str] = []

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            print(f"ERROR line {lineno}: malformed JSON: {exc}", file=sys.stderr)
            malformed = True
            continue
        if not isinstance(obj, dict):
            print(f"ERROR line {lineno}: expected JSON object, got {type(obj).__name__}", file=sys.stderr)
            malformed = True
            continue
        rows.append(obj)

        qid = obj.get("id", f"line:{lineno}")
        missing = sorted(REQUIRED_KEYS - set(obj.keys()))
        if missing:
            warnings.append(f"{qid} (line {lineno}): missing required keys: {', '.join(missing)}")

        qtype = obj.get("type")
        gt = obj.get("ground_truth")
        must_abstain = obj.get("must_abstain")

        if qtype == "answerable":
            if not isinstance(gt, dict):
                warnings.append(f"{qid}: answerable row must have non-null ground_truth dict")
            else:
                if not gt.get("source"):
                    warnings.append(f"{qid}: answerable row missing ground_truth.source")
                page = gt.get("page")
                if page is None or not isinstance(page, int) or page < 1:
                    warnings.append(f"{qid}: answerable row must have integer ground_truth.page >= 1")
            if must_abstain is not False:
                warnings.append(f"{qid}: answerable row has must_abstain={must_abstain!r} (expected false)")

        elif qtype == "unanswerable":
            if must_abstain is not True:
                warnings.append(f"{qid}: unanswerable row must have must_abstain=true")
            if gt is not None:
                warnings.append(f"{qid}: unanswerable row must have ground_truth=null")

    type_counts = Counter(r.get("type", "<missing>") for r in rows)
    diff_counts = Counter(r.get("difficulty", "<missing>") for r in rows)
    label_counts = Counter(r.get("label", "<missing>") for r in rows)
    reviewed_count = sum(1 for r in rows if r.get("reviewed") is True)
    abstain_count = sum(1 for r in rows if r.get("must_abstain") is True)
    sources = Counter(
        r["ground_truth"]["source"]
        for r in rows
        if isinstance(r.get("ground_truth"), dict) and r["ground_truth"].get("source")
    )

    if type_counts["answerable"] < 15:
        warnings.append(
            f"Count check: answerable={type_counts['answerable']} < 15 required"
        )
    if type_counts["unanswerable"] < 5:
        warnings.append(
            f"Count check: unanswerable={type_counts['unanswerable']} < 5 required"
        )

    print(f"=== Golden Set Validation Summary ({path}) ===")
    print(f"Total questions : {len(rows)}")
    print(f"Reviewed        : {reviewed_count}/{len(rows)}")
    print(f"Must abstain    : {abstain_count}/{len(rows)}")
    print()
    print("By type:")
    for k in ("answerable", "unanswerable", "hard"):
        print(f"  {k:<15} {type_counts.get(k, 0)}")
    for k, v in sorted(type_counts.items()):
        if k not in {"answerable", "unanswerable", "hard"}:
            print(f"  {k:<15} {v}")
    print()
    print("By difficulty:")
    for k in ("easy", "medium", "hard"):
        print(f"  {k:<15} {diff_counts.get(k, 0)}")
    print()
    print("By label:")
    for k in ("human", "synthetic"):
        print(f"  {k:<15} {label_counts.get(k, 0)}")
    print()
    print("By source (ground_truth.source):")
    for src, cnt in sorted(sources.items()):
        print(f"  {src:<58} {cnt}")

    if warnings:
        print()
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print()
        print("Warnings (0): all checks passed.")

    return 1 if malformed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eval/golden.jsonl.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden.jsonl",
    )
    args = parser.parse_args()
    return validate_golden(args.path)


if __name__ == "__main__":
    sys.exit(main())

