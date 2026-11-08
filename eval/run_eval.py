#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit codes: 0=pass, 1=gate failed, 2=config/unreachable error.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import yaml

HERE = Path(__file__).resolve().parent

# Gate thresholds (§5.4). Change only with a reason in docs/PIPELINE.md.
MAX_RECALL5_DROP    = 0.05   # recall@5 may fall at most 5 pp below baseline
MIN_CITATION_VALID  = 0.95   # ≥95 % of citations must match retrieved passages
SECTION_PREFIX_LEN  = 30     # chars used for section prefix matching


def load_golden(path: Path) -> list[dict]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        sys.exit(f"[eval] cannot parse {path}: {exc}")
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"[eval] no questions in {path}")
    for q in questions:
        if "id" not in q or "question" not in q:
            sys.exit(f"[eval] question missing 'id' or 'question': {q}")
    return questions


def _ask_one(client: httpx.Client, api: str, q: dict) -> tuple[Optional[dict], float]:
    """POST one question; (None, elapsed) on timeout/HTTP error."""
    t0 = time.perf_counter()
    try:
        r = client.post(
            f"{api.rstrip('/')}/api/ask",
            json={"question": q["question"]},
            headers={"X-Request-ID": f"eval-{q['id']}"},
        )
        r.raise_for_status()
        return r.json(), round(time.perf_counter() - t0, 2)
    except httpx.TimeoutException:
        print(f"  [warn] {q['id']} timed out — counting as miss")
    except httpx.HTTPStatusError as exc:
        print(f"  [warn] {q['id']} HTTP {exc.response.status_code} — counting as miss")
    return None, round(time.perf_counter() - t0, 2)


def _sec_match(a: str, b: str) -> bool:
    a, b = a[:SECTION_PREFIX_LEN].lower().strip(), b[:SECTION_PREFIX_LEN].lower().strip()
    return a.startswith(b) or b.startswith(a)


def _recall(q: dict, retrieved: list[dict], k: int) -> Optional[float]:
    """Binary recall@k: 1.0 if any top-k item matches an expected passage, else 0.0.
    None if no expected passages (must-abstain / unconfigured).
    """
    pairs: list[tuple[str, str]] = [
        (r["source"], r["section"]) for r in (q.get("relevant") or [])
        if isinstance(r, dict) and "source" in r and "section" in r
    ]
    ids: list[str] = [r for r in (q.get("relevant") or []) if isinstance(r, str)]
    if q.get("expected_source") and q.get("expected_section"):
        p = (q["expected_source"], q["expected_section"])
        if p not in pairs:
            pairs.append(p)
    if not pairs and not ids:
        return None
    for item in retrieved[:k]:
        if any(item.get("source") == s and _sec_match(item.get("section", ""), sec)
               for s, sec in pairs):
            return 1.0
        if item.get("chunk_id") in ids:
            return 1.0
    return 0.0


def _cit_validity(response: dict) -> float:
    """Fraction of citations backed by a retrieved passage (1.0 when no citations)."""
    cits = response.get("citations") or []
    if not cits:
        return 1.0
    keys = {(r.get("source"), r.get("section"), r.get("page"))
            for r in (response.get("retrieved") or [])}
    return sum(1 for c in cits if (c.get("source"), c.get("section"), c.get("page")) in keys) / len(cits)


def summarise(rows: list[dict], api_url: str) -> dict:
    r3s, r5s, by_type, cits = [], [], {}, []
    abstain_ok = abstain_total = 0
    per_q = []

    for row in rows:
        q, resp = row["q"], row["response"]
        qtype = q.get("type", "ordinary")
        is_abstain = bool(q.get("must_abstain")) or qtype in ("must_abstain", "abstain")
        retrieved = (resp or {}).get("retrieved") or []
        afc = (resp or {}).get("answered_from_context", True)
        cit_list = (resp or {}).get("citations") or []

        if not is_abstain and qtype != "prompt_injection" and resp is not None:
            if (v3 := _recall(q, retrieved, 3)) is not None:
                r3s.append(v3)
            if (v5 := _recall(q, retrieved, 5)) is not None:
                r5s.append(v5)
                by_type.setdefault(qtype, []).append(v5)

        if not is_abstain and resp is not None:
            cits.append(_cit_validity(resp))

        if is_abstain:
            abstain_total += 1
            if not afc:
                abstain_ok += 1

        rkeys = {(r.get("source"), r.get("section"), r.get("page")) for r in retrieved}
        n_valid = sum(1 for c in cit_list if (c.get("source"), c.get("section"), c.get("page")) in rkeys)
        per_q.append({
            "id": q["id"], "type": qtype, "question": q["question"][:80],
            "answered_from_context": afc if resp else None,
            "n_citations": len(cit_list), "n_valid_citations": n_valid,
            "top_k_sources": [r.get("source") for r in retrieved[:5]],
            "seconds": row["seconds"],
        })

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "api_url": api_url,
        "retrieval": {
            "recall_at_3": round(sum(r3s) / len(r3s), 4) if r3s else 0.0,
            "recall_at_5": round(sum(r5s) / len(r5s), 4) if r5s else 0.0,
            "by_type": {t: {"recall_at_5": round(sum(v) / len(v), 4), "n": len(v)}
                        for t, v in by_type.items()},
        },
        "generation": {
            "citation_validity_pct": round(sum(cits) / len(cits), 4) if cits else 1.0,
            "abstention_correctness_pct": round(abstain_ok / abstain_total, 4) if abstain_total else 1.0,
            "must_abstain_violations": abstain_total - abstain_ok,
        },
        "total_questions": len(rows),
        "per_question": per_q,
    }


def gate(current: dict, baseline: Optional[dict]) -> list[str]:
    """Return one failure string per broken gate (§5.4); empty list = pass."""
    failures: list[str] = []
    if baseline is not None:
        base_r5 = baseline.get("retrieval", {}).get("recall_at_5", 0.0)
        curr_r5 = current["retrieval"]["recall_at_5"]
        drop = base_r5 - curr_r5
        if drop > MAX_RECALL5_DROP:
            failures.append(
                f"recall@5 dropped {drop:.1%} (baseline {base_r5:.1%} → {curr_r5:.1%},"
                f" max drop {MAX_RECALL5_DROP:.1%})"
            )
    violations = current["generation"]["must_abstain_violations"]
    if violations:
        failures.append(f"{violations} must-abstain question(s) answered from context (required: 0)")
    cit = current["generation"]["citation_validity_pct"]
    if cit < MIN_CITATION_VALID:
        failures.append(f"citation validity {cit:.1%} < required {MIN_CITATION_VALID:.1%}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header for Ingress rules")
    parser.add_argument("--golden",   type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out",      type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    if not args.update_baseline and not args.baseline.exists():
        print("[eval] No baseline. Run with --update-baseline first.")
        return 2

    questions = load_golden(args.golden)
    print(f"[eval] {len(questions)} questions from {args.golden}")

    rows: list[dict] = []
    try:
        with httpx.Client(timeout=120.0, headers={"Host": args.host} if args.host else {}) as client:
            for q in questions:
                resp, elapsed = _ask_one(client, args.api, q)
                rows.append({"q": q, "response": resp, "seconds": elapsed})
                tag = "TIMEOUT" if resp is None else ("ABSTAIN" if not resp.get("answered_from_context") else "OK")
                print(f"  {q['id']:<5} {elapsed:>6.2f}s  [{tag}]  {q['question'][:55]}")
    except httpx.ConnectError as exc:
        print(f"[eval] API unreachable at {args.api}: {exc}")
        return 2

    results = summarise(rows, args.api)
    results["golden"] = str(args.golden)
    r, g = results["retrieval"], results["generation"]
    print(f"\n  recall@3={r['recall_at_3']:.1%}  recall@5={r['recall_at_5']:.1%}"
          f"  citation_validity={g['citation_validity_pct']:.1%}"
          f"  abstention={g['abstention_correctness_pct']:.1%}")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  results written to {args.out}")

    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  baseline updated: {args.baseline}")
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    failures = gate(results, baseline)
    if failures:
        print("\nGATE FAILED")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
