"""End-to-end answer evaluation (Assignment 03 Part B4).

Queries the running API for every question in eval/golden.jsonl and computes:
  - faithfulness (claim-bearing sentences with [N] citations)
  - abstention_correctness (% must_abstain Qs with answered_from_context=False)
  - citation_validity (% citations matching retrieved chunks' source + page)
  - latency_p50 / latency_p95 (ms per /api/ask call)
  - LLM-as-judge faithfulness via Ollama + agreement vs hand-labelled subset

Run from repository root:
    python -m eval.e2e_answers --api-url http://localhost:8000 --golden eval/golden.jsonl --out eval/results/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_GOLDEN_PATH = Path("eval/golden.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")
DEFAULT_LABELS_PATH = Path("eval/fixtures/labelled_answers.jsonl")
REQUEST_TIMEOUT_S = 120.0

_ACRONYMS_RE = re.compile(
    r"\b(?:CGPA|SGPA|GPA|FAST|NUCES|HEC|BBA|BS|MBA|MS|PhD|FYP|OPAC|ILL|ORIC|Rs|PKR|HoD)\b"
)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!Rs)(?<!Dr)(?<!Mr)(?<!Ms)(?<!e\.g)(?<!i\.e)(?<!\d)[.!?]+(?:\s+|$)"
)
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")

JudgeFn = Callable[[str, str, str], tuple[bool | None, str]]


def percentile(values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0..100) for a non-empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def is_claim_bearing(sentence: str) -> bool:
    """Return True if sentence contains a number or a proper noun / known acronym."""
    if re.search(r"\d", sentence):
        return True
    if _ACRONYMS_RE.search(sentence):
        return True
    tokens = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", sentence)
    for tok in tokens[1:]:
        if tok[0].isupper():
            return True
    return False


def is_answer_faithful(answer_text: str) -> bool:
    """Return True iff every claim-bearing sentence in answer_text contains a [N] citation."""
    sentences = _split_sentences(answer_text)
    if not sentences:
        return False
    for sent in sentences:
        if is_claim_bearing(sent) and not _CITATION_MARKER_RE.search(sent):
            return False
    return True


def citation_matches_retrieved(citation: dict[str, Any], retrieved: list[dict[str, Any]]) -> bool:
    """Return True iff citation's source (+ page when given) appears in retrieved."""
    c_src = str(citation.get("source") or "")
    c_page = citation.get("page")
    if not c_src:
        return False
    for r in retrieved:
        if str(r.get("source") or "") != c_src:
            continue
        if c_page is None or r.get("page") == c_page:
            return True
    return False


def compute_judge_human_agreement(
    per_question: list[dict[str, Any]],
    labels: dict[str, bool],
) -> float | None:
    """Compute fraction of hand-labelled answers where judge_faithful == human_faithful."""
    if not labels:
        return None
    by_id = {str(r["id"]): r for r in per_question}
    compared = 0
    matched = 0
    for qid, human_val in labels.items():
        row = by_id.get(qid)
        if not row:
            continue
        judge_val = row.get("judge_faithful")
        if judge_val is None:
            continue
        compared += 1
        if bool(judge_val) == bool(human_val):
            matched += 1
    if compared == 0:
        return None
    return round(matched / compared, 4)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _load_labels(labels_path: Path | None) -> dict[str, bool]:
    if labels_path is None or not labels_path.exists():
        return {}
    out: dict[str, bool] = {}
    for row in _load_jsonl(labels_path):
        qid = str(row.get("question_id") or "")
        if qid and "human_faithful" in row:
            out[qid] = bool(row["human_faithful"])
    return out


def _ask_once_or_retry(
    client: Any,
    api_url: str,
    question: str,
    qid: str,
) -> tuple[dict[str, Any] | None, float]:
    url = f"{api_url.rstrip('/')}/api/ask"
    headers = {"X-Request-ID": f"e2e-{qid}", "X-Cache-Bypass": "1"}
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            resp = client.post(
                url,
                json={"question": question},
                headers=headers,
                timeout=REQUEST_TIMEOUT_S,
            )
            if resp.status_code in {429, 500, 502, 503, 504}:
                if attempt == 0:
                    continue
                return None, round((time.perf_counter() - t0) * 1000.0, 2)
            resp.raise_for_status()
            return resp.json(), round((time.perf_counter() - t0) * 1000.0, 2)
        except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError):
            if attempt == 0:
                continue
            return None, round((time.perf_counter() - t0) * 1000.0, 2)
    return None, round((time.perf_counter() - t0) * 1000.0, 2)


def _format_evidence(retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "(none)"
    blocks: list[str] = []
    for i, item in enumerate(retrieved, 1):
        head = (
            f"[{i}] source={item.get('source', '')} | "
            f"section={item.get('section', '')} | page={item.get('page', '')}"
        )
        body = str(item.get("text") or "").strip()
        blocks.append(f"{head}\n{body}" if body else head)
    return "\n\n".join(blocks)


def make_ollama_judge_fn(
    ollama_url: str,
    model: str,
    http_client: Any | None = None,
) -> JudgeFn:
    """Build a strict Ollama LLM-as-judge callable with 1 retry on JSON parse failure."""
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"

    def _judge(question: str, evidence: str, answer: str) -> tuple[bool | None, str]:
        prompt = (
            "You are a strict judge. Given EVIDENCE and ANSWER, reply JSON only:\n"
            "{faithful: bool, reason: string}. Faithful means every claim in ANSWER is\n"
            "directly supported by EVIDENCE. Do not be lenient.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            f"ANSWER:\n{answer}"
        )
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        owned_client = http_client is None
        client = http_client if http_client is not None else httpx.Client(timeout=REQUEST_TIMEOUT_S)
        try:
            for attempt in range(2):
                try:
                    resp = client.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT_S)
                    resp.raise_for_status()
                    data = resp.json()
                    raw = data.get("response") if isinstance(data, dict) else None
                    if not isinstance(raw, str):
                        if attempt == 0:
                            continue
                        return None, "judge missing response string"
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and isinstance(parsed.get("faithful"), bool):
                        return bool(parsed["faithful"]), str(parsed.get("reason") or "")
                    if attempt == 0:
                        continue
                    return None, "judge JSON missing boolean 'faithful'"
                except (json.JSONDecodeError, httpx.HTTPError, ValueError, TypeError) as exc:
                    if attempt == 0:
                        continue
                    return None, f"judge failed: {exc}"
            return None, "judge failed"
        finally:
            if owned_client:
                client.close()

    return _judge


def evaluate_e2e(
    golden_path: Path,
    api_url: str = DEFAULT_API_URL,
    http_client: Any | None = None,
    judge_fn: JudgeFn | None = None,
    labels_path: Path | None = DEFAULT_LABELS_PATH,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_LLM_MODEL,
    skip_judge: bool = False,
) -> dict[str, Any]:
    """Run all Part B4 end-to-end answer checks against API (+ optional Ollama judge)."""
    questions = _load_jsonl(golden_path)
    labels = _load_labels(labels_path)

    owned_client = http_client is None
    client = http_client if http_client is not None else httpx.Client(timeout=REQUEST_TIMEOUT_S)
    active_judge: JudgeFn | None = None
    if not skip_judge:
        active_judge = judge_fn if judge_fn is not None else make_ollama_judge_fn(ollama_url, model)

    n_answerable = 0
    n_must_abstain = 0
    faithful_ok = 0
    faithful_total = 0
    abstain_ok = 0
    citations_valid = 0
    citations_total = 0
    latencies_ms: list[float] = []
    per_question: list[dict[str, Any]] = []

    try:
        for q in questions:
            qid = str(q.get("id", ""))
            qtext = str(q.get("question", ""))
            qtype = str(q.get("type", ""))
            must_abstain = bool(q.get("must_abstain")) or qtype == "unanswerable"
            if must_abstain:
                n_must_abstain += 1
            else:
                n_answerable += 1

            payload, lat_ms = _ask_once_or_retry(client, api_url, qtext, qid)
            latencies_ms.append(lat_ms)

            if payload is None:
                per_question.append(
                    {
                        "id": qid,
                        "question": qtext,
                        "answer": "",
                        "citations": [],
                        "abstained": True,
                        "latency_ms": lat_ms,
                        "judge_faithful": None,
                        "judge_reason": "api request failed",
                    }
                )
                if not must_abstain:
                    faithful_total += 1
                continue

            answer_text = str(payload.get("answer") or "")
            citations = list(payload.get("citations") or [])
            retrieved = list(payload.get("retrieved") or [])
            afc = payload.get("answered_from_context")
            abstained = not bool(afc)

            if must_abstain:
                if afc is False:
                    abstain_ok += 1
            else:
                if not abstained:
                    faithful_total += 1
                    if is_answer_faithful(answer_text):
                        faithful_ok += 1
                else:
                    faithful_total += 1

            for cit in citations:
                if isinstance(cit, dict):
                    citations_total += 1
                    if citation_matches_retrieved(cit, retrieved):
                        citations_valid += 1

            if active_judge is not None and not abstained:
                j_faithful, j_reason = active_judge(
                    qtext, _format_evidence(retrieved), answer_text
                )
            elif active_judge is not None and abstained:
                j_faithful, j_reason = (True if must_abstain else False), "abstained"
            else:
                j_faithful, j_reason = None, "skipped"

            per_question.append(
                {
                    "id": qid,
                    "question": qtext,
                    "answer": answer_text,
                    "citations": citations,
                    "abstained": abstained,
                    "latency_ms": lat_ms,
                    "judge_faithful": j_faithful,
                    "judge_reason": j_reason,
                }
            )
    finally:
        if owned_client:
            client.close()

    faithfulness = round(faithful_ok / faithful_total, 4) if faithful_total else 1.0
    abstention_correctness = round(abstain_ok / n_must_abstain, 4) if n_must_abstain else 1.0
    citation_validity = round(citations_valid / citations_total, 4) if citations_total else 1.0
    agreement = compute_judge_human_agreement(per_question, labels)

    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_questions": len(questions),
        "n_answerable": n_answerable,
        "n_must_abstain": n_must_abstain,
        "faithfulness": faithfulness,
        "abstention_correctness": abstention_correctness,
        "citation_validity": citation_validity,
        "latency_p50": round(percentile(latencies_ms, 50.0), 2),
        "latency_p95": round(percentile(latencies_ms, 95.0), 2),
        "judge_human_agreement": agreement,
        "per_question": per_question,
    }


def write_results(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Create out_dir if needed and write timestamped e2e_<UTC>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"e2e_{ts_slug}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


def format_summary_table(report: dict[str, Any]) -> str:
    """Render a human-readable summary table for stdout."""
    agree = report["judge_human_agreement"]
    agree_str = f"{agree:.1%}" if agree is not None else "n/a"
    lines = [
        "=== End-to-End Answer Evaluation (Part B4) ===",
        f"Questions              : {report['n_questions']} (answerable={report['n_answerable']}, must_abstain={report['n_must_abstain']})",
        f"Faithfulness (cited)   : {report['faithfulness']:.4f}",
        f"Abstention correctness : {report['abstention_correctness']:.4f}",
        f"Citation validity      : {report['citation_validity']:.4f}",
        f"Latency p50 / p95 (ms) : {report['latency_p50']:.2f} / {report['latency_p95']:.2f}",
        f"Judge/human agreement  : {agree_str}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Part B4 end-to-end answer evaluation.")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API_URL))
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_e2e(
        golden_path=args.golden,
        api_url=args.api_url,
        labels_path=args.labels,
        ollama_url=args.ollama_url,
        model=args.model,
        skip_judge=args.skip_judge,
    )
    out_path = write_results(report, out_dir=args.out)
    print(format_summary_table(report))
    print(f"Report written         : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
