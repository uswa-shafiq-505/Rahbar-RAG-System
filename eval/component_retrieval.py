"""Component-level retrieval evaluation (Assignment 03 Part B3).

Relevance matching heuristic:
    When exact ground-truth `chunk_id` is absent in `eval/golden.jsonl`, a retrieved
    chunk is treated as relevant if its metadata `source` matches `ground_truth.source`
    AND its `page` is within +/- 1 of `ground_truth.page` (or matching `source` alone
    when `ground_truth.page` is None).

Metrics computed:
    - Answerable questions: Recall@3, Recall@5, MRR, nDCG@5
    - Must-abstain questions: top-1 similarity score (excluded from main averages)
    - Noise test: deterministic 2-typo perturbation (seed=42) reporting clean vs noisy deltas

Run from repository root:
    python -m eval.component_retrieval --golden eval/golden.jsonl --out eval/results/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

EmbedFn = Callable[[str], list[float]]

DEFAULT_GOLDEN_PATH = Path("eval/golden.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")
NOISE_SEED = 42


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary Recall@k: 1.0 if any item in top-k retrieved_ids is in relevant_ids, else 0.0."""
    if k <= 0 or not relevant_ids:
        return 0.0
    return 1.0 if any(cid in relevant_ids for cid in retrieved_ids[:k]) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank for a single query."""
    if not relevant_ids:
        return 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k with binary relevance."""
    if k <= 0 or not relevant_ids:
        return 0.0
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def add_typos(text: str, rng: random.Random) -> str:
    """Apply 2 deterministic typos (swap 2 adjacent letters, drop 1 char) to query text."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]{4,}", text)]
    if not spans:
        return text

    chars = list(text)
    # Typo 1: swap two adjacent interior letters in a chosen word
    s1, e1 = rng.choice(spans)
    swap_candidates = [i for i in range(s1 + 1, e1 - 1) if chars[i] != chars[i + 1]]
    if not swap_candidates and e1 - s1 >= 2:
        swap_candidates = list(range(s1, e1 - 1))
    if swap_candidates:
        idx = rng.choice(swap_candidates)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    mutated = "".join(chars)

    # Typo 2: drop one interior character from a word (prefer a different span if available)
    spans2 = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]{4,}", mutated)]
    if not spans2:
        return mutated
    other_spans = [sp for sp in spans2 if sp[0] != s1] or spans2
    s2, e2 = rng.choice(other_spans)
    drop_idx = rng.randint(s2 + 1, e2 - 2) if e2 - s2 >= 4 else s2
    return mutated[:drop_idx] + mutated[drop_idx + 1 :]


def _load_golden_jsonl(golden_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _dist_to_score(dist: float) -> float:
    return max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))


def _query_collection(
    collection: Any,
    embed_fn: EmbedFn,
    query_text: str,
    n_results: int,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    query_vec = embed_fn(query_text)
    res = collection.query(query_embeddings=[query_vec], n_results=n_results) or {}

    raw_ids = (res.get("ids") or [[]])[0] or []
    raw_metas = (res.get("metadatas") or [[]])[0] or []
    raw_dists = (res.get("distances") or [[]])[0] or []

    chunk_ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    scores: list[float] = []

    count = max(len(raw_ids), len(raw_metas))
    for i in range(count):
        meta = raw_metas[i] if i < len(raw_metas) and isinstance(raw_metas[i], dict) else {}
        cid = (
            str(raw_ids[i])
            if i < len(raw_ids) and raw_ids[i]
            else f"{meta.get('source', 'unknown')}#p{meta.get('page', 0)}#{i}"
        )
        dist = float(raw_dists[i]) if i < len(raw_dists) and raw_dists[i] is not None else 0.0
        chunk_ids.append(cid)
        metadatas.append(meta)
        scores.append(round(_dist_to_score(dist), 6))

    return chunk_ids, metadatas, scores


def _matching_relevant_ids(
    chunk_ids: list[str],
    metadatas: list[dict[str, Any]],
    gt: dict[str, Any],
) -> set[str]:
    gt_chunk_id = gt.get("chunk_id")
    if gt_chunk_id:
        return {str(gt_chunk_id)}

    gt_source = str(gt.get("source") or "")
    gt_page_raw = gt.get("page")
    gt_page = int(gt_page_raw) if gt_page_raw is not None else None

    relevant: set[str] = set()
    for cid, meta in zip(chunk_ids, metadatas):
        if str(meta.get("source") or "") != gt_source:
            continue
        if gt_page is None:
            relevant.add(cid)
            continue
        meta_page = meta.get("page")
        if meta_page is not None and abs(int(meta_page) - gt_page) <= 1:
            relevant.add(cid)
    return relevant


def _mean(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def evaluate_retrieval(
    golden_path: Path,
    chroma_client: Any,
    collection_name: str,
    embed_fn: EmbedFn,
    k_values: tuple[int, ...] = (3, 5),
    embed_model: str | None = None,
) -> dict[str, Any]:
    """Run Part B3 retrieval evaluation (clean + deterministic noise test)."""
    questions = _load_golden_jsonl(golden_path)
    collection = chroma_client.get_collection(name=collection_name)
    max_k = max(k_values)
    k3 = k_values[0] if len(k_values) >= 1 else 3
    k5 = k_values[1] if len(k_values) >= 2 else max_k

    r3_list: list[float] = []
    r5_list: list[float] = []
    mrr_list: list[float] = []
    ndcg_list: list[float] = []
    per_question: list[dict[str, Any]] = []

    answerable_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    n_must_abstain = 0

    for q in questions:
        qid = str(q.get("id", ""))
        qtype = str(q.get("type", ""))
        qtext = str(q.get("question", ""))
        is_abstain = bool(q.get("must_abstain")) or qtype == "unanswerable"

        if is_abstain:
            n_must_abstain += 1
            _, _, scores = _query_collection(collection, embed_fn, qtext, max_k)
            top1 = scores[0] if scores else 0.0
            per_question.append(
                {
                    "id": qid,
                    "type": qtype or "unanswerable",
                    "recall_at_3": None,
                    "recall_at_5": None,
                    "mrr": None,
                    "ndcg_at_5": None,
                    "top1_score": top1,
                }
            )
            continue

        gt = q.get("ground_truth")
        if not isinstance(gt, dict) or not gt.get("source"):
            continue

        answerable_items.append((q, gt))
        chunk_ids, metadatas, scores = _query_collection(collection, embed_fn, qtext, max_k)
        rel_ids = _matching_relevant_ids(chunk_ids, metadatas, gt)

        r3 = recall_at_k(chunk_ids, rel_ids, k3)
        r5 = recall_at_k(chunk_ids, rel_ids, k5)
        q_mrr = mrr(chunk_ids, rel_ids)
        q_ndcg = ndcg_at_k(chunk_ids, rel_ids, k5)
        top1 = scores[0] if scores else 0.0

        r3_list.append(r3)
        r5_list.append(r5)
        mrr_list.append(q_mrr)
        ndcg_list.append(q_ndcg)
        per_question.append(
            {
                "id": qid,
                "type": qtype or "answerable",
                "recall_at_3": round(r3, 6),
                "recall_at_5": round(r5, 6),
                "mrr": round(q_mrr, 6),
                "ndcg_at_5": round(q_ndcg, 6),
                "top1_score": top1,
            }
        )

    clean_r3 = _mean(r3_list)
    clean_r5 = _mean(r5_list)
    clean_mrr = _mean(mrr_list)
    clean_ndcg = _mean(ndcg_list)

    # Noise test with fixed seed 42
    rng = random.Random(NOISE_SEED)
    noisy_r3_list: list[float] = []
    noisy_r5_list: list[float] = []
    noisy_mrr_list: list[float] = []
    noisy_ndcg_list: list[float] = []

    for q, gt in answerable_items:
        noisy_text = add_typos(str(q.get("question", "")), rng)
        chunk_ids, metadatas, _ = _query_collection(collection, embed_fn, noisy_text, max_k)
        rel_ids = _matching_relevant_ids(chunk_ids, metadatas, gt)
        noisy_r3_list.append(recall_at_k(chunk_ids, rel_ids, k3))
        noisy_r5_list.append(recall_at_k(chunk_ids, rel_ids, k5))
        noisy_mrr_list.append(mrr(chunk_ids, rel_ids))
        noisy_ndcg_list.append(ndcg_at_k(chunk_ids, rel_ids, k5))

    noisy_r3 = _mean(noisy_r3_list)
    noisy_r5 = _mean(noisy_r5_list)
    noisy_mrr = _mean(noisy_mrr_list)
    noisy_ndcg = _mean(noisy_ndcg_list)

    model_name = embed_model or os.environ.get("EMBED_MODEL", "nomic-embed-text")
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embed_model": model_name,
        "collection": collection_name,
        "n_questions": len(per_question),
        "n_answerable": len(answerable_items),
        "n_must_abstain": n_must_abstain,
        "metrics": {
            "recall_at_3": clean_r3,
            "recall_at_5": clean_r5,
            "mrr": clean_mrr,
            "ndcg_at_5": clean_ndcg,
        },
        "noise_test": {
            "recall_at_3_clean": clean_r3,
            "recall_at_3_noisy": noisy_r3,
            "delta_r3": round(noisy_r3 - clean_r3, 6),
            "recall_at_5_clean": clean_r5,
            "recall_at_5_noisy": noisy_r5,
            "delta_r5": round(noisy_r5 - clean_r5, 6),
            "mrr_clean": clean_mrr,
            "mrr_noisy": noisy_mrr,
            "delta_mrr": round(noisy_mrr - clean_mrr, 6),
            "ndcg_at_5_clean": clean_ndcg,
            "ndcg_at_5_noisy": noisy_ndcg,
            "delta_ndcg": round(noisy_ndcg - clean_ndcg, 6),
        },
        "per_question": per_question,
    }


def make_ollama_embed_fn(embed_model: str, ollama_url: str, timeout: float = 60.0) -> EmbedFn:
    """Build an embedding function calling Ollama's /api/embeddings endpoint."""
    endpoint = f"{ollama_url.rstrip('/')}/api/embeddings"

    def _embed(text: str) -> list[float]:
        resp = requests.post(
            endpoint,
            json={"model": embed_model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        vec = [float(x) for x in resp.json()["embedding"]]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    return _embed


def write_results(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Create out_dir if needed and write timestamped component_retrieval_<UTC>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"component_retrieval_{ts_slug}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


def format_summary_table(report: dict[str, Any]) -> str:
    """Format a human-readable summary table with clean metrics and noise deltas."""
    m = report["metrics"]
    nt = report["noise_test"]
    lines = [
        "=== Component Retrieval Evaluation (Part B3) ===",
        f"Collection     : {report['collection']}  |  Model: {report['embed_model']}",
        f"Questions      : {report['n_questions']} (answerable={report['n_answerable']}, must_abstain={report['n_must_abstain']})",
        "",
        f"{'Metric':<14} {'Clean':>10} {'Noisy':>10} {'Delta':>10}",
        f"{'-'*14} {'-'*10} {'-'*10} {'-'*10}",
        f"{'recall@3':<14} {m['recall_at_3']:>10.4f} {nt['recall_at_3_noisy']:>10.4f} {nt['delta_r3']:>+10.4f}",
        f"{'recall@5':<14} {m['recall_at_5']:>10.4f} {nt['recall_at_5_noisy']:>10.4f} {nt['delta_r5']:>+10.4f}",
        f"{'mrr':<14} {m['mrr']:>10.4f} {nt['mrr_noisy']:>10.4f} {nt['delta_mrr']:>+10.4f}",
        f"{'ndcg@5':<14} {m['ndcg_at_5']:>10.4f} {nt['ndcg_at_5_noisy']:>10.4f} {nt['delta_ndcg']:>+10.4f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Part B3 component-level retrieval evaluation.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chroma-host", default=os.environ.get("CHROMA_HOST", "localhost"))
    parser.add_argument("--chroma-port", type=int, default=int(os.environ.get("CHROMA_PORT", "8000")))
    parser.add_argument("--collection", default=os.environ.get("CHROMA_COLLECTION", "rahbar"))
    parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    args = parser.parse_args(argv)

    import chromadb

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    embed_fn = make_ollama_embed_fn(args.embed_model, ollama_url)

    report = evaluate_retrieval(
        golden_path=args.golden,
        chroma_client=client,
        collection_name=args.collection,
        embed_fn=embed_fn,
        embed_model=args.embed_model,
    )
    out_path = write_results(report, out_dir=args.out)
    print(format_summary_table(report))
    print(f"Report written : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
