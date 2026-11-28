"""Module-level embedding evaluation (Assignment 03 Part B2).

Measures:
  1. Mean cosine similarity on paraphrase pairs vs unrelated pairs (+ gap)
  2. Embedding latency per average-length (~500-token) chunk (mean, p50, p95 ms)
  3. Determinism (in-process repeat + fresh-subprocess repeat, max abs diff <= 1e-6)

Run from repository root:
    python -m eval.module_embedding
    python -m eval.module_embedding --skip-network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml  # type: ignore[import-untyped]

EmbedFn = Callable[[str], list[float]]

DEFAULT_PAIRS_PATH = Path("eval/embedding_pairs.yaml")
DEFAULT_OUT_DIR = Path("eval/results")
STUB_DIM = 384
DETERMINISM_THRESHOLD = 1e-6

_BASE_CHUNK_TEXT = (
    "National University of Computer and Emerging Sciences (FAST-NUCES) Undergraduate "
    "Academic Rules and Regulations. Section 5.4: Minimum Cumulative Grade Point Average "
    "(CGPA) required to avoid academic warning is 2.00 for all undergraduate programs. "
    "If at the end of a semester, including summer semester, CGPA drops below 2.00, "
    "academic warning is issued and the warning count is incremented by one. If CGPA "
    "remains below 2.00 for three consecutive semesters, the warning count reaches three "
    "and admission is automatically closed. Section 1.21: Students are required to "
    "maintain 100% attendance in lectures, labs, tutorials, seminars, and discussions; "
    "absence of up to 20% may be condoned by the Head of Department for genuine reasons "
    "such as documented illness, resulting in an 80% minimum threshold to sit finals. "
    "Tuition fee for Fall 2026-27 is Rs. 12,000 per credit hour alongside a one-time "
    "non-refundable admission fee of Rs. 30,000 and a refundable security deposit of "
    "Rs. 20,000. Tuition refunds follow HEC policy up to the 30th day of classes."
)


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


def stub_embed_384(text: str) -> list[float]:
    """Deterministic 384-dim feature-hashed embedding for offline/--skip-network runs."""
    vec = [0.0] * STUB_DIM
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    features = list(tokens)
    for tok in tokens:
        if len(tok) >= 4:
            features.extend(tok[i : i + 3] for i in range(len(tok) - 2))
    if not features:
        features = [text]
    for feat in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % STUB_DIM
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def make_ollama_embed_fn(embed_model: str, ollama_url: str, timeout: float = 60.0) -> EmbedFn:
    """Build an embedding function calling Ollama's /api/embeddings endpoint via requests."""
    endpoint = f"{ollama_url.rstrip('/')}/api/embeddings"

    def _embed(text: str) -> list[float]:
        resp = requests.post(
            endpoint,
            json={"model": embed_model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = [float(x) for x in data["embedding"]]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    return _embed


def _short(text: str, limit: int = 60) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def _load_pairs(pairs_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw = yaml.safe_load(pairs_path.read_text(encoding="utf-8")) or {}
    paraphrase = list(raw.get("paraphrase") or [])
    unrelated = list(raw.get("unrelated") or [])
    return paraphrase, unrelated


def _score_pairs(pairs: list[dict[str, str]], embed_fn: EmbedFn) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for item in pairs:
        va = embed_fn(item["a"])
        vb = embed_fn(item["b"])
        sim = cosine_similarity(va, vb)
        scored.append(
            {
                "id": str(item["id"]),
                "a_short": _short(item["a"]),
                "b_short": _short(item["b"]),
                "sim": round(sim, 6),
            }
        )
    mean_sim = sum(p["sim"] for p in scored) / len(scored) if scored else 0.0
    return {"mean": round(mean_sim, 6), "pairs": scored}


def _make_500_token_chunks(n: int) -> list[str]:
    # ~350 words / ~500 subword tokens per chunk
    base = (_BASE_CHUNK_TEXT + " ") * 3
    return [f"[Chunk {i + 1:02d}] {base.strip()}" for i in range(n)]


def _subprocess_embed_once(text: str, embed_model: str, ollama_url: str, skip_network: bool) -> list[float]:
    cmd = [sys.executable, "-m", "eval.module_embedding", "--embed-once", text]
    if skip_network:
        cmd.append("--skip-network")
    env = dict(os.environ)
    env["EMBED_MODEL"] = embed_model
    env["OLLAMA_URL"] = ollama_url
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return [float(x) for x in json.loads(proc.stdout.strip())]


def evaluate_embeddings(
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    embed_fn: EmbedFn | None = None,
    embed_model: str | None = None,
    ollama_url: str | None = None,
    skip_network: bool = False,
    n_timing: int = 50,
    check_subprocess: bool | None = None,
) -> dict[str, Any]:
    """Run all Part B2 embedding checks and return the structured report dictionary."""
    model_name: str = embed_model or os.environ.get("EMBED_MODEL", "nomic-embed-text")
    url: str = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")

    injected = embed_fn is not None
    if embed_fn is None:
        if skip_network:
            active_embed: EmbedFn = stub_embed_384
            model_name = f"stub-384 ({model_name})"
        else:
            active_embed = make_ollama_embed_fn(model_name, url)
    else:
        active_embed = embed_fn

    paraphrase_pairs, unrelated_pairs = _load_pairs(pairs_path)
    paraphrase_res = _score_pairs(paraphrase_pairs, active_embed)
    unrelated_res = _score_pairs(unrelated_pairs, active_embed)
    gap = round(paraphrase_res["mean"] - unrelated_res["mean"], 6)

    timings_ms: list[float] = []
    for chunk_text in _make_500_token_chunks(n_timing):
        t0 = time.perf_counter()
        active_embed(chunk_text)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(timings_ms) / len(timings_ms) if timings_ms else 0.0
    embed_ms = {
        "mean": round(mean_ms, 3),
        "p50": round(percentile(timings_ms, 50.0), 3),
        "p95": round(percentile(timings_ms, 95.0), 3),
        "n": n_timing,
    }

    probe_text = "Minimum CGPA required to avoid academic warning is 2.00 for undergraduate students."
    v1 = active_embed(probe_text)
    v2 = active_embed(probe_text)
    vectors_to_compare: list[tuple[list[float], list[float]]] = [(v1, v2)]

    run_sub = (not injected) if check_subprocess is None else check_subprocess
    if run_sub:
        v3 = _subprocess_embed_once(probe_text, model_name, url, skip_network=skip_network)
        vectors_to_compare.append((v1, v3))

    max_abs_diff = 0.0
    max_l2_diff = 0.0
    for va, vb in vectors_to_compare:
        if len(va) != len(vb):
            max_abs_diff = float("inf")
            max_l2_diff = float("inf")
            break
        pair_abs = max((abs(a - b) for a, b in zip(va, vb)), default=0.0)
        pair_l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(va, vb)))
        max_abs_diff = max(max_abs_diff, pair_abs)
        max_l2_diff = max(max_l2_diff, pair_l2)

    determinism_pass = max_abs_diff <= DETERMINISM_THRESHOLD and max_l2_diff <= DETERMINISM_THRESHOLD

    now_utc = datetime.now(timezone.utc)
    return {
        "ts": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embed_model": model_name,
        "paraphrase": paraphrase_res,
        "unrelated": unrelated_res,
        "gap": gap,
        "embed_ms": embed_ms,
        "determinism_max_abs_diff": max_abs_diff,
        "determinism_pass": determinism_pass,
    }


def write_results(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Create out_dir if needed and write timestamped module_embedding_<UTC>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"module_embedding_{ts_slug}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


def format_summary_table(report: dict[str, Any]) -> str:
    """Render a concise human-readable summary table for stdout."""
    emb_ms = report["embed_ms"]
    lines = [
        "=== Module Embedding Evaluation (Part B2) ===",
        f"Model                    : {report['embed_model']}",
        f"Timestamp (UTC)          : {report['ts']}",
        f"Paraphrase mean cos-sim  : {report['paraphrase']['mean']:.4f} (n={len(report['paraphrase']['pairs'])})",
        f"Unrelated mean cos-sim   : {report['unrelated']['mean']:.4f} (n={len(report['unrelated']['pairs'])})",
        f"Semantic gap             : {report['gap']:.4f}",
        f"Latency (ms, n={emb_ms['n']:>2})       : mean={emb_ms['mean']:.3f}  p50={emb_ms['p50']:.3f}  p95={emb_ms['p95']:.3f}",
        f"Determinism max |diff|   : {report['determinism_max_abs_diff']:.2e} ({'PASS' if report['determinism_pass'] else 'FAIL'})",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Part B2 module-level embedding evaluation.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_PATH, help="Path to pairs YAML")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--skip-network", action="store_true", help="Use deterministic 384-dim stub")
    parser.add_argument("--embed-once", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    model_name = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    if args.embed_once is not None:
        fn = stub_embed_384 if args.skip_network else make_ollama_embed_fn(model_name, ollama_url)
        print(json.dumps(fn(args.embed_once)))
        return 0

    report = evaluate_embeddings(
        pairs_path=args.pairs,
        embed_model=model_name,
        ollama_url=ollama_url,
        skip_network=args.skip_network,
    )
    out_path = write_results(report, out_dir=args.out)
    print(format_summary_table(report))
    print(f"Report written           : {out_path}")
    return 0 if report["determinism_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

