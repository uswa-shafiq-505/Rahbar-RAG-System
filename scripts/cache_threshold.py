"""Choose CACHE_THRESHOLD with data (section 3.3): the lowest threshold with zero false hits.

Edit PARAPHRASES (pairs that mean the same: should HIT) and NEAR_MISSES (pairs about the same
topic but a different question: must MISS) to use questions about YOUR corpus, then run:

    python scripts/cache_threshold.py            # Ollama on http://localhost:11434

Copy the printed table into docs/PIPELINE.md and the false hit that ruled a threshold out
into docs/adr/0003-cache-store-redis.md.
"""

from __future__ import annotations

import math
import os
import sys

import httpx

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

PARAPHRASES = [
    ("What is the minimum attendance requirement?", "How much attendance do I need to be eligible for exams?"),
    ("What CGPA puts a student on academic warning?", "Below which CGPA is a student placed on warning?"),
    ("How much is the tuition fee per credit hour?", "What is the per credit hour tuition?"),
    ("Can I repeat a course I failed?", "Am I allowed to retake a failed course?"),
]
NEAR_MISSES = [
    ("What is the minimum attendance requirement?", "What is the penalty for missing a lab session?"),
    ("What CGPA puts a student on academic warning?", "What CGPA is needed for the dean's honour list?"),
    ("How much is the tuition fee per credit hour?", "What is the refund policy for tuition fees?"),
    ("Can I repeat a course I failed?", "Can I repeat a course I passed to improve my grade?"),
]


def embed(text: str) -> list[float]:
    r = httpx.post(f"{URL}/api/embed", json={"model": MODEL, "input": text}, timeout=60.0)
    r.raise_for_status()
    v = r.json()["embeddings"][0]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a: str, b: str) -> float:
    return sum(x * y for x, y in zip(embed(a), embed(b), strict=False))


def main() -> int:
    para = [cos(a, b) for a, b in PARAPHRASES]
    near = [cos(a, b) for a, b in NEAR_MISSES]
    print(f"model={MODEL}\n")
    print("paraphrases (should hit):")
    for (a, b), s in zip(PARAPHRASES, para, strict=True):
        print(f"  {s:.3f}  {a!r} ~ {b!r}")
    print("near misses (must NOT hit):")
    for (a, b), s in zip(NEAR_MISSES, near, strict=True):
        print(f"  {s:.3f}  {a!r} ~ {b!r}")
    print("\n| threshold | paraphrases hit | false hits |\n|---|---|---|")
    for t in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95):
        print(f"| {t:.2f} | {sum(s >= t for s in para)}/{len(para)} | {sum(s >= t for s in near)} |")
    safe = [t / 100 for t in range(70, 100) if all(s < t / 100 for s in near)]
    if safe:
        print(f"\nlowest threshold with zero false hits: {safe[0]:.2f}")
    else:
        print("\nno threshold below 1.00 avoids every false hit: add near-misses, or do not cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
