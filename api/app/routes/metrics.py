"""Prometheus metrics at /metrics (section 4.8).

Request count and latency are wired up already. Add your own the same way:
retrieval latency, generation latency, cache hits, abstentions, fallbacks, tokens.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

router = APIRouter()

REQUESTS = Counter("rahbar_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "rahbar_request_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

_STAGE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)
RETRIEVAL_SECONDS = Histogram("rahbar_retrieval_seconds", "Retrieval latency",
                              buckets=_STAGE_BUCKETS)
GENERATION_SECONDS = Histogram("rahbar_generation_seconds", "LLM generation latency",
                               buckets=_STAGE_BUCKETS)
CACHE_LOOKUPS = Counter("rahbar_cache_lookups_total", "Semantic cache lookups", ["result"])
CACHE_HIT_RATIO = Gauge("rahbar_cache_hit_ratio", "Cache hits / lookups since process start")
ABSTENTIONS = Counter("rahbar_abstentions_total", "Answers declined: corpus lacks the answer")
FALLBACKS = Counter("rahbar_fallbacks_total", "Safe canned responses after a validation failure")
TOKENS = Counter("rahbar_tokens_total", "Approximate LLM tokens", ["direction"])


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
