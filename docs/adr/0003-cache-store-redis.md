# ADR 0003: The semantic cache lives in Redis

## Status
Accepted · 5 Nov 2026

## Context
Generation is by far the slowest and most expensive step of a request. Students ask the same
few questions in different words ("attendance requirement", "minimum attendance"), so an
exact-match cache would almost never hit; a cache that compares meaning would. The choice was
between Redis and a second Chroma collection.

## Decision
**Redis** (`k8s/base/redis.yaml`, Deployment + PVC `cache-data`), accessed only by the api.
Code: `api/app/services/cache.py`. Each index build has one Redis list under
`rahbar:cache:<fingerprint>`, holding the question text, its unit-length embedding and the
full response. A lookup embeds the new question and takes the best dot product (equal to
cosine for unit vectors); at or above `CACHE_THRESHOLD` (ConfigMap) it is a hit.

Rules enforced in code (`never_cache`, `cache.py`):
- Never cache follow-ups, abstentions, fallbacks, answers without citations, or answers whose
  citations do not match the retrieved passages. A cached mistake is served to everyone.
- **Invalidate on re-index.** The key contains a fingerprint of the index (collection id,
  chunk count, optional `index_version`). When a pod sees a new fingerprint it deletes every
  cache list of older indexes. The indexer never touches Redis, which keeps the NetworkPolicy
  "redis reachable from api only".

**Threshold.** Chosen with data: the lowest value with zero false hits on the test questions
(see the table in `docs/PIPELINE.md`; fill in the thresholds tried and the false hit that
ruled the lowest out). The shipped value is in `k8s/base/configmap.yaml`.

## Options considered
- **A second Chroma collection.** One fewer moving part, but the cache would share a store and
  a PVC with the index, the indexer's collection swap could delete it by accident, and the
  api would need write access to Chroma, breaking "the indexer is the only writer".
- **Redis with RediSearch vector indexes.** Faster at scale, but needs the Redis Stack image
  (larger, more privileges). The cache is capped at 200 entries, so a plain list scanned in
  Python costs milliseconds.
- **In-process dict.** Each of 3 replicas would have its own cache and lose it on restart.

## Consequences
- One more StatefulSet-like dependency: `/readyz` also checks Redis.
- Any cache or embedding failure degrades to "no cache", never to an error
  (`cache_layer.py`).
- With more than a few thousand entries a vector index would be needed.
