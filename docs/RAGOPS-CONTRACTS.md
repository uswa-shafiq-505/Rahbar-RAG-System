# RAGOps contracts (Assignment 03): what the platform side and the RAG side agree on

Member 1 (platform) owns the Kubernetes objects, the gate, the trace tools and the metrics.
Member 2 (RAG) owns ingestion, verification, evaluation and the content of a trace. This file
is the interface between them: change it in a pull request, never silently.

## 1. The volume `ragops-data`, mounted at `/data` (env `DATA_DIR=/data`)

| Path | Written by | Content |
|---|---|---|
| `lake/incoming/` | `scripts/load-docs.sh` | documents waiting to be ingested |
| `lake/` | ingestion | accepted documents (the data lake) |
| `quarantine/` | ingestion | documents that failed verification (not indexed) |
| `archive/` | ingestion | superseded versions (restorable) |
| `state/` | ingestion | hashes, chunk manifest per document id |
| `traces/` | api pods | `traces-<pod>.jsonl`, one file per pod |
| `results/` | ingestion, eval Job, gate | verification reports, `<UTC timestamp>.json` result files, `gate-*.json`, `baseline.json` |

## 2. Ingestion (the only writer to Chroma)
`python -m pipeline.run` reads `$DATA_DIR/lake/incoming`, runs verification, moves failures to
`quarantine/`, updates Chroma incrementally and writes `results/verification-<UTC timestamp>.json`.
Environment: `DATA_DIR`, `CHROMA_COLLECTION`, `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `OLLAMA_URL`.
Exit code 0 even when documents are quarantined (that is a result, not a failure); non-zero only
when the run itself broke. The collection metadata keeps `embed_model`, `embed_dim`, `index_version`
(a new value on every build: the api's readiness check and cache use them).
Run as: the `indexer` Job, the `reindex` CronJob (weekly) and `kubectl create job --from=cronjob/reindex <name>`.

## 3. Evaluation suite and result file
`python -m ragops.suite --out /data/results --api-url http://api:8000 --collection rahbar`
(module under `eval/ragops/`, imports allowed from `api/app`). It must send the header
`X-Cache-Bypass: 1` on every request, so the semantic cache cannot distort results. It writes
`/data/results/<UTC timestamp>.json`:

```json
{"meta": {"timestamp": "...", "collection": "rahbar", "embed_model": "nomic-embed-text",
          "index_version": "...", "seed": 42, "git_sha": "..."},
 "metrics": {"recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0,
             "faithfulness": 0.0, "abstention_correctness": 0.0, "citation_validity": 0.0,
             "latency_p50": 0.0, "latency_p95": 0.0}}
```
Ratios are fractions between 0 and 1; latencies are seconds. The module-level metrics (embedding
gap, determinism, time per chunk) and the per-question details go in extra keys of the same file.

## 4. Gate thresholds
`config/gate.yaml` (and the same text in `k8s/base/gate-thresholds.yaml`). Rules: recall@5 drop,
faithfulness floor, citation validity floor, latency p95 ceiling, abstention not below the baseline.
`scripts/gate.py` exits 0 (pass) or 1 (fail). The baseline is `/data/results/baseline.json`.

## 5. Collections and the release
Live collection `rahbar`; candidate `rahbar_candidate` (any other name works). A release is
`make release COLLECTION=<name> EMBED_MODEL=<model>`: a rolling update of the live Deployment
with the new environment. `make rollback` is `kubectl rollout undo`.

## 6. Trace record (one JSON line per query; `trace_sink.write_trace(record)`)
The answer pipeline builds the dict; the sink adds `trace_id`, `ts`, `pod`, `event` when missing.
Use `trace_sink.base_fields(settings)` for `collection`, `embed_model`, `pod`.

| Field | Type | Notes |
|---|---|---|
| `trace_id`, `ts`, `pod` | str | added by the sink if absent |
| `query`, `rewritten_query` | str | rewritten may be absent |
| `embed_model`, `embed_version`, `collection`, `index_version` | str | previous release must be visible after `rollout undo` |
| `retrieved` | list | `{chunk_id, doc_id, doc_version, score}` in retrieval order |
| `filters` | object | for example `{"superseded": false}` |
| `reranked_order` | list of chunk ids | absent when there is no reranker |
| `context` | list | `{chunk_id, text}` of what was put in the prompt (needed by `scripts/trace.py`) |
| `prompt` or `prompt_hash` | str | store the full prompt; the hash alone cannot be rebuilt |
| `model` | object | `{name, params: {temperature, ...}}` |
| `response`, `citations` | str, list | `citations`: `{source, section, page}` |
| `guardrail_actions` | list | `{rail: "injection", action: "removed" or "flagged" or "blocked", detail}` |
| `latency_ms` | object | `embed`, `retrieve`, `rerank`, `generate`, `total` |
| `cache` | str | `hit`, `miss` or `bypass` |
| `abstained` | bool | |

Metric hooks (`app/services/metrics_hooks.py`): `injection_flagged()` whenever the injection
rail fires; `fallback()` on a safe canned response. Top-1 similarity is recorded automatically.

## 7. Files that belong to the groups
`corpus/batch_b/` and `MANIFEST-batch-b.json` (built by `scripts/make_batch_b.py`, Member 1),
`eval/live_queries.jsonl` (`{id, topic, group, query}`, Member 1), `eval/golden.jsonl` (Member 2).
