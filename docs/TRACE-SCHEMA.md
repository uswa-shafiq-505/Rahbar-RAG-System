# RAHBAR Query Trace Schema (Contract — Part D1)

This document defines the canonical trace record contract for Assignment 03 (RAGOps) query observability. Every call to `POST /api/ask` emits **one JSON line (JSONL)** capturing inputs, retrieval state, prompt identity, guardrail decisions, citations, cache status, and per-stage latency. Each trace line is written simultaneously to **stdout** (for container log collection) and appended to a persistent volume file at `/data/traces/<pod>_<ts>.jsonl` (where `<pod>` is the pod hostname and `<ts>` is the UTC file rotation/startup timestamp). Downstream diagnostic, reporting, coverage, and offline evaluation tools consume this schema directly.

---

## Canonical JSON Example

Formatted for readability (in production logs and `.jsonl` files, each record is emitted as a single line):

```json
{
  "trace_id": "9f8c2d7e-4b3a-4c12-8e71-1d2f9a0b3c45",
  "request_id": "7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d",
  "ts": "2026-11-25T13:05:12.482Z",
  "user_query": "What CGPA is required to avoid academic warning?",
  "rewritten_query": null,
  "embed_model": "nomic-embed-text",
  "embed_model_version": "nomic-embed-text:latest",
  "collection": "rahbar",
  "index_version": "v1",
  "retrieved": [
    {
      "chunk_id": "academic-rules-handbook-jun-2026.pdf::25::0",
      "doc_id": "academic-rules-handbook-jun-2026.pdf",
      "doc_version": "2026-06",
      "source": "academic-rules-handbook-jun-2026.pdf",
      "section": "5. Academic Assessment",
      "page": 25,
      "score": 0.874
    },
    {
      "chunk_id": "grading-page.html::1::2",
      "doc_id": "grading-page.html",
      "doc_version": "2026",
      "source": "grading-page.html",
      "section": "Warning/Probation",
      "page": 1,
      "score": 0.812
    }
  ],
  "filters_applied": {
    "superseded": false
  },
  "reranked_order": [
    "academic-rules-handbook-jun-2026.pdf::25::0",
    "grading-page.html::1::2"
  ],
  "prompt_hash": "5a9a94bff0cad44f88a1dd1140e890ecd22260e17bd2b30d7fb1bbc86bc93636",
  "prompt_text": "You are a strict, grounded question-answering assistant for a university policy corpus.\n\nEvidence:\n[1] source=academic-rules-handbook-jun-2026.pdf | section=5. Academic Assessment | page=25\n5.4 Minimum CGPA required to avoid academic warning is 2.00 for all undergraduate programs.\n\nQuestion: What CGPA is required to avoid academic warning?",
  "model": "llama3.2:3b",
  "model_params": {
    "temperature": 0.2,
    "top_k": 5
  },
  "response": "Undergraduate students must maintain a minimum CGPA of 2.00 to avoid academic warning [1].",
  "citations": [
    {
      "source": "academic-rules-handbook-jun-2026.pdf",
      "section": "5. Academic Assessment",
      "page": 25
    }
  ],
  "guardrail_actions": [
    {
      "rail": "input",
      "action": "allow",
      "reason": "no prompt injection pattern detected"
    },
    {
      "rail": "retrieval",
      "action": "pass",
      "reason": "2 chunks passed relevance_floor=0.35"
    },
    {
      "rail": "output",
      "action": "pass",
      "reason": "1/1 citations matched retrieved chunks"
    }
  ],
  "latency_ms": {
    "embed": 42.3,
    "retrieve": 18.7,
    "rerank": 0.4,
    "generate": 1480.5,
    "validate": 1.1,
    "total": 1543.0
  },
  "cache": "miss",
  "abstained": false
}
```

---

## Field Specification

| Field | Type | Required | Source | Purpose |
| --- | --- | --- | --- | --- |
| `trace_id` | `string` (UUIDv4) | Yes | Generated per query in API | Unique identifier for a single traced query execution |
| `request_id` | `string` | Yes | `X-Request-ID` header or generated hex ID | Correlates trace line with HTTP access logs and `AskResponse.request_id` |
| `ts` | `string` (ISO-8601 UTC) | Yes | API clock at completion (`YYYY-MM-DDTHH:MM:SS.sssZ`) | Enables time-window aggregation and chronological replay |
| `user_query` | `string` | Yes | `AskRequest.question` | Captures exact user input before rewriting or normalization |
| `rewritten_query` | `string \| null` | Yes | Query rewriter (or `null` when unchanged) | Records any query expansion/reformulation sent to retrieval |
| `embed_model` | `string` | Yes | `EMBED_MODEL` ConfigMap/env setting | Tracks configured query embedding model for drift/pin checks |
| `embed_model_version` | `string` | Yes | Chroma collection metadata | Verifies exact embedding model/tag active in the queried index |
| `collection` | `string` | Yes | `CHROMA_COLLECTION` setting | Distinguishes production (`"rahbar"`) vs candidate (`"rahbar_candidate"`) traces |
| `index_version` | `string` | Yes | Chroma collection metadata | Pins trace to the corpus/index build version |
| `retrieved` | `array<object>` | Yes | Retriever output (`chunk_id`, `doc_id`, `doc_version`, `source`, `section`, `page`, `score`) | Audits every candidate chunk returned from the vector store |
| `filters_applied` | `object` | Yes | Retriever query filters (e.g. `{"superseded": false}`) | Explains metadata exclusions (such as superseded versions) during root-cause analysis |
| `reranked_order` | `array<string>` | Yes | Reranker / prompt evidence selector | Ordered list of `chunk_id`s matching the exact evidence order in the prompt |
| `prompt_hash` | `string` (SHA-256 hex) | Yes | SHA-256 digest of exact prompt bytes sent to LLM (empty string `""` if LLM not called) | Verifies prompt template/context reproducibility without always logging full text |
| `prompt_text` | `string \| null` | Yes | Exact prompt sent to LLM when `TRACE_INCLUDE_PROMPT=true`, else `null` | Enables full prompt inspection in debugging tools when enabled |
| `model` | `string` | Yes | `LLM_MODEL` ConfigMap/env setting | Identifies the generator model name |
| `model_params` | `object` | Yes | Generator config (`temperature`, `top_k`, and `top_p` if applicable) | Records sampling and retrieval parameters affecting output |
| `response` | `string` | Yes | `AskResponse.answer` | Final answer text returned to the caller (never includes internal `reasoning`) |
| `citations` | `array<object>` | Yes | Post-validation `AskResponse.citations` (`{source, section, page}`) | Captures verified citations surviving citation validation |
| `guardrail_actions` | `array<object>` | Yes | Input, retrieval, and output guardrails (`{rail, action, reason}`) | Audits injection checks, relevance-floor filtering, and citation/output drops |
| `latency_ms` | `object` | Yes | Stage timers (`embed`, `retrieve`, `rerank`, `generate`, `validate`, `total`) | Quantifies p50/p95 stage performance and cache savings |
| `cache` | `string` (`"hit"` \| `"miss"`) | Yes | Semantic cache lookup outcome | Indicates whether the answer was replayed from Redis cache or computed live |
| `abstained` | `bool` | Yes | `not AskResponse.answered_from_context` | Tracks corpus coverage gaps and abstention rate |

### Nested Object Shapes

- **`retrieved[]` element:** `{ "chunk_id": string, "doc_id": string, "doc_version": string, "source": string, "section": string, "page": int | null, "score": float }`
- **`model_params`:** `{ "temperature": float, "top_k": int, "top_p"?: float }` (`top_p` included when configured on the active provider)
- **`citations[]` element:** `{ "source": string, "section": string, "page": int | null }`
- **`guardrail_actions[]` element:** `{ "rail": "input" | "retrieval" | "output", "action": string, "reason": string }`
- **`latency_ms`:** `{ "embed": float, "retrieve": float, "rerank": float, "generate": float, "validate": float, "total": float }` (non-negative milliseconds; `0.0` for skipped stages)

---

## Invariants

1. **Citation-to-retrieval lineage:** Every entry in `citations` MUST match a chunk (`chunk_id` via `(source, section, page)`) present in `retrieved`. Post-validation must drop any fabricated or unretrieved citation before emission.
2. **Abstention consistency:** Whenever `abstained` is `true`, `response` MUST explain that the indexed corpus does not cover the question, and `citations` MUST be empty (`[]`).
3. **Cache-hit semantics:** Whenever `cache` is `"hit"`, `latency_ms.generate` MUST be `0` (`0.0`) because the LLM was not invoked, and `retrieved` may be empty (`[]`) if the cached `AskResponse` replays a prior live answer without re-querying Chroma.
4. **Prompt hash reproducibility:** Whenever `prompt_text` is non-null, `prompt_hash` MUST equal the lowercase hexadecimal SHA-256 digest of the exact UTF-8 bytes sent to the LLM: `hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()`.

---

## Consumer Compatibility

- **`scripts/trace.py <trace_id>` (D2):** Scans the JSONL trace log(s) for the record matching `trace_id`, extracts `user_query`, `retrieved`, `reranked_order`, `filters_applied`, `prompt_hash`, `prompt_text`, `guardrail_actions`, `response`, and `citations`, and prints the exact retrieved context chunks (in `reranked_order`) alongside the prompt that produced the answer.
- **`scripts/trace_report.py` (D3):** Reads trace records over a specified time window using `ts` and aggregates operational health metrics across `retrieved` (top-1 similarity score distribution), `abstained` (abstention rate), `cache` (cache hit rate), `latency_ms` (p95 latency per stage), and `guardrail_actions` (count of flagged injection attempts and rail actions).
- **`eval/e2e_answers.py` (B4):** Consumes `citations`, `abstained`, `latency_ms` (plus `response` and `retrieved`) from API/trace outputs to score citation validity, abstention correctness, faithfulness, and end-to-end p50/p95 latency.
- **`eval/coverage.py` (E2):** Reads `user_query` and `embed_model` from `traces/live_queries.jsonl` (and trace logs) to embed live queries using the matching model, measure nearest-neighbor similarity against `eval/golden.jsonl`, and flag uncovered query topics.

