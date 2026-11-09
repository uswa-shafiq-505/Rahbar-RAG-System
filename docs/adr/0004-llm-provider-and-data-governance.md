# ADR 0004: LLM provider pluggability and data governance

## Status
Accepted · 2026-11-07

## Context
RAHBAR requires generation capabilities over indexed institutional policies while supporting heterogeneous operational environments: local development, air-gapped CI/testing, Minikube Kubernetes clusters, and potential cloud environments. The architecture needed:
1. A provider-agnostic LLM interface that isolates application logic from downstream model hosting APIs.
2. Robust data governance and defense against indirect prompt injections contained within untrusted corpus documents (e.g. adversarial student manuals or syllabus files containing embedded override instructions).

## Decision
1. **Interface Abstraction**: All LLM operations sit behind the `LLMProvider` interface (`api/app/providers/llm/base.py`), dynamically dispatched via the `LLM_PROVIDER` environment variable:
   - `ollama-host` (**Chosen default**): Connects to Ollama running on the host machine (`http://host.minikube.internal:11434` or WSL bridge), avoiding heavy GPU workloads inside Minikube.
   - `ollama-incluster`: Dedicated in-cluster DaemonSet/Deployment for air-gapped deployments (bonus, unselected due to resource footprints).
   - `hosted`: Third-party APIs (e.g., Groq, Google AI Studio) authenticated using credentials stored exclusively in Kubernetes Secrets.
2. **Untrusted Data Isolation & Delimiting**:
   - Retrieved passages are strictly treated as data rather than executable instructions.
   - Evidence is bracketed between explicit bounding delimiters: `<<<EVIDENCE_START>>>` and `<<<EVIDENCE_END>>>`.
   - The system prompt contains explicit guardrail instructions commanding the model to ignore any directive (such as "ignore all previous instructions") embedded within the evidence blocks.

## Consequences
- **Portability**: Switching from local Ollama to external cloud inference requires only modifying `LLM_PROVIDER` and supplying `LLM_API_KEY` without changing orchestration code.
- **Credential Hygiene**: `LLM_API_KEY` is never committed to Git and is injected via `kubectl create secret generic rahbar-secrets --from-literal=...`.
- **Indirect Prompt Injection Defense**: Tested and verified via `api/tests/test_poison.py`; injected instructions inside chunk bodies remain inert within data delimiters.
- **Trade-off**: Retrieval throughput and generation quality remain bounded by context-window truncation (2,000 chars) for oversized chunks.

