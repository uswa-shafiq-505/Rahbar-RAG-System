# ADR 0005: A pipeline change is released by switching collection, behind a gate

## Status
Accepted

## Context
A change to chunking, the embedding model or top-k changes the vector index, not just code. Vectors
from two embedding models are not comparable, so a candidate cannot be tried by editing the live
collection in place: a failed experiment would damage the live service, and there would be nothing
to roll back to.

## Decision
The candidate is built into its **own collection** (`candidate-ingest` Job) and served by its **own
Deployment and Service** (`api-candidate`, cache off). The evaluation Job runs the suite against
that Service and the gate decides (`scripts/gate.py`, thresholds in the `gate-thresholds`
ConfigMap). Only a passing candidate is released: the live Deployment gets the new
`CHROMA_COLLECTION` and `EMBED_MODEL` through a rolling update (`maxSurge: 1, maxUnavailable: 0`),
watched with `kubectl rollout status`. A bad release is reversed with `kubectl rollout undo`: the
old pod template, with the old collection and model, comes back, and the next trace names them.

## Consequences
- The live collection is never rebuilt in place by an experiment.
- The readiness check (collection exists, non-empty, embedding model matches) stops a pod that
  points at a wrong or empty collection from taking traffic, so a bad release stalls instead of serving.
- Old collections stay in Chroma until someone deletes them: they cost disk, and they are what
  makes the rollback instant.
- `kubectl rollout undo` reverts the environment of the Deployment only; a ConfigMap edit is not
  versioned by Kubernetes, which is why the release changes env on the Deployment, not the ConfigMap.
