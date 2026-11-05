# ADR 0001: The indexer is a Job, not API start-up code

## Status
Accepted · 5 Nov 2026

## Context
RAHBAR has two very different workloads. Querying runs on every question, must be fast and
must scale horizontally (the api Deployment runs 3 to 8 replicas). Indexing (extract, clean,
chunk, embed, store) runs once and again only when the documents change, takes minutes and
holds far more memory than a query. The obvious first design is to index when the API starts.
With 3 replicas and one Chroma volume that design breaks in four ways:

1. **Three writers race.** Every replica starts at the same time, so three processes write
   one collection, each halfway through a different document. The result is an index that no
   single run of the pipeline produced.
2. **Every restart re-indexes.** A rolling update, an OOMKill, a node drain or one readiness
   blip pays the whole extraction and embedding cost again, and during it the pod is useless.
3. **Readiness becomes a lie.** Either the pod reports ready before the index exists and
   serves empty retrievals, or it takes minutes to become ready and the startup probe kills
   it halfway, so a rolling update swaps three working pods for three that are still reading
   PDFs.
4. **No re-index without a redeploy.** A changed document or a new chunk size would need a
   new rollout of the whole service.

## Decision
A one-shot **Job** (`k8s/base/indexer-job.yaml`, image `rahbar-indexer`) is the only writer.
The api only reads. Re-indexing is a command, not a deployment:

    kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar

The weekly `CronJob reindex` (`k8s/base/reindex-cronjob.yaml`) uses the same image and the
same arguments, with `concurrencyPolicy: Forbid` so two writers can never overlap.

The api's `/readyz` (`api/app/routes/health.py`, `api/app/services/index_state.py`) answers
503 until Chroma is reachable, the collection exists and is non-empty, its embedding model
matches `EMBED_MODEL`, and Redis is reachable. A pod is therefore withheld from the Service
until there is something to retrieve, and it is never restarted for waiting.

## Options considered
- **Index in a FastAPI `lifespan` hook.** Rejected for the four reasons above.
- **An initContainer on every api pod.** Same three-writer race, only slower to start.
- **A single "leader" api replica that indexes.** Needs leader election, and still re-indexes
  on every restart of the leader.
- **Index by hand from a laptop.** Works once, but nobody can reproduce it from a clean clone.
  A Job is declared in the repository and is part of the documented apply order.

## Consequences
- The indexer's memory limit is sized for embedding, the api's for answering (the two numbers
  are justified separately in `docs/evidence/kubectl-top.txt`).
- A Job's pod template is immutable, so `make apply` deletes and re-creates it.
- While the Job rebuilds the collection the api keeps serving from the old one until the
  store stage swaps it. How `pipeline/store.py` does that swap is Member 2's decision and is
  recorded in `docs/PIPELINE.md`.
- After a re-index the semantic cache must not serve old answers: see ADR 0003.
- Watch for: a Job in `Error`/`BackoffLimitExceeded` (Chroma not ready, or exit code 137 =
  OOMKilled) and an api stuck at `0/1 READY` (the Job has not finished).
