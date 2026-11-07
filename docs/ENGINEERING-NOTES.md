# Engineering notes

> Answers 1, 2, 3, 4 and 6 are written by Member 1 (platform). Answers 5 and 7 are Member 2's
> (and each member adds their own question 7). Numbers marked **MEASURE** must be replaced by
> your own `kubectl top` / `kubectl get` output before you submit: the marker checks them.

## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?
Three api pods start together and all three would run extract, chunk, embed and store against
the same collection on the one Chroma PVC. That is a **write race**: each pod is partway through
a different document, so the index that remains was produced by no single run. A rolling update
or an OOMKill would repeat the whole job, and readiness would either lie (ready before the
index exists) or time out (the startup probe kills a pod that is still reading PDFs).
The object that prevents it is the **Job** in `k8s/base/indexer-job.yaml` (line 10, `kind: Job`;
`restartPolicy: OnFailure` line 18, `backoffLimit: 4` line 12): one pod writes, then exits. The
api only reads, and `/readyz` (`api/app/routes/health.py:17`) returns 503 until the collection
is non-empty (`api/app/services/index_state.py:52`). The weekly re-run uses
`concurrencyPolicy: Forbid` (`k8s/base/reindex-cronjob.yaml`), so two writers never overlap.

## 2. Readiness depends on Chroma, liveness does not. What if we swap them and Chroma is slow for 90 s?
Liveness (`k8s/base/api.yaml:42`) calls `/healthz`, which never touches a dependency, and
readiness (`api.yaml:46`) calls `/readyz`, which checks Chroma, the collection, the pin and Redis.
If they were swapped, `/healthz` would depend on Chroma. Chroma slow for 90 s means every
liveness probe times out three times in a row (`failureThreshold: 3`, `periodSeconds: 20`, so
about 60 s). The kubelet then **kills and restarts all three api pods at the same moment**, and
each restart makes a cold process that has to reconnect to a Chroma that is still slow. What
was merely slow becomes down: a restart storm with no healthy replica. With the correct wiring
the same 90 s only removes the pods from the Service (readiness fails, nothing is restarted),
requests get a 503 from the Ingress for that time, and the pods come back by themselves as soon
as Chroma answers.

## 3. While the model generates, what is the CPU-based HPA measuring? What should scale it?
`k8s/base/hpa.yaml:15` scales on CPU `averageUtilization: 70`, which is usage divided by
`requests.cpu` (`api.yaml:59`, 100m). While the api waits for the LLM the pod is **blocked on
network I/O**, so its CPU stays near zero even when every request is slow and users queue up.
The HPA therefore measures how busy the Python process is, not how loaded the service is, and
it would not scale during exactly the overload we care about. A better signal is demand:
**in-flight requests per pod** or **request latency (p95) of `rahbar_generation_seconds`**,
both of which `/metrics` already exposes. To use it we would run Prometheus (or
metrics-server's custom-metrics path via the Prometheus adapter or KEDA) and point an HPA
`Pods`/`External` metric at it. We did not build that; CPU is kept because it is required and
cheap, and this answer documents its limit.

## 4. Indexer peak memory: limit, margin, exit code
Peak memory of the indexer during embedding, from `kubectl top pods --containers`
(`docs/evidence/kubectl-top.txt`): **MEASURE: ___ Mi**. The limit is 1536Mi
(`k8s/base/indexer-job.yaml:41`) and the request is 512Mi (`:38`), so the margin is
**MEASURE: limit minus peak = ___ Mi (___ %)**. Aim for at least 30 % headroom, because OCR on
the scanned PDF spikes above embedding. If the limit is set too low the kernel OOM-kills the
container and the pod shows `OOMKilled` with **exit code 137** (128 + signal 9); the Job
restarts it up to `backoffLimit: 4` times and then fails, which is the most common
CrashLoopBackOff in this assignment. CPU over its limit is only throttled (slow but alive);
memory over its limit kills.

## 5. recall@5 by question type
Member 2 writes this answer from `eval/results.json`.

## 6. EMBED_MODEL changed on the API only
The api compares the collection's recorded model with its own on every readiness probe
(`api/app/services/index_state.py:69`, the `mismatch` message) and once at startup
(`api/app/main.py`, lifespan). On a mismatch `/readyz` answers 503 and the log has one
line naming both models, for example `embedding model mismatch: index was built with
'dummy-768' but this API has EMBED_MODEL='nomic-embed-text'`. The pod stays `0/1 READY`, the
Service has no endpoints from it, and nothing returns wrong answers. Because
`maxUnavailable: 0` (`api.yaml:12`), a rolling update with the wrong value never replaces the
old working pods: the rollout simply hangs until it is fixed or undone. Without the guardrail
the new pod would be ready, embed questions with model B and compare them against vectors from
model A (or fail on a dimension error), and retrieval would return plausible nonsense with no
error anywhere, which is the silent failure the assignment warns about.

## 7. The failure that cost us more than an hour
Member 1: write your own, from something that really happened. Candidates: Calico/NetworkPolicy
blocking DNS, `ImagePullBackOff` from `minikube image load`, Chroma pod permission errors on
the PVC, or the Job being immutable on re-apply. State what you believed first and the exact
command that told you the truth.
