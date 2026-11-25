# Kubernetes manifests

A fresh cluster reaches a working pipeline by running these commands from the repository root.
Everything lives in the namespace `rahbar`.

## 1. Cluster and images
```bash
make cluster                      # minikube with Calico, metrics-server and the ingress addon
make images                       # api, indexer, ui (stages the corpus into the indexer image)
make eval-image                   # the evaluation and gate image
```

## 2. The platform (apply order is handled by kustomize)
```bash
kubectl apply -k k8s/overlays/dev # namespace, volume, ConfigMaps, Chroma, Redis, api x3, ui,
                                  # Services, Ingress, HPA, NetworkPolicies, indexer Job, CronJob
# Ollama on the host: OLLAMA_HOST=0.0.0.0 ollama serve  (llama3.2:3b and nomic-embed-text pulled)
kubectl wait --for=condition=complete job/indexer -n rahbar --timeout=900s
kubectl rollout status deployment/api -n rahbar
```
`k8s/overlays/dev` still has its stub-mode `patches:` block: delete it to use the real pipeline.

## 3. What each manifest is for
| File | Object | RAGOps responsibility |
|---|---|---|
| `base/ragops-data-pvc.yaml` | PVC `ragops-data` | data lake, quarantine, archive, state, traces and results survive pod restarts |
| `base/chroma-statefulset.yaml` | StatefulSet `chroma` + its own PVC | the vector store, with a stable identity |
| `base/indexer-job.yaml` | Job `indexer` | one-shot ingestion with verification: the only writer to Chroma |
| `base/reindex-cronjob.yaml` | CronJob `reindex` | the same ingestion on a weekly schedule (`concurrencyPolicy: Forbid`) |
| `base/gate-thresholds.yaml` | ConfigMap `gate-thresholds` | release-gate thresholds, editable without rebuilding an image |
| `base/api.yaml` | Deployment `api` x3 + probes | the live query pipeline, rolling update with `maxUnavailable: 0` |
| `base/networkpolicy*.yaml` | NetworkPolicies | default deny; only the paths the architecture needs |
| `eval-job.yaml` | Job `eval-baseline` | suite on the live api, then the gate; exit status = gate result |
| `gate-job.yaml` | Job `gate-check` | the gate alone, on the newest result file |
| `candidate-ingest-job.yaml` | Job `candidate-ingest` | builds the candidate collection |
| `candidate.yaml` | Deployment + Service `api-candidate` | the candidate, on its own collection, cache off |
| `eval-job-candidate.yaml` | Job `eval-candidate` | suite and gate against the candidate |
| `loader-pod.yaml` | Pod `data-loader` | copies documents in and results out of the volume |

## 4. Part by part (the commands the demo uses)
```bash
make load-docs DIR=corpus/batch_b                         # put a batch in the data lake
kubectl create job --from=cronjob/reindex ingest-batch-b -n rahbar   # ingestion Job on demand
make eval-baseline                                        # evaluation Job + gate (promotes baseline)
make gate-check                                           # gate only; edit gate-thresholds first
make candidate-ingest COLLECTION=rahbar_candidate CHUNK_SIZE=300 CHUNK_OVERLAP=50
make candidate-up COLLECTION=rahbar_candidate
make eval-candidate COLLECTION=rahbar_candidate           # exit status 1 = rejected, never released
make release COLLECTION=rahbar_candidate                  # rolling update (only after a pass)
make rollback                                             # kubectl rollout undo
make pull-results                                         # results and traces into the repository
```

## 5. Notes
- Jobs cannot be edited in place; `scripts/run-job.sh` deletes the old one first.
- `ragops-data` is ReadWriteOnce: all pods that mount it must run on one node (true on minikube).
- Traces: `kubectl logs -l app=api -n rahbar --tail=-1 | grep '"event": "trace"'`;
  metrics: `kubectl port-forward svc/api 8000:8000 -n rahbar` then `curl localhost:8000/metrics`.
