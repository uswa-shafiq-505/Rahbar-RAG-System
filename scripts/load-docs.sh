#!/usr/bin/env bash
# Copy a folder of documents into the data lake, ready for the ingestion Job.
#   scripts/load-docs.sh corpus/batch_b
#   kubectl create job --from=cronjob/reindex ingest-batch-b -n rahbar     # then run ingestion
set -euo pipefail
NS="${NS:-rahbar}"
dir="${1:?usage: load-docs.sh <folder>}"
[ -d "$dir" ] || { echo "load-docs: $dir is not a folder" >&2; exit 2; }
kubectl apply -f k8s/loader-pod.yaml
kubectl wait --for=condition=ready pod/data-loader -n "$NS" --timeout=180s
tar cf - -C "$dir" . | kubectl exec -i -n "$NS" data-loader -- tar xf - -C /data/lake/incoming
echo "--- /data/lake/incoming now holds:"
kubectl exec -n "$NS" data-loader -- ls -la /data/lake/incoming
