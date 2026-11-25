#!/usr/bin/env bash
# Copy results and traces from the volume into the repository, so they can be committed:
#   eval/results/   result files, gate decisions, verification reports, baseline
#   traces/         traces-<pod>.jsonl
set -euo pipefail
NS="${NS:-rahbar}"
kubectl apply -f k8s/loader-pod.yaml >/dev/null
kubectl wait --for=condition=ready pod/data-loader -n "$NS" --timeout=180s
mkdir -p eval/results traces
kubectl exec -n "$NS" data-loader -- tar cf - -C /data/results . | tar xf - -C eval/results
kubectl exec -n "$NS" data-loader -- tar cf - -C /data/traces . | tar xf - -C traces
echo "--- eval/results:"; ls -la eval/results
echo "--- traces:";       ls -la traces
