#!/usr/bin/env bash
# Run a Job from a manifest, stream its logs, and exit with the Job's result.
#   scripts/run-job.sh k8s/eval-job.yaml eval-baseline
#   COLLECTION=rahbar_candidate scripts/run-job.sh k8s/eval-job-candidate.yaml eval-candidate
# A Job's pod template cannot be edited in place, so an old Job of the same name is deleted first.
# Exit code: 0 = Job succeeded (for the eval Jobs: the gate passed), 1 = failed, 2 = timed out.
set -u
NS="${NS:-rahbar}"
manifest="${1:?usage: run-job.sh <manifest> <job-name> [timeout-seconds]}"
job="${2:?usage: run-job.sh <manifest> <job-name> [timeout-seconds]}"
timeout="${3:-3600}"

kubectl delete job "$job" -n "$NS" --ignore-not-found --wait=true >/dev/null
if grep -q '\${' "$manifest"; then
  python3 "$(dirname "$0")/render.py" "$manifest" | kubectl apply -f - || exit 2
else
  kubectl apply -f "$manifest" || exit 2
fi

# the pod needs a moment before its log can be read
for _ in $(seq 1 90); do
  kubectl logs -f "job/$job" -n "$NS" --all-containers 2>/dev/null && break
  sleep 2
done

deadline=$((SECONDS + timeout))
while [ "$SECONDS" -lt "$deadline" ]; do
  ok="$(kubectl get job "$job" -n "$NS" -o jsonpath='{.status.succeeded}' 2>/dev/null)"
  failed="$(kubectl get job "$job" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null)"
  if [ "${ok:-0}" -ge 1 ]; then echo "JOB $job: SUCCEEDED"; exit 0; fi
  if [ "${failed:-0}" -ge 1 ]; then echo "JOB $job: FAILED"; exit 1; fi
  sleep 3
done
echo "JOB $job: TIMED OUT after ${timeout}s"
exit 2
