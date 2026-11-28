## RAGOps (Assignment 03): the command for each part

Prerequisite: the platform is running (`k8s/README.md`, steps 1 and 2) and the data lake holds Batch A.

| Part | Command | Result |
|---|---|---|
| A: build Batch B | `python scripts/make_batch_b.py --out corpus/batch_b` | 11 documents + `MANIFEST-batch-b.json` |
| A: verify and ingest | `make load-docs DIR=corpus/batch_b && kubectl create job --from=cronjob/reindex ingest-batch-b -n rahbar && kubectl logs -f job/ingest-batch-b -n rahbar` | verification report in `eval/results/` after `make pull-results` |
| A: unchanged corpus embeds 0 chunks | run the same ingestion Job again | log shows 0 chunks embedded |
| B: suite | `make eval-baseline` | `eval/results/<timestamp>.json` with all three levels |
| C: release gate | `make eval-candidate COLLECTION=...` | exit status 0 = pass, 1 = fail |
| C: gate only | `make gate-check` | PASS or FAIL table |
| D: trace by id | `kubectl logs -l app=api -n rahbar --tail=-1 \| python scripts/trace.py <trace_id> --stdin` | context and prompt |
| D: metrics summary | `python scripts/trace_report.py --dir traces --since 24h` | queries, similarity, abstention, cache, p95, injections |
| E: coverage | `python -m ragops.coverage` (Member 2) | share above the threshold; non-zero below it |
| F: deploy and release | `kubectl get all,pvc,configmap,cronjob,job -n rahbar`, `make release`, `make rollback` | see `k8s/README.md` |
| All: results into git | `make pull-results` | `eval/results/`, `traces/` |
