# RAHBAR: Member 1 (Platform) - complete step-by-step guide

You own: Docker, Kubernetes, security, CI/CD, probes + embedding pin, semantic cache,
metrics, and the platform docs/evidence. Member 2 owns: corpus, indexer stages, retriever,
LLM provider, answer logic, evaluation.

**Everything in this guide is already written for you** in `rahbar-member1.zip` (the starter kit
with the platform layer filled in). Your job is to (1) set up the tools, (2) put the code in a
GitHub repo, (3) run it, (4) produce the evidence the marker asks for, and (5) understand it well
enough to explain it. Do the parts in order. Budget: about one long day.

> **Honesty note.** I validated what can be validated without a cluster: `pytest` (28 tests),
> `ruff`, `mypy`, `kustomize build` + `kubeconform -strict` on all three overlays, YAML parse of
> the workflows, and the starter's `check_submission.py`. I could not run Docker, minikube or
> GitHub Actions here, so expect a small fix or two on first run. Part 9 lists the most likely
> ones and what to do. The "what broke and how I fixed it" is also your engineering-notes answer 7.

---

## Part 0 - What is in the zip (the map)

| You will touch | What it is |
|---|---|
| `api/Dockerfile`, `indexer/Dockerfile`, `ui/Dockerfile` (+ `.dockerignore`) | The three images. Non-root, pinned base images. |
| `k8s/base/*.yaml` | Namespace, ServiceAccounts, ConfigMap, Chroma StatefulSet, Redis, api, ui, Services, Ingress, HPA, NetworkPolicies, indexer Job, reindex CronJob, Secret placeholder |
| `k8s/overlays/{dev,prod,ci}` | dev = local minikube (stub mode at first), prod = what CD deploys (SHA images), ci = e2e test on kind |
| `api/app/services/index_state.py` | `/readyz` logic + the **embedding pin** check |
| `api/app/services/cache.py`, `cache_layer.py`, `embedding.py` | The Redis semantic cache |
| `api/app/services/instrument.py`, `routes/metrics.py` | The extra Prometheus metrics |
| `.github/workflows/ci.yml`, `cd.yml` | The two pipelines |
| `docs/adr/0001-indexer-as-job.md`, `0003-cache-store-redis.md`, `docs/ARCHITECTURE.md`, `docs/ENGINEERING-NOTES.md` (Q1-4, 6) | Your documents (drafts you must read and put in your own words) |
| `scripts/cache_threshold.py` | Measures the cache threshold for ADR 0003 |

The things Member 2 fills in: `indexer/pipeline/*`, `api/app/retrieval/chroma.py`,
`api/app/providers/llm/*`, `api/app/services/answer.py`, `eval/*`, `corpus/*`.
You never need to edit those files. (Contracts between you are in Part 6.)

---

## Part 1 - Install the tools (about 45 minutes, once)

You need: **Git, Python 3.12, make, Docker, kubectl, minikube**. On Windows use **WSL 2**.

### 1.1 Windows only: WSL 2 + Ubuntu
1. Open PowerShell as Administrator: `wsl --install -d Ubuntu`, then reboot, then open "Ubuntu"
   and create a Linux username/password.
2. Install **Docker Desktop**, then Settings -> Resources -> WSL integration -> turn on Ubuntu.
3. From now on, run every command in the **Ubuntu terminal**, and keep the project under your
   Linux home (`~/rahbar`), not under `/mnt/c/...` (that is very slow).

### 1.2 Inside Ubuntu (or any Linux/macOS terminal)
```bash
sudo apt update && sudo apt install -y git make python3.12 python3.12-venv curl unzip   # Ubuntu
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/ && rm kubectl
# minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
# check everything
git --version; python3 --version; make --version | head -1
docker run --rm hello-world | head -3; kubectl version --client; minikube version
```
If `python3.12` is not available, any Python 3.11+ works for the local tests.

### 1.3 Ollama on your laptop (needed from Part 6; install now)
Install Ollama from ollama.com (Windows/macOS app, or the Linux script). Then:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

## Part 2 - Create the GitHub repo and protect `main` (about 30 minutes)

### 2.1 Put the code in a repo
```bash
cd ~ && unzip /path/to/rahbar-member1.zip && cd rahbar
git init -b main
git add -A
git status            # read the list. There must be NO .env, NO *secret*.yaml, NO corpus/raw.
git commit -m "Starter kit + platform layer"
```
On GitHub: **New repository** (empty, no README). Then:
```bash
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main        # this is the ONLY direct push to main you are allowed
git checkout -b dev && git push -u origin dev
```
Add your partner as collaborator (Settings -> Collaborators) and, if the repo is private, the
two instructors.

> **Rule:** from now on never `git push origin main`. Work on `dev` (or a branch from it) and
> merge with a pull request. Direct pushes to `main` cost -4.

### 2.2 Branch protection (screenshot = evidence)
GitHub -> Settings -> Branches -> **Add branch ruleset / protection rule** for `main`:
- Require a pull request before merging, **1 approval**
- Require status checks to pass: after your first CI run, select `lint`, `test`, `manifests`,
  `build`, `scan`, `e2e-eval`
- Block force pushes
Take a screenshot -> `docs/evidence/branch-protection.png`.

### 2.3 Docker Hub token and GitHub Secrets
1. hub.docker.com -> Account settings -> Personal access tokens -> **Generate** (Read & Write).
2. GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret:
   - `DOCKERHUB_USERNAME` = your Docker Hub username
   - `DOCKERHUB_TOKEN` = the token (never your password)
3. In `README.md` replace `YOUR_GITHUB_USER/YOUR_REPO` with your real ones (badges).
4. In `k8s/overlays/prod/kustomization.yaml` replace `YOUR_DOCKERHUB_USER` with your username
   (cd.yml overrides it, but keep the file honest).

---

## Part 3 - Run the tests locally (10 minutes)
```bash
make install          # creates .venv, installs everything
make test             # api + indexer tests: stubs only, no cluster
make lint             # ruff + mypy
```
Expected: `28 passed` for the api. If `make install` fails on `python3`, run
`make install PY=python3.12`.

Try the API without a cluster: `make dev`, open http://localhost:8000, ask anything.

---

## Part 4 - Start the cluster and run everything in STUB mode (about 1.5 hours)

Stub mode proves the whole platform without any RAG code or model: the api uses the stub
retriever/LLM and the indexer Job writes a 3-chunk dummy index.

### 4.1 Cluster
```bash
make cluster
```
This runs `minikube start --driver=docker --cni=calico --memory=6144 --cpus=4` and enables
`metrics-server` and `ingress`. **Calico matters**: the default network plugin accepts
NetworkPolicy manifests and silently ignores them.
```bash
kubectl get nodes                                  # Ready
kubectl get pods -n kube-system | grep -E "calico|metrics"
kubectl get pods -n ingress-nginx                  # controller Running
```
If your PC has < 8 GB free RAM use `make cluster` after editing `--memory=4096` in the Makefile.

### 4.2 Build and load the images
```bash
make images
```
`make images` (1) stages documents into the indexer build context (the real `corpus/raw` if it
exists, otherwise the fixtures), (2) builds the three images, (3) loads them into minikube.
minikube cannot see images from your own Docker unless they are loaded: that is why
`imagePullPolicy: IfNotPresent` is set everywhere.
```bash
minikube image ls | grep rahbar     # three images, tag :dev
```

### 4.3 Apply the manifests
```bash
make apply            # = kubectl delete job indexer (if any) + kubectl apply -k k8s/overlays/dev
kubectl get all,pvc,networkpolicy,ingress -n rahbar
```
Watch it come up (Ctrl+C to stop):
```bash
kubectl get pods -n rahbar -w
```
Expected order: `chroma-0` and `redis-...` Running -> the `indexer-...` pod Completed -> the
three `api-...` pods `1/1` (they stay `0/1` until the dummy index exists: that is `/readyz` working).

Check the Job ran: `kubectl logs job/indexer -n rahbar` (you should see `indexer finished`, `chunks: 3`).

**The diagnosis order, memorise it** (also viva material): `kubectl get pods -n rahbar` ->
`kubectl describe pod <pod> -n rahbar` (read **Events** at the bottom) -> `kubectl logs <pod>
-n rahbar` (add `--previous` for a crashed container). Status, then events, then app output.

### 4.4 Reach it through the Ingress
```bash
minikube ip                                     # e.g. 192.168.49.2
echo "$(minikube ip)  rahbar.local" | sudo tee -a /etc/hosts
make ask Q="hello stub"
```
Windows/macOS with the docker driver: the minikube IP is not reachable. In a **second terminal**
run `minikube tunnel` (keep it open), put `127.0.0.1  rahbar.local` in the hosts file, and use
`make ask INGRESS_IP=127.0.0.1`. Then open http://rahbar.local in a browser for the chat page.

You should get JSON with `"provider": "stub"` and three retrieved passages.

---

## Part 5 - Prove every requirement and collect evidence (about 2 hours)

Create the evidence folder files as you go. A command first, the expected result second.

### 5.1 Delete a pod: nothing breaks
```bash
kubectl delete pod -n rahbar -l app=api --wait=false
for i in 1 2 3 4 5 6; do make ask Q="still there?" | grep answered_from_context; sleep 2; done
kubectl get pods -n rahbar -l app=api
```
Requests keep working (the other two replicas serve; replacements come back). Then the
database: `kubectl delete pod chroma-0 -n rahbar`. It returns and re-attaches the same PVC; api
pods go `0/1` for a few seconds and come back **without re-indexing** (`kubectl get jobs -n rahbar`
shows the Job count unchanged). Save the outputs in `docs/evidence/pod-delete.txt`.

### 5.2 Endpoints: the Service found the pods by label
```bash
kubectl get endpoints api -n rahbar | tee docs/evidence/endpoints.txt      # 3 IPs
kubectl get pods -n rahbar -l app=api -o wide                              # same 3 IPs
```

### 5.3 HPA shows a real number (not `<unknown>`)
```bash
kubectl get hpa -n rahbar | tee docs/evidence/hpa.txt     # TARGETS like 2%/70%
```
If it says `<unknown>/70%`: wait a minute; check `kubectl top pods -n rahbar` works (metrics-server).

### 5.4 Resources: justify the numbers with `kubectl top`
```bash
kubectl top pods -n rahbar --containers | tee docs/evidence/kubectl-top.txt
```
Look at the `MEMORY` of each container idle. The request should be near typical use; the limit
about 2x peak. Edit `resources:` in `k8s/base/*.yaml` if a number is far off, re-`make apply`,
and re-capture. **For the indexer** you need the number *during* the real run (Part 6):
```bash
kubectl create job --from=cronjob/reindex mem-test -n rahbar
watch -n 2 "kubectl top pods -n rahbar --containers | grep indexer"
```
Write the peak into `docs/ENGINEERING-NOTES.md` answer 4 (replace the **MEASURE** markers).
Exit code 137 = OOMKilled: `kubectl describe pod <indexer pod> | grep -A3 "Last State"`.

### 5.5 NetworkPolicy: the ui cannot reach chroma
Pod Security is `restricted`, so run these with `kubectl exec` into the existing pods:
```bash
# ui -> chroma : must FAIL (timeout)
kubectl exec -n rahbar deploy/ui -- wget -qO- -T 3 http://chroma:8000/api/v2/heartbeat
# ui -> api : must WORK
kubectl exec -n rahbar deploy/ui -- wget -qO- -T 3 http://api:8000/healthz
# api -> chroma : must WORK
kubectl exec -n rahbar deploy/api -- python -c "import urllib.request as u; print(u.urlopen('http://chroma:8000/api/v2/heartbeat', timeout=3).read())"
# api -> redis : works (python socket); ui -> redis : must fail
kubectl exec -n rahbar deploy/ui -- wget -qO- -T 3 http://redis:6379 ; echo "exit=$?"
```
Save the transcript as `docs/evidence/networkpolicy.txt`. If `ui -> chroma` WORKS, the cluster
was started without Calico: `minikube delete`, then `make cluster` again.

### 5.6 Rollout, visible change, undo
```bash
# make a visible change: edit api/app/main.py -> FastAPI(title="RAHBAR API v2"), then:
docker build -t rahbar-api:v2 api && minikube image load rahbar-api:v2
kubectl set image deployment/api api=rahbar-api:v2 -n rahbar
kubectl rollout status deployment/api -n rahbar
curl -s -H "Host: rahbar.local" http://$(minikube ip)/openapi.json | grep -o '"title":"[^"]*"'   # v2
kubectl rollout undo deployment/api -n rahbar
kubectl rollout status deployment/api -n rahbar
curl -s -H "Host: rahbar.local" http://$(minikube ip)/openapi.json | grep -o '"title":"[^"]*"'   # old title
```
Paste everything into `docs/evidence/rollout-undo.txt`, then **revert the title edit** in git
(`git checkout api/app/main.py`). During the rollout, `maxUnavailable: 0` means the pod count
never drops below 3 ready.

### 5.7 The section 4.4 demonstration: wrong EMBED_MODEL -> not ready
Needs the stub-mode state (`EMBED_MODEL=dummy-768`, index built by the dummy Job):
```bash
kubectl patch configmap rahbar-config -n rahbar --type merge -p '{"data":{"EMBED_MODEL":"nomic-embed-text"}}'
kubectl rollout restart deployment/api -n rahbar
sleep 20
kubectl get pods -n rahbar -l app=api            # a NEW pod sits at 0/1 READY
kubectl logs -n rahbar -l app=api --tail=200 | grep -i mismatch
kubectl exec -n rahbar deploy/api -- python -c "import urllib.request as u; print(u.urlopen('http://localhost:8000/readyz'))" 2>&1 | tail -3
```
Save as `docs/evidence/pin-mismatch.txt`. The old pods stay ready (`maxUnavailable: 0`), so users
see nothing wrong, but the rollout cannot complete: `kubectl rollout status deployment/api`
waits. **Undo it:** patch the ConfigMap back to `dummy-768` and `kubectl rollout restart deployment/api -n rahbar`.

### 5.8 Metrics and logs
```bash
kubectl exec -n rahbar deploy/api -- python -c "import urllib.request as u; print(u.urlopen('http://localhost:8000/metrics').read().decode())" | grep -E "^rahbar_" | head -30
kubectl logs -n rahbar deploy/api --tail=5       # JSON lines with request_id
```
You should see: `rahbar_requests_total`, `rahbar_request_seconds`, `rahbar_retrieval_seconds`,
`rahbar_generation_seconds`, `rahbar_cache_lookups_total`, `rahbar_cache_hit_ratio`,
`rahbar_abstentions_total`, `rahbar_fallbacks_total`, `rahbar_tokens_total`.

### 5.9 Re-index with no redeploy
```bash
kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar
kubectl logs -f job/manual-reindex-1 -n rahbar
```
During it, keep asking questions: the api stays up. Screenshot/paste into `docs/evidence/reindex.txt`.

When you finish Part 5, run `make evidence` once more to refresh `cluster-state.txt`.

---

## Part 6 - Go real with Member 2's code (when their pieces land)

Agree these **contracts** with Member 2 before they code (send them this table):

| Contract | Value |
|---|---|
| Chroma collection name | `rahbar` (env `CHROMA_COLLECTION`) |
| Collection metadata written by `indexer/pipeline/store.py` | must include `embed_model` (= the `EMBED_MODEL` used) and `dimension`; optionally `index_version` (any new string per build). The dummy indexer already does `embed_model` + `dimension`. |
| Chunk metadata | `source, section, page, doc_version, superseded` |
| Cache | Member 2 does **not** touch it. `api/app/routes/ask.py` calls `answer_with_cache()` which calls Member 2's `answer_question()`. |
| Fallback | In the validation-failure fallback, return `answered_from_context=False` (so it is never cached) and call `metrics_hooks.fallback()` from `app/services/metrics_hooks.py`. |
| Embedding | `app/services/embedding.py::embed_query(text, settings)` exists (Ollama `/api/embed`, memoised). The retriever may reuse it so a question is embedded once. |
| Retriever, provider | Keep their names/interfaces (`retrieve`, `generate`): `main.py` wraps them for the latency/token metrics. |
| Collection swap on re-index | Member 2 decides (delete + create is simplest). The cache notices via the new collection id/count. |

Then switch the cluster to real mode:
1. Open `k8s/overlays/dev/kustomization.yaml` and **delete the whole `patches:` block** (the
   three patches that force stub mode).
2. Start Ollama on the host: `OLLAMA_HOST=0.0.0.0 ollama serve`.
3. `make images && make apply`. The Job now runs the real pipeline (`kubectl logs -f job/indexer -n rahbar`).
4. If the api cannot reach Ollama: `minikube ssh "grep host.minikube.internal /etc/hosts"`, and
   check `api-egress` in `k8s/base/networkpolicy.yaml` allows that address on port 11434 (it allows
   everything outside `10.0.0.0/8`; tighten it to the single IP and mention it in the README).
5. `make ask Q="What is the attendance requirement?"` -> cited answer; ask something the corpus
   lacks -> a refusal (`answered_from_context: false`).

### Cache: measure the threshold and test it
```bash
python scripts/cache_threshold.py          # needs Ollama; edit the question pairs for YOUR corpus
```
Copy the printed table to `docs/PIPELINE.md` (Member 2's file: put it under a heading
"Semantic cache threshold") and write the lowest zero-false-hit value in
`k8s/base/configmap.yaml` (`CACHE_THRESHOLD`) and in ADR 0003. Then test:
```bash
make ask Q="What is the minimum attendance requirement?"
make ask Q="How much attendance do I need?"      # similar: response shows "cached": true
curl -si -H "Host: rahbar.local" -H 'Content-Type: application/json' http://$(minikube ip)/api/ask \
  -d '{"question":"What is the minimum attendance requirement?"}' | grep -i x-cache   # X-Cache: HIT
kubectl create job --from=cronjob/reindex ri-2 -n rahbar     # re-index, wait for it to finish
make ask Q="What is the minimum attendance requirement?"     # "cached": false again (cache invalidated)
kubectl logs -n rahbar deploy/api | grep "cache invalidated"
```

### Resource numbers for the real run
Capture `kubectl top pods --containers` **while the real indexer runs and while answering**, update
`resources:` in `k8s/base`, and fill ENGINEERING-NOTES answer 4.

---

## Part 7 - CI/CD (about 1.5 hours, plus waiting time)

### 7.1 First CI run
```bash
git checkout dev
git add -A && git commit -m "Platform layer" && git push origin dev
```
GitHub -> Actions -> `ci`. Jobs: `lint`, `test`, `manifests`, `build`, `scan`, `e2e-eval`.
Fix red jobs in order; read the failing step's log. The e2e job needs two files from Member 2:
- `eval/fixtures/baseline.fixture.json` (create it once by running the eval on the fixtures with
  `--update-baseline`, see below)
- a working real pipeline (indexer + retriever). Until then expect `e2e-eval` to fail; that is
  fine, disable it from the required checks temporarily but **re-enable it before submitting**.

Create the fixture baseline (needs the real stack running, locally):
```bash
python eval/run_eval.py --api http://rahbar.local --golden eval/fixtures/golden.fixture.yaml \
  --baseline eval/fixtures/baseline.fixture.json --out /tmp/fixture-results.json --update-baseline
git add eval/fixtures/baseline.fixture.json
```
(If you run it through `minikube tunnel`, use `--api http://127.0.0.1 --host rahbar.local`.)

### 7.2 Trivy fails on HIGH/CRITICAL
Read the table in the `scan` log. If a finding has a fix, bump the base image tag or the pinned
package in `requirements.txt` / Dockerfile. Findings without a fix are ignored on purpose.

### 7.3 Open a pull request `dev` -> `main`
All six jobs must be green and a teammate must approve (Part 2.2). Merging triggers `cd`:
`test` -> `build-push` (3 images tagged with the commit SHA) -> `deploy` (kind + Ingress +
rollout status + smoke test). Copy the link of the green `cd` run and your Docker Hub page
(showing the SHA tags) for the submission.

### 7.4 The "gate working" screenshots
1. Branch `break-gate` from `dev`. Make the system worse in a way recall notices (for example in
   the ConfigMap set `TOP_K: "1"`, or a chunk size of 20). Push, open a PR to `main`.
2. `e2e-eval` goes red; the **Merge** button is greyed out -> screenshot
   `docs/evidence/blocked-merge.png`.
3. Revert the change on that branch, push, wait for green -> `docs/evidence/green-merge.png`.
(The gate rules are Member 2's `run_eval.py`; the workflow just runs it.)

---

## Part 8 - Finish the documents, run the checker, submit

Fill in / rewrite **in your own words** (the marker reads these first; generic text scores 0):
- `docs/adr/0001-indexer-as-job.md` - read it, fix anything that is not what you did.
- `docs/adr/0003-cache-store-redis.md` - add your real thresholds and the false hit.
- `docs/ARCHITECTURE.md` - the StatefulSet reason is required.
- `docs/ENGINEERING-NOTES.md` - answers 1, 2, 3, 4, 6 are drafts with real file:line references
  (re-check the line numbers after any edit: `grep -n "maxUnavailable" k8s/base/api.yaml`).
  Replace the **MEASURE** markers. Write your own **question 7** about a real failure.
- `docs/AI-USAGE.md` - already drafted: add what *you* changed and why.
- `docs/FAILURE-ANALYSIS.md` - your two stages to break (agreed split): **C** (change
  `EMBED_MODEL` on the api only: this is exactly 5.7) and **E** (remove the `superseded` filter,
  ask a version-sensitive question). For each: symptom, diagnosis, stage, remedy, and which metric/
  log line would have shown it.

Before you submit:
```bash
python scripts/check_submission.py      # = make check; fix every FAIL that is in your area
git log --all --oneline -- '*secret*' '.env'      # must print nothing
```
Submission checklist (platform items):
- [ ] `docs/evidence/`: branch-protection, blocked-merge, green-merge, kubectl-top, endpoints, hpa, pin-mismatch, rollout-undo
- [ ] Link to a green `cd.yml` run, Docker Hub link showing SHA tags
- [ ] README apply order tested from a **clean clone** (do it: clone to a new folder, follow the README only)
- [ ] No secret anywhere in Git history; real key only via `make secret`
- [ ] All pods in namespace `rahbar`; no `:latest` anywhere (`grep -rn ":latest" k8s .github`)

---

## Part 9 - Troubleshooting (symptom -> cause -> fix)

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImagePullBackOff` / `ErrImagePull` | image not loaded into minikube | `make images`; `minikube image ls \| grep rahbar` |
| Pod rejected: `violates PodSecurity "restricted"` | a pod lacks a security setting | `kubectl describe` / `kubectl get events -n rahbar`; the manifests comply, so check you did not edit `securityContext`. Quick escape only if stuck: remove the `pod-security...enforce` label in `k8s/base/namespace.yaml` |
| `chroma-0` CrashLoopBackOff with `Permission denied` | the Chroma image wants to write somewhere other than `/data` as UID 1000 | `kubectl logs chroma-0 -n rahbar --previous`. Try removing `runAsUser`/`runAsGroup` in `k8s/base/chroma-statefulset.yaml` (keep `runAsNonRoot` off for this one pod only if you must, and say so in the ADR) |
| `chroma-0` Pending | PVC cannot bind | `kubectl get pvc -n rahbar`; `kubectl describe pvc`; `minikube addons enable storage-provisioner` |
| Every service name stops resolving | DNS egress blocked | `allow-dns-egress` in `networkpolicy.yaml` must exist; Calico must be running |
| api `Running` but `0/1` | readiness says no | `kubectl logs deploy/api -n rahbar \| grep readiness`; or `curl` `/readyz` via `kubectl exec`: the JSON names the failing check |
| api `0/1`, log says `mismatch` | ConfigMap `EMBED_MODEL` differs from the index | re-run the indexer or fix the ConfigMap + `kubectl rollout restart deployment/api -n rahbar` |
| api `0/1`, `collection ... does not exist` / `empty` | the indexer Job has not finished/failed | `kubectl get job,pods -n rahbar`; `kubectl logs job/indexer -n rahbar` |
| Indexer pod `CrashLoopBackOff`, exit code 137 | OOMKilled | raise `limits.memory` in `indexer-job.yaml` **and** `reindex-cronjob.yaml` (they must match), guided by `kubectl top` |
| `make apply` error: Job field is immutable | re-applying a changed Job | `make apply` already deletes the Job first; or `kubectl delete job indexer -n rahbar` |
| HPA `<unknown>/70%` | metrics-server off or just started | `minikube addons enable metrics-server`; wait ~1 min; ensure `requests.cpu` is set (it is) |
| `ui` pod `CrashLoopBackOff` | nginx cannot write | the `/tmp` emptyDir must be mounted (it is); check `kubectl logs deploy/ui -n rahbar --previous` |
| Browser shows nginx page but chat gets 404 on `/api/ask` | Ingress path / host | `kubectl describe ingress rahbar -n rahbar`; the hosts-file entry must say `rahbar.local` |
| api cannot reach Ollama | Ollama on 127.0.0.1 only, or NetworkPolicy | `OLLAMA_HOST=0.0.0.0 ollama serve`; check `api-egress` and `indexer-egress` |
| `ci` job `e2e-eval`: indexer Job fails | embeddings unreachable from kind | the log shows the connection error; the workflow patches `OLLAMA_URL` to the docker gateway: print it with `docker network inspect kind` |
| `kustomize edit add patch` unknown flag in CI | old kustomize on the runner | the workflows install kustomize v5.4.3 first; keep that step before it |
| Trivy fails on a base image | known CVE with a fix | bump the base tag (`python:3.12-slim`, `nginxinc/nginx-unprivileged:1.27-alpine`) |
| GitHub says secret `DOCKERHUB_TOKEN` not found | secret not created, or PR from a fork | repo Settings -> Secrets; CD runs only on `main` |

When you hit a real problem that cost you more than an hour, **write it down while it is fresh**:
that is your ENGINEERING-NOTES question 7 (what you believed first, and the exact command or log
line that told you the truth).

---

## Part 10 - Understand it (viva cheat sheet)

Be able to explain each of these in two sentences, pointing at the file:
1. Why a Job and not startup code (ADR 0001): 3 writers race, restarts re-index, readiness lies, no re-index without redeploy.
2. Three probes (`k8s/base/api.yaml`): startup = finished booting; liveness = still alive and touches no dependency; readiness = may take traffic (Chroma, non-empty collection, pin, Redis). Swapped = restart storm.
3. Why `maxUnavailable: 0` + `preStop sleep`: old pod keeps serving until the new one is READY and is removed from the Service before SIGTERM.
4. StatefulSet vs Deployment for Chroma: stable identity + its own PVC through `volumeClaimTemplates`.
5. The pin: indexer writes `embed_model` into collection metadata; `index_state.py::check_chroma` compares and answers 503.
6. Cache: threshold from the ConfigMap, never caches abstentions/follow-ups/bad citations, invalidates by index fingerprint, Redis only reachable from the api.
7. NetworkPolicy: default-deny, then DNS, ingress->ui/api, ui->api, api->chroma/redis/LLM, indexer->chroma/LLM. Calico or it is ignored.
8. HPA on CPU measures the wrong thing for an I/O-bound generation service; scale on in-flight requests or latency.
9. Security: per-workload ServiceAccounts with no token, non-root, read-only filesystem, capabilities dropped, Secret from the command line, base64 is not encryption.
10. CI/CD: `needs:` everywhere that publishes, images tagged by `${{ github.sha }}`, least-privilege `permissions:`, secrets from GitHub Secrets.
