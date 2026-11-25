"""Apply the Assignment 03 platform edits to files that other people also change.

Run it once from the repository root, on a branch:   python3 scripts/apply_a3_patches.py
It edits the CURRENT versions of these files in place (so Ermish's changes and the
wait-for-chroma init container are kept) and is idempotent: a second run changes nothing.

  k8s/base/configmap.yaml, api.yaml, indexer-job.yaml, reindex-cronjob.yaml,
  serviceaccounts.yaml, kustomization.yaml, Makefile, .github/workflows/ci.yml
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path.cwd()
changed = 0


def edit(rel: str, fn) -> None:        # type: ignore[no-untyped-def]
    global changed
    path = ROOT / rel
    if not path.is_file():
        print(f"SKIP   {rel}: file not found")
        return
    old = path.read_text(encoding="utf-8")
    new, note = fn(old)
    if new == old:
        print(f"ok     {rel}: {note or 'nothing to do'}")
        return
    path.write_text(new, encoding="utf-8")
    changed += 1
    print(f"EDITED {rel}: {note}")


def configmap(s: str) -> tuple[str, str]:
    if "DATA_DIR" in s:
        return s, "already has DATA_DIR"
    marker = re.search(r'^( *)CORPUS_DIR:.*\n', s, flags=re.M)
    block = ('  # --- RAGOps (Assignment 03): the shared volume ---\n'
             '  DATA_DIR: "/data"              # lake/, quarantine/, archive/, state/, results/\n'
             '  TRACE_DIR: "/data/traces"      # one traces-<pod>.jsonl per api pod\n')
    if not marker:
        return s.rstrip("\n") + "\n" + block, "appended DATA_DIR and TRACE_DIR"
    return s[:marker.end()] + block + s[marker.end():], "added DATA_DIR and TRACE_DIR"


def api(s: str) -> tuple[str, str]:
    if "claimName: ragops-data" in s:
        return s, "already mounts ragops-data"
    s = s.replace("        runAsGroup: 10001\n        seccompProfile",
                  "        runAsGroup: 10001\n        fsGroup: 10001\n        seccompProfile", 1)
    s = s.replace(
        "            - secretRef: {name: rahbar-secret, optional: true}   # LLM_API_KEY, hosted mode only\n",
        "            - secretRef: {name: rahbar-secret, optional: true}   # LLM_API_KEY, hosted mode only\n"
        "          env:\n            - name: POD_NAME           # names this pod's trace file\n"
        "              valueFrom:\n                fieldRef: {fieldPath: metadata.name}\n", 1)
    s = s.replace("            - {name: tmp, mountPath: /tmp}\n      volumes:\n        - name: tmp\n"
                  "          emptyDir: {sizeLimit: 64Mi}",
                  "            - {name: tmp, mountPath: /tmp}\n"
                  "            - {name: data, mountPath: /data}     # trace files (TRACE_DIR)\n"
                  "      volumes:\n        - name: tmp\n          emptyDir: {sizeLimit: 64Mi}\n"
                  "        - name: data\n          persistentVolumeClaim: {claimName: ragops-data}", 1)
    ok = "claimName: ragops-data" in s and "POD_NAME" in s and "fsGroup" in s
    return s, "mounted ragops-data, POD_NAME, fsGroup" if ok else "PARTIAL: check the file by hand"


def job_like(s: str) -> tuple[str, str]:
    if "claimName: ragops-data" in s:
        return s, "already mounts ragops-data"
    indent_sc = re.search(r'^( *)runAsGroup: 10001\n', s, flags=re.M)
    if indent_sc:
        s = s[:indent_sc.end()] + f"{indent_sc.group(1)}fsGroup: 10001\n" + s[indent_sc.end():]
    mount = re.search(r'^( *)- \{name: tmp, mountPath: /tmp\}[^\n]*\n', s, flags=re.M)
    if mount:
        s = s[:mount.end()] + (f"{mount.group(1)}- {{name: data, mountPath: /data}}"
                               "     # data lake, quarantine, archive, state, reports\n") + s[mount.end():]
    vol = re.search(r'^( *)emptyDir: \{sizeLimit: 1Gi\}\n', s, flags=re.M)
    if vol:
        ind = vol.group(1)[:-2]
        s = s[:vol.end()] + f"{ind}- name: data\n{vol.group(1)}persistentVolumeClaim: {{claimName: ragops-data}}\n" \
            + s[vol.end():]
    ok = bool(indent_sc and mount and vol)
    return s, "mounted ragops-data" if ok else "PARTIAL: check the file by hand"


def serviceaccounts(s: str) -> tuple[str, str]:
    if "eval-sa" in s:
        return s, "already has eval-sa"
    return (s.rstrip("\n") + "\n---\napiVersion: v1\nkind: ServiceAccount\n"
            "metadata: {name: eval-sa, namespace: rahbar}\nautomountServiceAccountToken: false\n",
            "added eval-sa")


def kustomization(s: str) -> tuple[str, str]:
    add = [r for r in ("networkpolicy-ragops.yaml", "ragops-data-pvc.yaml", "gate-thresholds.yaml")
           if r not in s]
    if not add:
        return s, "already lists the RAGOps resources"
    anchor = "  - networkpolicy.yaml\n"
    if anchor not in s:
        return s, "PARTIAL: add " + ", ".join(add) + " under resources by hand"
    return s.replace(anchor, anchor + "".join(f"  - {r}\n" for r in add), 1), "added " + ", ".join(add)


MAKE_BLOCK = '''
# ---- Assignment 03 (RAGOps) ------------------------------------------------------------------
COLLECTION ?= rahbar_candidate
EMBED_MODEL ?= nomic-embed-text
CHUNK_SIZE ?= 400
CHUNK_OVERLAP ?= 50
export COLLECTION EMBED_MODEL CHUNK_SIZE CHUNK_OVERLAP

eval-image: ## build the evaluation image and load it into minikube
	docker build -f eval/Dockerfile -t rahbar-eval:dev .
	minikube image load rahbar-eval:dev

load-docs: ## copy a folder into the data lake: make load-docs DIR=corpus/batch_b
	scripts/load-docs.sh $(DIR)

pull-results: ## copy results and traces from the volume into eval/results and traces
	scripts/pull-results.sh

eval-baseline: ## run the suite on the live api and the gate; promotes a passing run to baseline
	scripts/run-job.sh k8s/eval-job.yaml eval-baseline

gate-check: ## run only the gate on the newest result (after editing gate-thresholds)
	scripts/run-job.sh k8s/gate-job.yaml gate-check 600

candidate-ingest: ## build the candidate collection: make candidate-ingest COLLECTION=... CHUNK_SIZE=...
	scripts/run-job.sh k8s/candidate-ingest-job.yaml candidate-ingest

candidate-up: ## start the candidate api on its own collection
	python3 scripts/render.py k8s/candidate.yaml | kubectl apply -f -
	kubectl rollout status deployment/api-candidate -n $(NS) --timeout=240s

eval-candidate: ## run the suite on the candidate api and the gate (non-zero exit = rejected)
	scripts/run-job.sh k8s/eval-job-candidate.yaml eval-candidate

release: ## roll the live api onto a collection (rolling update, maxUnavailable 0)
	kubectl set env deployment/api CHROMA_COLLECTION=$(COLLECTION) EMBED_MODEL=$(EMBED_MODEL) -n $(NS)
	kubectl annotate deployment/api kubernetes.io/change-cause="release collection $(COLLECTION), embed model $(EMBED_MODEL)" --overwrite -n $(NS)
	kubectl rollout status deployment/api -n $(NS) --timeout=300s

rollback: ## undo the last release
	kubectl rollout undo deployment/api -n $(NS)
	kubectl rollout status deployment/api -n $(NS) --timeout=300s
'''


def makefile(s: str) -> tuple[str, str]:
    if "eval-baseline:" in s:
        return s, "already has the RAGOps targets"
    return s.rstrip("\n") + "\n" + MAKE_BLOCK, "added the RAGOps targets"


def ci(s: str) -> tuple[str, str]:
    notes = []
    if "pytest scripts/tests" not in s:
        m = re.search(r'^( *)- run: cd indexer && pytest -q\n', s, flags=re.M)
        if m:
            s = s[:m.end()] + f"{m.group(1)}- run: pytest scripts/tests -q\n" + s[m.end():]
            notes.append("script tests")
        else:
            notes.append("MISSING indexer pytest line: add 'pytest scripts/tests -q' by hand")
    if "corpus scripts" not in s and "ruff check api indexer eval corpus" in s:
        s = s.replace("ruff check api indexer eval corpus", "ruff check api indexer eval corpus scripts", 1)
        notes.append("ruff on scripts")
    if "scripts/render.py" not in s:
        m = re.search(r'^( *)kubectl kustomize k8s/overlays/\$overlay \| kubeconform -strict -summary\n( *)done\n',
                      s, flags=re.M)
        if m:
            i = m.group(2)
            block = (f'{i}echo "== Assignment 03 manifests (templated ones are rendered first)"\n'
                     f'{i}for f in loader-pod eval-job eval-job-candidate gate-job candidate-ingest-job candidate; do\n'
                     f'{i}  COLLECTION=rahbar_candidate CHUNK_SIZE=300 CHUNK_OVERLAP=50 EMBED_MODEL=nomic-embed-text \\\n'
                     f'{i}    python3 scripts/render.py k8s/$f.yaml | kubeconform -strict -summary\n'
                     f'{i}done\n')
            s = s[:m.end()] + block + s[m.end():]
            notes.append("kubeconform on the new manifests")
        else:
            notes.append("MISSING kubeconform loop: add the new manifests by hand")
    return s, ", ".join(notes) if notes else "already patched"


if __name__ == "__main__":
    if not (ROOT / "k8s" / "base").is_dir():
        sys.exit("run this from the repository root")
    edit("k8s/base/configmap.yaml", configmap)
    edit("k8s/base/api.yaml", api)
    edit("k8s/base/indexer-job.yaml", job_like)
    edit("k8s/base/reindex-cronjob.yaml", job_like)
    edit("k8s/base/serviceaccounts.yaml", serviceaccounts)
    edit("k8s/base/kustomization.yaml", kustomization)
    edit("Makefile", makefile)
    edit(".github/workflows/ci.yml", ci)
    print(f"\n{changed} file(s) edited. Review with: git diff --stat")
