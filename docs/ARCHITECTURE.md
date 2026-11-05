# Architecture notes (platform)

## Why Chroma is a StatefulSet and not a Deployment
`k8s/base/chroma-statefulset.yaml` gives the vector store a stable identity (`chroma-0`) and
its own PersistentVolumeClaim through `volumeClaimTemplates`. When the pod is deleted, the
StatefulSet controller re-creates `chroma-0` and re-attaches the same volume, so the index
survives. A Deployment shares one pod template and, with a `ReadWriteOnce` volume, either
cannot schedule a second replica or lets replicas fight over one claim. We run one replica
because Chroma is a single-node database.

## Objects and who talks to whom
Namespace `rahbar`. Ingress sends `/` to the ui and `/api` to the api. The api reads Chroma
and Redis and calls the LLM provider; the indexer Job is the only writer to Chroma. Everything
is denied by default (`k8s/base/networkpolicy.yaml`) and only these paths are opened: ingress
to ui and api, ui to api, api to chroma and redis, indexer to chroma, and DNS plus the LLM
host as egress. The ui pod cannot reach chroma, and only the api reaches redis.

## Configuration
One ConfigMap (`rahbar-config`) feeds both the api and the indexer, so `EMBED_MODEL` has one
source of truth. The LLM key is a Secret created from the command line with `make secret`
(`k8s/base/secret.placeholder.yaml` only documents its shape).

## Overlays
`dev` runs on minikube with locally built images and, while it still has its `patches:`
block, in stub mode (dummy index, stub LLM). `prod` is what `cd.yml` deploys, with images
pinned by commit SHA. `ci` is used by the e2e job on kind.
