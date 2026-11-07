"""Liveness and readiness (section 4.3). They answer different questions."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    """Can this pod take traffic? 200 only if Chroma is reachable, the collection is
    non-empty and was built with OUR embedding model, and Redis is reachable. Otherwise
    503 naming the dependency that failed. Not ready = withheld from the Service, not
    restarted (that is liveness's job, and liveness must not look at dependencies)."""
    ready, checks = request.app.state.index_state.readiness()
    failed = {name: why for name, why in checks.items() if why != "ok"
              and not why.startswith("skipped")}
    if len(failed) > 0 or not ready:
        response.status_code = 503
    return {"status": "ready" if ready else "not ready", "checks": checks, "failed": failed}
