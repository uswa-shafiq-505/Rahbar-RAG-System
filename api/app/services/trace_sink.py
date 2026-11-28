"""Where trace records go (Assignment 03, parts D1 and F4).

The answer pipeline (Member 2) builds ONE dict per query and calls write_trace(record).
This module only transports it:
  * one JSON line on stdout, readable with `kubectl logs` (event = "trace");
  * the same line appended to $TRACE_DIR/traces-<pod>.jsonl when TRACE_DIR is set (the
    ragops-data volume). One file per pod, so three replicas never write to the same file.

The record fields are listed in docs/RAGOPS-CONTRACTS.md. Nothing here can raise: a full
disk or a read-only volume must never turn into an HTTP 500 for the user.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from ..config import Settings, get_settings

log = logging.getLogger("rahbar.trace")
_lock = threading.Lock()
_warned = False


def new_trace_id() -> str:
    return uuid.uuid4().hex


def base_fields(settings: Settings) -> dict[str, Any]:
    """The fields every trace needs and the platform already knows."""
    return {
        "collection": settings.collection,
        "embed_model": settings.embed_model,
        "pod": settings.pod_name,
    }


def write_trace(record: dict[str, Any], settings: Settings | None = None) -> str:
    """Write one trace line. Returns the trace id (generated when the record has none)."""
    global _warned
    settings = settings or get_settings()
    rec = dict(record)
    rec.setdefault("trace_id", new_trace_id())
    rec.setdefault("ts", datetime.now(UTC).isoformat(timespec="milliseconds"))
    rec.setdefault("pod", settings.pod_name)
    rec["event"] = "trace"
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with _lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        if settings.trace_dir:
            try:
                os.makedirs(settings.trace_dir, exist_ok=True)
                path = os.path.join(settings.trace_dir, f"traces-{settings.pod_name}.jsonl")
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                if not _warned:
                    _warned = True
                    log.warning("cannot write trace file", extra={"error": str(exc)})
    return str(rec["trace_id"])
