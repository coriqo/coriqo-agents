"""Local fallback buffer for decision traces that couldn't reach Coriqo.

track_decision() (decorator.py) never lets a Coriqo outage raise into the
caller's own code path — a customer's agent must keep running even when
Coriqo is down. When client.record_trace() raises anything other than
AgentSuspendedError (that one is a deliberate governance signal and is
allowed to propagate), the failure is logged and the trace payload is
appended here instead of being dropped silently.

This is a local safety net, not a durability guarantee: it writes to a
single JSONL file on the machine running the agent, with no retry, no
rotation, and no upload path. Rows are the same dict passed to
record_trace(). A caller who wants those rows replayed into Coriqo once it
recovers reads the file and calls record_trace()/record_traces() itself
(or an operator inspects it manually) — this module only guarantees the
data isn't lost on the customer's disk between now and then.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("coriqo_agents.buffer")

_DEFAULT_BUFFER_PATH = "coriqo_traces_buffer.jsonl"
_ENV_BUFFER_PATH = "CORIQO_TRACE_BUFFER_PATH"

_lock = threading.Lock()


def buffer_path() -> str:
    """Where buffered traces are written. `CORIQO_TRACE_BUFFER_PATH`
    overrides the default (`./coriqo_traces_buffer.jsonl`), so a deployment
    can point it somewhere durable (a mounted volume) without a code change."""
    return os.environ.get(_ENV_BUFFER_PATH, _DEFAULT_BUFFER_PATH)


def buffer_trace(agent_id: str, payload: dict[str, Any], error: Exception) -> None:
    """Appends one failed record_trace() call to the local buffer file and
    logs it. Never raises: a broken disk or an unwritable path must not turn
    an already-degraded situation (Coriqo unreachable) into a second failure
    inside the customer's own code. The buffer write failing is itself only
    logged."""
    row = {
        "buffered_at": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "error": str(error),
        "payload": payload,
    }
    logger.warning(
        "coriqo_agents: record_trace failed for agent %s (%s); buffering locally to %s",
        agent_id, error, buffer_path(),
    )
    try:
        path = buffer_path()
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
    except OSError:
        logger.exception(
            "coriqo_agents: could not write local trace buffer at %s; this trace is lost",
            buffer_path(),
        )


def read_buffered_traces(path: str | None = None) -> list[dict[str, Any]]:
    """Reads back every row `buffer_trace()` has written, for a caller that
    wants to replay them once Coriqo is reachable again. Returns an empty
    list if the file doesn't exist yet; a corrupt line is skipped and logged
    rather than failing the whole read."""
    rows: list[dict[str, Any]] = []
    target = path or buffer_path()
    if not os.path.exists(target):
        return rows
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("coriqo_agents: skipping corrupt line in trace buffer %s", target)
    return rows
