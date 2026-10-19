"""C9 — a Coriqo outage must never raise into the customer's own code.

track_decision() calls client.record_trace() after the wrapped function
returns (or raises). If Coriqo itself is unreachable, that call can raise —
and, before this fix, that exception replaced whatever the wrapped function
actually did: a successful business result, or the caller's own exception,
both got clobbered by an unrelated network error. This proves the wrapped
function's real result/exception always wins, and the failed trace is
logged + buffered locally instead of being dropped or raised.

No live API needed — record_trace is faked to fail, so this runs as a
plain script (matching this directory's other tests) or under pytest.

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_decorator_outage_buffer.py
"""
from __future__ import annotations

import json
import os
import tempfile

from coriqo_agents import AgentSuspendedError, read_buffered_traces, track_decision


class _FailingClient:
    """Stands in for CoriqoAgentClient: record_trace always raises, as it
    would during a real outage (connection refused, timeout, 5xx)."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def record_trace(self, *args, **kwargs):
        self.calls += 1
        raise self._exc


def _main() -> None:
    failures = []

    def check(cond, msg):
        print(("  ok  " if cond else " FAIL ") + msg)
        if not cond:
            failures.append(msg)

    buffer_dir = tempfile.mkdtemp()
    buffer_file = os.path.join(buffer_dir, "buf.jsonl")
    os.environ["CORIQO_TRACE_BUFFER_PATH"] = buffer_file

    # --- successful call survives a record_trace outage -------------------
    outage_client = _FailingClient(ConnectionError("connection refused"))

    @track_decision(outage_client, "agent-1")
    def approve_loan(applicant_id: str) -> str:
        return f"approved for {applicant_id}"

    result = approve_loan("a-1")
    check(result == "approved for a-1",
          "the wrapped function's real result is returned even though record_trace failed")
    check(outage_client.calls == 1, "record_trace was attempted exactly once")

    rows = read_buffered_traces(buffer_file)
    check(len(rows) == 1, "the failed trace was appended to the local buffer")
    check(rows[0]["agent_id"] == "agent-1" and "connection refused" in rows[0]["error"],
          "buffered row carries the agent id and the outage error")
    check(rows[0]["payload"]["output"] == "approved for a-1",
          "buffered row carries the trace payload that would have been sent")

    # --- caller's own exception still propagates, not the outage's --------
    @track_decision(outage_client, "agent-1")
    def always_fails(x: int) -> int:
        raise ValueError("boom")

    raised = None
    try:
        always_fails(1)
    except Exception as e:  # noqa: BLE001
        raised = e
    check(isinstance(raised, ValueError) and str(raised) == "boom",
          "the wrapped function's own exception propagates unchanged, not the outage error")
    check(len(read_buffered_traces(buffer_file)) == 2,
          "the failed call's trace was buffered too")

    # --- a plain JSON line is written, not a pickled/opaque blob -----------
    with open(buffer_file, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    check(len(lines) == 2, "buffer file has one JSON line per failed trace")

    # --- AgentSuspendedError is NOT swallowed — it's a governance signal ---
    suspended_client = _FailingClient(AgentSuspendedError(423, "agent suspended"))

    @track_decision(suspended_client, "agent-2")
    def do_thing() -> str:
        return "did it"

    suspended_raised = None
    try:
        do_thing()
    except AgentSuspendedError as e:
        suspended_raised = e
    check(suspended_raised is not None, "AgentSuspendedError still propagates — it isn't an outage")
    check(len(read_buffered_traces(buffer_file)) == 2,
          "an AgentSuspendedError is not buffered as a failed/outage trace")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All decorator outage-buffer checks passed.")


if __name__ == "__main__":
    _main()
