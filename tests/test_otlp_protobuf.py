"""Proves `otlp_proto_bytes_to_traces` reuses
`otlp_json_to_traces`'s span-to-event mapping rather than re-implementing it
for protobuf. The bar is exact: build ONE
OTLP export, encode it both as protobuf and as JSON, and assert the two
resulting trace dicts are byte-identical — not just "both produce a
trace", which would let a duplicated (and potentially drifted) mapping
pass silently.

Skips cleanly if the optional `otel-proto` extra isn't installed
(`pip install 'coriqo-agents[otel-proto]'`) — this extra must never be
required by the base or `runner` install (module docstring in
otel_ingest.py), so CI for the base package must still pass without it.

Run standalone:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_otlp_protobuf.py
"""
from __future__ import annotations

import base64
import json

import pytest

from coriqo_agents.runner.otel_ingest import (
    SOURCE_OTLP_JSON,
    otlp_json_to_trace,
    otlp_json_to_traces,
)
from coriqo_agents.runner.trace import EventType, events_of

try:
    from google.protobuf.json_format import ParseDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    from coriqo_agents.runner.otel_ingest import (
        otlp_proto_bytes_to_trace,
        otlp_proto_bytes_to_traces,
    )

    HAVE_OTEL_PROTO = True
except ImportError:
    HAVE_OTEL_PROTO = False

requires_otel_proto = pytest.mark.skipif(
    not HAVE_OTEL_PROTO,
    reason="optional 'otel-proto' extra not installed (pip install 'coriqo-agents[otel-proto]')",
)

TRACE_ID_HEX = "5b8aa5a2d2c872e8321cf37308d69df2"
SPAN_ID_HEX = "051581bf3cb55c13"
MCP_SPAN_ID_HEX = "051581bf3cb55c14"


def _b64(hex_str: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_str)).decode("ascii")


def _kv(key: str, **value: object) -> dict:
    return {"key": key, "value": value}


def _otlp_json_payload(spans: list[dict], *, resource_attrs: list[dict] | None = None) -> dict:
    """One `ExportTraceServiceRequest`, in its OTLP/HTTP+JSON encoding —
    the same shape a real OTel JSON exporter POSTs and `otlp_json_to_traces`
    already consumes."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs or [
                    _kv("service.name", stringValue="checkout-agent"),
                    _kv("service.version", stringValue="3.1.0"),
                ]},
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


def _one_export_worth_of_spans() -> list[dict]:
    """A small but representative mix: an LLM call, a generic tool call/
    result pair (exercised via `gen_ai.tool.name`), and an MCP tools/list
    span — enough surface to prove the LLM_CALL, TOOL_CALL/TOOL_RESULT, and
    MCP mapping branches all survive the proto round-trip identically."""
    llm_span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(SPAN_ID_HEX),
        "name": "chat", "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000001000000000",
        "attributes": [
            _kv("gen_ai.system", stringValue="anthropic"),
            _kv("gen_ai.request.model", stringValue="claude-sonnet"),
        ],
    }
    tool_span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64("051581bf3cb55c20"),
        "name": "lookup_balance", "startTimeUnixNano": "1700000002000000000",
        "endTimeUnixNano": "1700000003000000000",
        "attributes": [
            _kv("gen_ai.tool.name", stringValue="lookup_balance"),
        ],
    }
    mcp_span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(MCP_SPAN_ID_HEX),
        "name": "tools/list", "startTimeUnixNano": "1700000004000000000",
        "endTimeUnixNano": "1700000004000000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="tools/list"),
            _kv("mcp.session.id", stringValue="sess-1"),
            _kv("coriqo.mcp.tools", stringValue=json.dumps(
                [{"name": "lookup_balance", "description": "Looks up a balance"}]
            )),
        ],
    }
    return [llm_span, tool_span, mcp_span]


def _proto_bytes_for(json_payload: dict) -> bytes:
    """Build the exact same `ExportTraceServiceRequest` as protobuf bytes,
    from the identical JSON structure used for the JSON-path assertion —
    `ParseDict` is the JSON-format-to-protobuf inverse of the
    `MessageToDict` call `otlp_proto_bytes_to_traces` uses internally, so
    this proves round-tripping through protobuf changes nothing."""
    message = ExportTraceServiceRequest()
    ParseDict(json_payload, message)
    return message.SerializeToString()


@requires_otel_proto
def test_proto_and_json_encodings_of_one_export_produce_identical_trace_dict():
    payload = _otlp_json_payload(_one_export_worth_of_spans())
    proto_bytes = _proto_bytes_for(payload)

    json_traces = otlp_json_to_traces(payload)
    proto_traces = otlp_proto_bytes_to_traces(proto_bytes)

    assert json_traces == proto_traces, (
        "otlp_proto_bytes_to_traces must reuse otlp_json_to_traces's mapping "
        "exactly, not a separately maintained protobuf mapping — any diff "
        "here is drift between the two paths"
    )
    # Sanity: both paths actually produced real content, not two empty lists
    # that trivially compare equal.
    assert json_traces and json_traces[0]["events"]
    assert json_traces[0]["source"] == SOURCE_OTLP_JSON
    assert json_traces[0]["agent_id"] == "checkout-agent"
    assert events_of(json_traces[0], EventType.LLM_CALL)
    assert events_of(json_traces[0], EventType.TOOL_CALL)
    assert events_of(json_traces[0], EventType.MCP_TOOL_LIST)


@requires_otel_proto
def test_single_agent_proto_wrapper_matches_json_wrapper():
    payload = _otlp_json_payload(_one_export_worth_of_spans())
    proto_bytes = _proto_bytes_for(payload)

    assert otlp_json_to_trace(payload) == otlp_proto_bytes_to_trace(proto_bytes)


@requires_otel_proto
def test_proto_path_still_resolves_multiple_agents_separately():
    """Same multi-identity grouping rule (module docstring: "each
    resourceSpans entry gets its OWN identity resolution") must hold for
    protobuf-decoded input too, since it comes from the same
    `otlp_json_to_traces` call."""
    span_a = _one_export_worth_of_spans()[0]
    span_b = dict(_one_export_worth_of_spans()[1])
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": [_kv("service.name", stringValue="agent-a")]},
                "scopeSpans": [{"spans": [span_a]}],
            },
            {
                "resource": {"attributes": [_kv("service.name", stringValue="agent-b")]},
                "scopeSpans": [{"spans": [span_b]}],
            },
        ]
    }
    proto_bytes = _proto_bytes_for(payload)

    json_traces = otlp_json_to_traces(payload)
    proto_traces = otlp_proto_bytes_to_traces(proto_bytes)

    assert {t["agent_id"] for t in json_traces} == {"agent-a", "agent-b"}
    assert json_traces == proto_traces


def test_otel_proto_extra_is_not_in_base_install():
    """Guards the hard rule that `otel-proto` stays an OPTIONAL extra: the
    base package import must not transitively require `opentelemetry` or
    `google.protobuf` — otel_ingest.py's JSON path is pure stdlib.

    Code review round 1, finding #3: the previous version of this test
    asserted `"opentelemetry" not in sys.modules or HAVE_OTEL_PROTO` in the
    SAME process that had already imported `opentelemetry.proto...` at
    module scope (to build `HAVE_OTEL_PROTO`/the other tests in this file)
    — so `HAVE_OTEL_PROTO` was always True whenever this test could run at
    all, making the assertion tautologically true and unable to catch a
    future eager `import opentelemetry` added to `otel_ingest.py`.

    This version runs a real isolation check in a SEPARATE subprocess that
    blocks `opentelemetry`/`google.protobuf` from being importable (a
    `sys.meta_path` finder raising `ImportError` for those names, installed
    before `otel_ingest` is ever imported), then imports `otel_ingest` and
    exercises its default JSON path (`otlp_json_to_traces`) end to end. If
    `otel_ingest.py` ever grows an eager top-level `import opentelemetry`,
    the subprocess's own import of `otel_ingest` fails and this test fails
    with it — a guarantee the previous assertion could not provide.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    sdk_root = Path(__file__).resolve().parents[1]  # sdk/coriqo-agents

    script = """
import sys

_BLOCKED_PREFIXES = ("opentelemetry", "google.protobuf", "google")

class _BlockImport:
    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + ".") for p in _BLOCKED_PREFIXES):
            raise ImportError(f"blocked for isolation test: {name}")
        return None

sys.meta_path.insert(0, _BlockImport())

from coriqo_agents.runner.otel_ingest import otlp_json_to_traces

payload = {
    "resourceSpans": [{
        "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "isolation-agent"}},
        ]},
        "scopeSpans": [{"spans": [{
            "traceId": "WwqlotLIcugyHPNzCNad8g==",
            "spanId": "BRWBvzy1XBM=",
            "name": "chat",
            "startTimeUnixNano": "1700000000000000000",
            "endTimeUnixNano": "1700000001000000000",
            "attributes": [
                {"key": "gen_ai.system", "value": {"stringValue": "anthropic"}},
            ],
        }]}],
    }],
}
traces = otlp_json_to_traces(payload)
assert traces and traces[0]["agent_id"] == "isolation-agent", traces

# The blocker must never have been bypassed — otel_ingest.py's JSON path
# must not have imported anything under the blocked prefixes.
blocked_loaded = [m for m in sys.modules if m.split(".")[0] in ("opentelemetry", "google")]
assert not blocked_loaded, f"otel_ingest's JSON path imported: {blocked_loaded}"
print("ISOLATION_OK")
"""

    result = subprocess.run(
        [_sys.executable, "-c", script],
        cwd=str(sdk_root), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0 and "ISOLATION_OK" in result.stdout, (
        f"isolation subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
