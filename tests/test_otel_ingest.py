"""Proves `otel_ingest.otlp_json_to_trace` turns a real OTLP/HTTP+JSON
export payload into the exact `trace.py` shape every validator already
consumes — the phase-3 validator tests build that shape by hand
(`_event`/`CLEAN_TRACE` in api/tests/test_assurance_workbench_phase3_validators.py);
this proves the converter produces the same shape from real OTel wire data
instead.

Pure-stdlib, no app/DB dependency (unlike test_client_smoke.py) — run
standalone:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_otel_ingest.py
"""
from __future__ import annotations

import base64

from coriqo_agents.runner.otel_ingest import (
    SOURCE_OTLP_JSON,
    MultipleAgentsInBatchError,
    otlp_json_bytes_to_trace,
    otlp_json_to_trace,
    otlp_json_to_traces,
)
from coriqo_agents.runner.trace import EventType, events_of, payload_text

TRACE_ID_HEX = "5b8aa5a2d2c872e8321cf37308d69df2"
SPAN_ID_HEX = "051581bf3cb55c13"


def _b64(hex_str: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_str)).decode("ascii")


def _kv(key: str, **value: object) -> dict:
    return {"key": key, "value": value}


def _otlp_payload(spans: list[dict]) -> dict:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _kv("coriqo.agent_id", stringValue="agent_44"),
                        _kv("coriqo.agent_version", stringValue="2.4.1"),
                    ]
                },
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


def test_explicit_event_type_wins():
    span = {
        "traceId": _b64(TRACE_ID_HEX),
        "spanId": _b64(SPAN_ID_HEX),
        "name": "final-answer",
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000000500000000",
        "attributes": [_kv("coriqo.event_type", stringValue=EventType.FINAL_RESPONSE)],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))

    assert trace["trace_id"] == TRACE_ID_HEX
    assert trace["agent_id"] == "agent_44"
    assert trace["agent_version"] == "2.4.1"
    assert trace["source"] == SOURCE_OTLP_JSON
    final = events_of(trace, EventType.FINAL_RESPONSE)
    assert len(final) == 1
    assert final[0]["event_id"] == SPAN_ID_HEX


def test_gen_ai_attributes_map_to_llm_call():
    span = {
        "traceId": _b64(TRACE_ID_HEX),
        "spanId": _b64(SPAN_ID_HEX),
        "name": "chat gpt-4",
        "startTimeUnixNano": "1700000001000000000",
        "endTimeUnixNano": "1700000001200000000",
        "attributes": [
            _kv("gen_ai.system", stringValue="openai"),
            _kv("gen_ai.request.model", stringValue="gpt-4"),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    llm_calls = events_of(trace, EventType.LLM_CALL)
    assert len(llm_calls) == 1
    assert "gpt-4" in payload_text(llm_calls[0])


def test_tool_span_expands_to_call_and_result_pair():
    span = {
        "traceId": _b64(TRACE_ID_HEX),
        "spanId": _b64(SPAN_ID_HEX),
        "name": "update_application",
        "startTimeUnixNano": "1700000002000000000",
        "endTimeUnixNano": "1700000002300000000",
        "attributes": [
            _kv("gen_ai.tool.name", stringValue="loan_system"),
            _kv("coriqo.authorization", stringValue="APPROVED"),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))

    calls = events_of(trace, EventType.TOOL_CALL)
    results = events_of(trace, EventType.TOOL_RESULT)
    assert len(calls) == 1 and len(results) == 1
    assert calls[0]["tool"] == "loan_system"
    assert calls[0]["authorization"] == "APPROVED"
    assert calls[0]["timestamp"] < results[0]["timestamp"]
    assert calls[0]["event_id"] != results[0]["event_id"]


def test_unclassified_span_falls_back_to_external_side_effect():
    span = {
        "traceId": _b64(TRACE_ID_HEX),
        "spanId": _b64(SPAN_ID_HEX),
        "name": "mystery-span",
        "startTimeUnixNano": "1700000003000000000",
        "endTimeUnixNano": "1700000003000000000",
        "attributes": [],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    assert len(events_of(trace, EventType.EXTERNAL_SIDE_EFFECT)) == 1


def test_agent_id_override_beats_resource_attributes():
    span = {
        "traceId": _b64(TRACE_ID_HEX),
        "spanId": _b64(SPAN_ID_HEX),
        "name": "noop",
        "startTimeUnixNano": "1700000004000000000",
        "endTimeUnixNano": "1700000004000000000",
        "attributes": [],
    }
    trace = otlp_json_to_trace(
        _otlp_payload([span]), agent_id="override_agent", agent_version="9.9.9"
    )
    assert trace["agent_id"] == "override_agent"
    assert trace["agent_version"] == "9.9.9"


def _multi_agent_payload() -> dict:
    """Two resourceSpans naming two different agents in one export — the
    shape a bank's shared Collector/Fluent Bit/Vector pipeline produces when
    it multiplexes several services through one OTLP batch."""
    def resource_span(agent_id: str, span_id_hex: str) -> dict:
        return {
            "resource": {"attributes": [_kv("coriqo.agent_id", stringValue=agent_id)]},
            "scopeSpans": [{"spans": [{
                "traceId": _b64(TRACE_ID_HEX),
                "spanId": _b64(span_id_hex),
                "name": "noop",
                "startTimeUnixNano": "1700000006000000000",
                "endTimeUnixNano": "1700000006000000000",
                "attributes": [],
            }]}],
        }
    return {"resourceSpans": [
        resource_span("agent_A", "0000000000000001"),
        resource_span("agent_B", "0000000000000002"),
    ]}


def test_batch_with_two_agents_splits_into_two_traces():
    traces = otlp_json_to_traces(_multi_agent_payload())
    assert len(traces) == 2
    by_agent = {t["agent_id"]: t for t in traces}
    assert set(by_agent) == {"agent_A", "agent_B"}
    assert len(by_agent["agent_A"]["events"]) == 1
    assert len(by_agent["agent_B"]["events"]) == 1


def test_singular_wrapper_raises_on_mixed_agent_batch():
    try:
        otlp_json_to_trace(_multi_agent_payload())
    except MultipleAgentsInBatchError as exc:
        assert "agent_A" in str(exc) and "agent_B" in str(exc)
    else:
        raise AssertionError("expected MultipleAgentsInBatchError")


def test_override_forces_one_identity_even_across_mixed_resources():
    trace = otlp_json_to_trace(
        _multi_agent_payload(), agent_id="override_agent", agent_version="1.0",
    )
    assert trace["agent_id"] == "override_agent"
    assert len(trace["events"]) == 2, "both resources' spans land under the one forced identity"


def test_bytes_entrypoint_parses_json_body():
    span = {
        "traceId": _b64(TRACE_ID_HEX),
        "spanId": _b64(SPAN_ID_HEX),
        "name": "noop",
        "startTimeUnixNano": "1700000005000000000",
        "endTimeUnixNano": "1700000005000000000",
        "attributes": [],
    }
    import json

    body = json.dumps(_otlp_payload([span])).encode("utf-8")
    trace = otlp_json_bytes_to_trace(body)
    assert trace["trace_id"] == TRACE_ID_HEX


def test_empty_payload_returns_empty_trace():
    trace = otlp_json_to_trace({})
    assert trace == {"trace_id": "", "agent_id": "", "agent_version": "", "events": []}


# --- mcp.method.name spans map to MCP_SERVER_CONNECT/MCP_TOOL_LIST/
# TOOL_CALL rather than the generic tool-pair/LLM_CALL/EXTERNAL_SIDE_EFFECT
# rules above (module docstring case 0).

def test_mcp_initialize_maps_to_server_connect():
    span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(SPAN_ID_HEX),
        "name": "initialize", "startTimeUnixNano": "1700000006000000000",
        "endTimeUnixNano": "1700000006000000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="initialize"),
            _kv("mcp.session.id", stringValue="sess-1"),
            _kv("server.address", stringValue="mcp.example.com"),
            _kv("server.port", intValue="443"),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    connects = events_of(trace, EventType.MCP_SERVER_CONNECT)
    assert len(connects) == 1
    assert connects[0]["payload"]["server_id"] == "sess-1"
    assert connects[0]["payload"]["url_hash"].startswith("sha256:")
    assert connects[0]["payload"]["tools"] == []


def test_mcp_initialize_reads_declared_tools_when_exporter_captures_them():
    """Regression (round 3, plausible): a malicious server could declare a
    tool only in its `initialize` capabilities and never emit a separate
    tools/list span — a hardcoded `tools: []` on MCP_SERVER_CONNECT would
    then leave mcp_tool_shadowing/mcp_tool_description_injection nothing
    to check. Upstream OTel GenAI semconv defines no tool-array attribute
    on `initialize` (checked again for this fix), so this reads the same
    `coriqo.mcp.tools` extension `tools/list` already reads, for an
    exporter that does capture initialize-time capabilities."""
    span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(SPAN_ID_HEX),
        "name": "initialize", "startTimeUnixNano": "1700000006000000000",
        "endTimeUnixNano": "1700000006000000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="initialize"),
            _kv("mcp.session.id", stringValue="sess-1"),
            _kv("coriqo.mcp.tools", stringValue='[{"name": "search", "description": "Search the case files."}]'),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    connects = events_of(trace, EventType.MCP_SERVER_CONNECT)
    assert connects[0]["payload"]["tools"][0]["name"] == "search"


def test_mcp_tools_list_maps_to_tool_list_with_tools():
    span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(SPAN_ID_HEX),
        "name": "tools/list", "startTimeUnixNano": "1700000007000000000",
        "endTimeUnixNano": "1700000007000000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="tools/list"),
            _kv("mcp.session.id", stringValue="sess-1"),
            _kv("coriqo.mcp.tools", stringValue='[{"name": "search", "description": "Search the case files."}]'),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    listings = events_of(trace, EventType.MCP_TOOL_LIST)
    assert len(listings) == 1
    tools = listings[0]["payload"]["tools"]
    assert tools[0]["name"] == "search"
    assert tools[0]["description"] == "Search the case files."
    assert tools[0]["description_hash"].startswith("sha256:")


def test_mcp_tools_call_maps_to_tool_call_and_result_with_server_id():
    span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(SPAN_ID_HEX),
        "name": "tools/call", "startTimeUnixNano": "1700000008000000000",
        "endTimeUnixNano": "1700000008100000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="tools/call"),
            _kv("mcp.session.id", stringValue="sess-1"),
            _kv("gen_ai.tool.name", stringValue="search"),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    calls = events_of(trace, EventType.TOOL_CALL)
    results = events_of(trace, EventType.TOOL_RESULT)
    assert len(calls) == 1 and len(results) == 1
    assert calls[0]["tool"] == "search"
    assert calls[0]["payload"]["server_id"] == "sess-1"
    assert results[0]["payload"]["server_id"] == "sess-1"


def test_mcp_tools_call_payload_keeps_flattened_attribute_content():
    """Regression: the tools/call mapping used to build the TOOL_CALL/
    TOOL_RESULT payload as a bare {"server_id": ...}, dropping the same
    flattened-attribute payload every other span type gets. A validator
    reading the actual call/result CONTENT (mcp_cross_server_flow's
    substring match) then had nothing to match against on any real
    OTel-sourced trace — a hand-built test event that sets payload content
    directly never exercises this path, which is exactly how this slipped
    through review. Any non-mcp.*/coriqo.* attribute on the span (an
    argument, a result value) must survive into the payload alongside
    server_id."""
    span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(SPAN_ID_HEX),
        "name": "tools/call", "startTimeUnixNano": "1700000008000000000",
        "endTimeUnixNano": "1700000008100000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="tools/call"),
            _kv("mcp.session.id", stringValue="sess-1"),
            _kv("gen_ai.tool.name", stringValue="pay"),
            _kv("account_number", stringValue="account-9988776655"),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    call = events_of(trace, EventType.TOOL_CALL)[0]
    result = events_of(trace, EventType.TOOL_RESULT)[0]
    assert call["payload"]["server_id"] == "sess-1"
    assert call["payload"]["account_number"] == "account-9988776655"
    assert result["payload"]["account_number"] == "account-9988776655"


def test_mcp_cross_server_flow_validator_catches_a_real_otlp_sourced_leak():
    """End-to-end: real OTLP/HTTP+JSON spans -> otlp_json_to_trace ->
    mcp_cross_server_flow.evaluate — not a hand-built event dict. This is
    the path the hand-built validator unit tests never exercised, which is
    how the tools/call bare-payload regression above went undetected."""
    from coriqo_agents.runner.contract import ValidatorContext
    from coriqo_agents.runner.validators.mcp_cross_server_flow import McpCrossServerFlowValidator

    result_span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64("0000000000000001"),
        "name": "tools/call", "startTimeUnixNano": "1700000010000000000",
        "endTimeUnixNano": "1700000010100000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="tools/call"),
            _kv("mcp.session.id", stringValue="srv-a"),
            _kv("gen_ai.tool.name", stringValue="lookup"),
            _kv("account_number", stringValue="account-9988776655"),
        ],
    }
    leak_span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64("0000000000000002"),
        "name": "tools/call", "startTimeUnixNano": "1700000011000000000",
        "endTimeUnixNano": "1700000011100000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="tools/call"),
            _kv("mcp.session.id", stringValue="srv-evil"),
            _kv("gen_ai.tool.name", stringValue="pay"),
            _kv("account_number", stringValue="account-9988776655"),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([result_span, leak_span]))

    context = ValidatorContext(
        execution_id="e1", agent_model_id="agent_44", agent_version_ref="2.4.1",
        environment="PRODUCTION", configuration={}, inputs={"trace": trace},
    )
    validator = McpCrossServerFlowValidator()
    raw = validator.execute(context)
    result = validator.evaluate(raw)
    assert result.status == "FAIL", (
        "the leaked account number must be found in a real OTLP-derived trace, "
        "not just a hand-built one — this is the regression the bare-payload "
        "bug above hid from every hand-built validator test"
    )


def test_mcp_tools_list_with_malformed_tools_json_degrades_to_empty():
    span = {
        "traceId": _b64(TRACE_ID_HEX), "spanId": _b64(SPAN_ID_HEX),
        "name": "tools/list", "startTimeUnixNano": "1700000009000000000",
        "endTimeUnixNano": "1700000009000000000",
        "attributes": [
            _kv("mcp.method.name", stringValue="tools/list"),
            _kv("mcp.session.id", stringValue="sess-1"),
            _kv("coriqo.mcp.tools", stringValue="not json"),
        ],
    }
    trace = otlp_json_to_trace(_otlp_payload([span]))
    listings = events_of(trace, EventType.MCP_TOOL_LIST)
    assert listings[0]["payload"]["tools"] == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
