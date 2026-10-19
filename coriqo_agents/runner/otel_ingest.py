"""Converts real OpenTelemetry OTLP trace data into `trace.py`'s event
schema, so the validators can evaluate a trace collected by an
agent's own OTel instrumentation instead of a hand-built dict.

The runner's native event schema is called "OTLP-shaped," but until this
module nothing in the runner actually parsed real OTLP — every existing
trace was constructed by hand (see the phase-3 validator tests' `_event`/
`CLEAN_TRACE` fixtures). This module is the missing half: it accepts the
OTLP/HTTP+JSON export payload a real OTel SDK/collector sends
(`ExportTraceServiceRequest`'s JSON encoding, one `resourceSpans` list of
`scopeSpans` of `spans`) and produces the `{"trace_id", "agent_id",
"agent_version", "events": [...]}` shape `events_of`/`payload_text` and
every validator already consume. No changes to any validator, `engine.py`,
or `contract.py` are needed — this only produces their existing input shape
from a different source.

JSON by default, protobuf as an optional extra: the OTLP/HTTP+JSON content
type decodes with the stdlib `json` module alone, keeping the base install
dependency-free like the rest of the runner (zero LLM deps; the
`runner` extra's only dependency is `cryptography`, for signing). A customer
whose collector only speaks protobuf (OTLP/gRPC, or
`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`) installs the separate
`otel-proto` extra and calls `otlp_proto_bytes_to_traces`
instead — it decodes the protobuf `ExportTraceServiceRequest` into the exact
same dict shape `otlp_json_to_traces` reads (OTLP/HTTP+JSON *is* the
protobuf JSON mapping) and then calls straight into it, so every mapping
rule below lives in one place regardless of wire format. A customer who
never installs `otel-proto` keeps the zero-extra-dependency JSON path as
the default; nothing about it changed.

Event-type mapping (best-effort — a span carries no EventType natively):

0. A span carrying `mcp.method.name` (the OTel GenAI MCP semantic
   convention) maps to MCP_SERVER_CONNECT ("initialize"), MCP_TOOL_LIST
   ("tools/list"), or a TOOL_CALL/TOOL_RESULT pair ("tools/call") — checked
   before every case below, since a tools/call span also carries
   `gen_ai.tool.name` and would otherwise match case 2's generic tool-pair
   rule and lose MCP routing (`payload.server_id`).
1. An explicit `coriqo.event_type` span attribute wins outright, if its
   value is one of `EventType.ALL`.
2. Otherwise, a span carrying `gen_ai.tool.name` or `tool.name` maps to a
   TOOL_CALL/TOOL_RESULT *pair* — one span models a whole call in OTel, but
   the runner's schema tracks the call and its result as separate events, so this
   module emits both from the same span's start/end. Checked before case 3
   because a tool-call span commonly carries other `gen_ai.*` attributes
   too (model, system) alongside `gen_ai.tool.name`.
3. Otherwise, a span carrying any `gen_ai.*` attribute (the OTel Semantic
   Conventions for Generative AI: `gen_ai.system`, `gen_ai.request.model`,
   etc.) maps to `LLM_CALL`.
4. Anything else maps to `EXTERNAL_SIDE_EFFECT`, the schema's catch-all for
   an observed action that isn't a modeled lifecycle stage. This is a
   deliberately weak default — case 1 is how a customer gets exact mapping
   for spans this heuristic gets wrong.

`authorization` is only ever set from an explicit `coriqo.authorization`
span attribute (matching `Authorization.APPROVED/DENIED/PENDING`) — a span
carries no authorization concept of its own, so inventing one from span
status would manufacture a signal the validator contract does not allow.

One export batch can name more than one agent. A bank's own OTel Collector
(or Fluent Bit, Vector, whatever they already run — anything that speaks
OTLP is a valid emitter, not just the Collector) commonly multiplexes many
services through one pipeline, so a single `ExportTraceServiceRequest` can
carry `resourceSpans` for several different agents in one batch. Each
`resourceSpans` entry gets its OWN identity resolution (`coriqo.agent_id`/
`service.name` on THAT resource) — spans are grouped by resolved
`(agent_id, agent_version)`, never merged across a different identity, since
an execution's trace belongs to exactly one agent. `otlp_json_to_traces`
returns one trace dict per distinct identity found; `otlp_json_to_trace`
is the single-agent convenience wrapper and raises `MultipleAgentsInBatchError`
rather than silently collapsing a mixed batch into one trace under whichever
identity happened to resolve first.

Every trace this module returns carries `"source": SOURCE_OTLP_JSON` — see
that constant's own docstring for why: it's how a validator run can later
disclose, in the uploaded manifest, that it evaluated a real OTel export
rather than a hand-built trace dict.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .trace import Authorization, EventType

_CORIQO_EVENT_TYPE_ATTR = "coriqo.event_type"
_CORIQO_AUTHORIZATION_ATTR = "coriqo.authorization"
_CORIQO_ACTOR_ATTR = "coriqo.actor"
_GEN_AI_TOOL_NAME_ATTRS = ("gen_ai.tool.name", "tool.name")

# MCP span attributes, per the OpenTelemetry GenAI semantic
# conventions repo (`open-telemetry/semantic-conventions-genai`,
# `docs/gen-ai/mcp.md` / `docs/registry/attributes/mcp.md`, checked when
# this mapping was written. MCP attributes live there now, moved out
# of the core `open-telemetry/semantic-conventions` repo, and are still
# "Development" stability, so a future OTel release can rename them).
# Recognized ahead of the generic `_classify` cases below (a tools/call span
# also carries `gen_ai.tool.name`, which would otherwise match the
# `_TOOL_PAIR` case first and lose the MCP server_id/routing information).
_MCP_METHOD_ATTR = "mcp.method.name"
_MCP_SESSION_ID_ATTR = "mcp.session.id"
#: The GenAI semconv repo defines no stable attribute carrying a tools/list
#: RESPONSE's full tool array with descriptions — `gen_ai.tool.name` is
#: singular, scoped to one invoked tool. Coriqo's own runner instrumentation
#: emits the roster as this JSON-encoded custom attribute on the tools/list
#: span itself; documented as a Coriqo extension, not upstream semconv.
_MCP_TOOLS_ATTR = "coriqo.mcp.tools"


def _sha256(value: str | None) -> str | None:
    if not value:
        return None
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mcp_server_url(attrs: dict[str, Any]) -> str | None:
    address = attrs.get("server.address")
    if not address:
        return None
    port = attrs.get("server.port")
    return f"{address}:{port}" if port else str(address)


def _mcp_tools(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Read defensively: absent, malformed, or non-list JSON all degrade to
    no tools rather than raising — a span from an exporter that predates
    this convention should not crash ingestion."""
    raw = attrs.get(_MCP_TOOLS_ATTR)
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        {
            "name": t.get("name"),
            "description": t.get("description"),
            "description_hash": _sha256(t.get("description")),
        }
        for t in parsed if isinstance(t, dict) and t.get("name")
    ]

SOURCE_OTLP_JSON = "otlp/json"
"""Stamped onto every trace this module produces (`trace["source"]`) — an
extra key `events_of`/`payload_text` and every validator already ignore
(trace.py's `{"trace_id", "agent_id", "agent_version", "events"}` shape has
no closed-schema check), so it costs nothing to consumers that don't care.
`RunnerEngine.run()` reads it to disclose, in the uploaded manifest, whether
a validator ran against a real OTLP export or a hand-built trace dict —
otherwise nothing distinguishes the two once a trace has been converted."""


class MultipleAgentsInBatchError(ValueError):
    """`otlp_json_to_trace` was handed a batch naming more than one distinct
    agent identity — merging their spans into one trace would attribute one
    agent's events to another's execution. Use `otlp_json_to_traces` (plural)
    to get one trace dict per agent found in the batch instead."""


def _decode_id(value: Any) -> str:
    """OTLP/JSON encodes proto `bytes` fields (traceId, spanId) as base64
    per the protobuf JSON mapping — decode to the hex string every other
    OTel-adjacent tool (Jaeger, the collector's own logs) displays."""
    if not value:
        return ""
    if isinstance(value, str):
        try:
            return base64.b64decode(value).hex()
        except (ValueError, TypeError):
            return value
    return str(value)


def _unwrap_any_value(value: dict[str, Any] | None) -> Any:
    """OTLP's `AnyValue` is a oneof wrapper: {"stringValue": "x"},
    {"intValue": "3"}, {"boolValue": true}, {"arrayValue": {"values": [...]}},
    {"kvlistValue": {"values": [...]}}. Unwrap to a plain Python value."""
    if not value:
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "bytesValue" in value:
        return value["bytesValue"]
    if "arrayValue" in value:
        return [_unwrap_any_value(v) for v in (value["arrayValue"].get("values") or [])]
    if "kvlistValue" in value:
        return _attributes_to_dict(value["kvlistValue"].get("values") or [])
    return None


def _attributes_to_dict(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(key): _unwrap_any_value(a.get("value"))
        for a in attributes
        if (key := a.get("key"))
    }


def _nanos_to_iso(nanos: Any) -> str:
    """Span timestamps are `fixed64` nanoseconds since epoch, encoded as a
    JSON string. `datetime` only carries microsecond precision, which is
    already finer than any validator here reads timestamps for."""
    try:
        seconds = int(nanos) / 1_000_000_000
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _classify(attrs: dict[str, Any]) -> str:
    explicit = attrs.get(_CORIQO_EVENT_TYPE_ATTR)
    if explicit in EventType.ALL:
        return explicit
    if any(k in attrs for k in _GEN_AI_TOOL_NAME_ATTRS):
        return "_TOOL_PAIR"  # internal marker, expanded by the caller
    if any(k.startswith("gen_ai.") for k in attrs):
        return EventType.LLM_CALL
    return EventType.EXTERNAL_SIDE_EFFECT


def _tool_name(attrs: dict[str, Any]) -> str | None:
    for key in _GEN_AI_TOOL_NAME_ATTRS:
        if attrs.get(key) is not None:
            return str(attrs[key])
    return None


def _span_to_events(
    span: dict[str, Any],
    *,
    trace_id: str,
    agent_id: str,
    agent_version: str,
) -> list[dict[str, Any]]:
    attrs = _attributes_to_dict(span.get("attributes") or [])
    span_id = _decode_id(span.get("spanId"))
    name = span.get("name") or ""
    start_iso = _nanos_to_iso(span.get("startTimeUnixNano"))
    end_iso = _nanos_to_iso(span.get("endTimeUnixNano")) or start_iso
    authorization = attrs.get(_CORIQO_AUTHORIZATION_ATTR)
    if authorization not in (Authorization.APPROVED, Authorization.DENIED, Authorization.PENDING):
        authorization = None
    actor = attrs.get(_CORIQO_ACTOR_ATTR) or "agent"
    payload = {k: v for k, v in attrs.items() if not k.startswith("coriqo.")}

    def _event(event_type: str, *, event_id: str, timestamp: str, action: str | None,
               tool: str | None) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "trace_id": trace_id,
            "agent_id": agent_id,
            "agent_version": agent_version,
            "event_type": event_type,
            "tool": tool,
            "action": action,
            "actor": actor,
            "authorization": authorization,
            "timestamp": timestamp,
            "payload": payload,
        }

    # MCP spans get their own payload shape (server_id/tools, per
    # trace.py's MCP_SERVER_CONNECT/MCP_TOOL_LIST docstrings) rather than the
    # generic flattened-attributes payload every other span gets — the four
    # mcp_* validators key off `payload.server_id`/`payload.tools`
    # specifically, not off raw span attribute names. Checked before
    # `_classify` because a tools/call span also carries `gen_ai.tool.name`
    # and would otherwise match `_TOOL_PAIR` first, losing `server_id`.
    method = attrs.get(_MCP_METHOD_ATTR)
    if method == "initialize":
        server_id = attrs.get(_MCP_SESSION_ID_ATTR)
        # Round 3: the MCP `initialize` handshake is where a server DECLARES
        # its capabilities, including which tools it exposes — a malicious
        # server could declare a tool only here and never emit a separate
        # tools/list span, in which case a hardcoded `tools: []` would leave
        # `mcp_tool_shadowing`/`mcp_tool_description_injection` with nothing
        # to check and silently PASS a trace that actually carries the
        # declaration. Checked the OTel GenAI semconv again (same source as
        # the arguments/result fix): the `initialize` span's OWN attribute
        # table (docs/gen-ai/mcp.md, Server/Client sections) carries no tool-
        # array attribute at all — upstream has not defined one. This reads
        # the same `coriqo.mcp.tools` extension `tools/list` already reads
        # (module docstring) so an exporter that DOES capture initialize-time
        # capabilities has somewhere to put them; one that doesn't degrades
        # to the same empty list as before, which is the honest state for
        # what upstream semconv can express today.
        return [{
            "event_id": span_id, "trace_id": trace_id, "agent_id": agent_id,
            "agent_version": agent_version, "event_type": EventType.MCP_SERVER_CONNECT,
            "tool": None, "action": name, "actor": actor, "authorization": authorization,
            "timestamp": start_iso,
            "payload": {
                "server_id": server_id, "url_hash": _sha256(_mcp_server_url(attrs)),
                "tools": _mcp_tools(attrs),
            },
        }]
    if method == "tools/list":
        server_id = attrs.get(_MCP_SESSION_ID_ATTR)
        return [{
            "event_id": span_id, "trace_id": trace_id, "agent_id": agent_id,
            "agent_version": agent_version, "event_type": EventType.MCP_TOOL_LIST,
            "tool": None, "action": name, "actor": actor, "authorization": authorization,
            "timestamp": start_iso,
            "payload": {"server_id": server_id, "tools": _mcp_tools(attrs)},
        }]
    if method == "tools/call":
        server_id = attrs.get(_MCP_SESSION_ID_ATTR)
        tool = _tool_name(attrs)
        # TOOL_CALL and TOOL_RESULT must NOT share one payload dict —
        # round-1 fixed the (worse) bug of dropping content entirely, but
        # handing both events the identical flattened-attribute blob is
        # still wrong: `gen_ai.tool.call.arguments` and `gen_ai.tool.call.
        # result` are the two GenAI semconv attributes that actually
        # distinguish request content from response content on a tools/call
        # span (checked via GitHits against
        # open-telemetry/semantic-conventions-genai — both "Opt-In",
        # `docs/gen-ai/mcp.md`'s Server attribute table). Sharing one
        # payload means a legitimate ARGUMENT value passed by design to two
        # different servers (the same account number sent to a lookup
        # server and a posting server, no real cross-server flow) reads as
        # server A's RESULT leaking into server B's call — a false FAIL on
        # a fully compliant trace.
        #
        # `gen_ai.tool.call.result` is Opt-In and many exporters won't set
        # it on a request-shaped tools/call span at all. When it's absent,
        # this disclosed the gap explicitly by leaving TOOL_RESULT's
        # content empty rather than silently reusing the call's own
        # arguments as if they were response content.
        base = {k: v for k, v in payload.items()
                if k not in ("gen_ai.tool.call.arguments", "gen_ai.tool.call.result")}
        call_payload = {**base, "server_id": server_id}
        if "gen_ai.tool.call.arguments" in payload:
            call_payload["arguments"] = payload["gen_ai.tool.call.arguments"]
        result_payload = {**base, "server_id": server_id}
        if "gen_ai.tool.call.result" in payload:
            result_payload["result"] = payload["gen_ai.tool.call.result"]
        return [{
            "event_id": f"{span_id}:call", "trace_id": trace_id, "agent_id": agent_id,
            "agent_version": agent_version, "event_type": EventType.TOOL_CALL,
            "tool": tool, "action": name, "actor": actor, "authorization": authorization,
            "timestamp": start_iso, "payload": call_payload,
        }, {
            "event_id": f"{span_id}:result", "trace_id": trace_id, "agent_id": agent_id,
            "agent_version": agent_version, "event_type": EventType.TOOL_RESULT,
            "tool": tool, "action": name, "actor": actor, "authorization": authorization,
            "timestamp": end_iso, "payload": result_payload,
        }]

    classification = _classify(attrs)
    if classification == "_TOOL_PAIR":
        tool = _tool_name(attrs)
        return [
            _event(EventType.TOOL_CALL, event_id=f"{span_id}:call", timestamp=start_iso,
                   action=name, tool=tool),
            _event(EventType.TOOL_RESULT, event_id=f"{span_id}:result", timestamp=end_iso,
                   action=name, tool=tool),
        ]

    return [_event(classification, event_id=span_id, timestamp=start_iso, action=name, tool=None)]


def _resolve_identity(
    resource_attrs: dict[str, Any], *, agent_id: str | None, agent_version: str | None,
) -> tuple[str, str]:
    resolved_id = agent_id if agent_id is not None else (
        resource_attrs.get("coriqo.agent_id") or resource_attrs.get("service.name")
    )
    resolved_version = agent_version if agent_version is not None else (
        resource_attrs.get("coriqo.agent_version") or resource_attrs.get("service.version")
    )
    return resolved_id or "", resolved_version or ""


def otlp_json_to_traces(
    payload: dict[str, Any],
    *,
    agent_id: str | None = None,
    agent_version: str | None = None,
) -> list[dict[str, Any]]:
    """Convert one OTLP/HTTP+JSON `ExportTraceServiceRequest` body into a
    `trace.py`-shaped trace dict PER distinct agent identity found in the
    batch (module docstring — a shared pipeline commonly multiplexes several
    agents through one export). `agent_id`/`agent_version`, if given,
    override every resource's own attributes and force the WHOLE batch under
    one identity — for a caller that already knows which single agent this
    upload belongs to out of band (e.g. from the authenticated device),
    the same override behavior `otlp_json_to_trace` has always had.

    Order is stable but not otherwise meaningful: one entry per identity, in
    the order each identity was first encountered."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    for resource_span in payload.get("resourceSpans") or []:
        resource_attrs = _attributes_to_dict(
            (resource_span.get("resource") or {}).get("attributes") or []
        )
        identity = _resolve_identity(resource_attrs, agent_id=agent_id, agent_version=agent_version)
        group = groups.setdefault(identity, {"trace_id": "", "events": []})

        for scope_span in resource_span.get("scopeSpans") or []:
            for span in scope_span.get("spans") or []:
                span_trace_id = _decode_id(span.get("traceId"))
                if span_trace_id and not group["trace_id"]:
                    group["trace_id"] = span_trace_id
                group["events"].extend(
                    _span_to_events(
                        span,
                        trace_id=span_trace_id or group["trace_id"],
                        agent_id=identity[0],
                        agent_version=identity[1],
                    )
                )

    traces = []
    for (resolved_agent_id, resolved_agent_version), group in groups.items():
        group["events"].sort(key=lambda e: e.get("timestamp") or "")
        traces.append({
            "trace_id": group["trace_id"],
            "agent_id": resolved_agent_id,
            "agent_version": resolved_agent_version,
            "events": group["events"],
            "source": SOURCE_OTLP_JSON,
        })
    return traces


def otlp_json_to_trace(
    payload: dict[str, Any],
    *,
    agent_id: str | None = None,
    agent_version: str | None = None,
) -> dict[str, Any]:
    """Single-agent convenience wrapper over `otlp_json_to_traces`. Returns
    the empty trace shape (`trace_id`/`agent_id`/`agent_version` all `""`,
    no events) for a batch with no spans. Raises `MultipleAgentsInBatchError`
    if the batch resolves to more than one distinct agent identity — call
    `otlp_json_to_traces` instead for that case rather than have this
    function guess which one the caller meant."""
    traces = otlp_json_to_traces(payload, agent_id=agent_id, agent_version=agent_version)
    if not traces:
        return {"trace_id": "", "agent_id": "", "agent_version": "", "events": []}
    if len(traces) > 1:
        identities = [(t["agent_id"], t["agent_version"]) for t in traces]
        raise MultipleAgentsInBatchError(
            f"batch resolves to {len(traces)} distinct agent identities {identities} — "
            "use otlp_json_to_traces to get one trace per agent instead"
        )
    return traces[0]


def otlp_json_bytes_to_traces(
    data: bytes | str,
    *,
    agent_id: str | None = None,
    agent_version: str | None = None,
) -> list[dict[str, Any]]:
    """Same as `otlp_json_to_traces`, from the raw request body an OTLP/HTTP
    JSON exporter POSTs (`Content-Type: application/json`)."""
    return otlp_json_to_traces(json.loads(data), agent_id=agent_id, agent_version=agent_version)


def otlp_json_bytes_to_trace(
    data: bytes | str,
    *,
    agent_id: str | None = None,
    agent_version: str | None = None,
) -> dict[str, Any]:
    """Same as `otlp_json_to_trace`, from the raw request body an OTLP/HTTP
    JSON exporter POSTs (`Content-Type: application/json`)."""
    return otlp_json_to_trace(json.loads(data), agent_id=agent_id, agent_version=agent_version)


def _otlp_proto_bytes_to_json_dict(data: bytes) -> dict[str, Any]:
    """Decode an OTLP/gRPC+protobuf `ExportTraceServiceRequest` payload into
    the SAME dict shape `otlp_json_to_traces` already consumes, so every
    span-to-event mapping rule lives in exactly one place (`_span_to_events`/
    `_classify`/the MCP cases above) rather than being re-implemented for
    protobuf. This works because OTLP/HTTP+JSON is *defined* as the
    protobuf JSON mapping (github.com/open-telemetry/opentelemetry-proto,
    `docs/specification.md` "OTLP/HTTP") — `MessageToDict` with
    `preserving_proto_field_name=False` (the default) produces the exact
    same camelCase field names and base64-encoded `bytes` fields
    (`traceId`, `spanId`) that a real OTLP/HTTP+JSON exporter sends, so the
    decoded proto and the JSON export of the same data hit the identical
    code path from here on.

    Deferred import: `opentelemetry-proto` (and the `protobuf` package it
    pulls in) is the optional `otel-proto` extra, not a base or `runner`
    dependency — importing it at module load time would force every
    customer to install protobuf just to use the JSON path this module
    defaults to. Raises `ImportError` with an actionable message if the
    extra isn't installed, rather than a bare `ModuleNotFoundError` deep in
    an import chain the caller didn't write.
    """
    try:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
        from google.protobuf.json_format import MessageToDict
    except ImportError as exc:  # pragma: no cover - exercised via the skip guard in tests
        raise ImportError(
            "otlp_proto_bytes_to_traces requires the optional 'otel-proto' extra "
            "(pip install 'coriqo-agents[otel-proto]') — the JSON path "
            "(otlp_json_to_traces / otlp_json_bytes_to_traces) needs no extra "
            "dependency and is the default for OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/json."
        ) from exc

    request = ExportTraceServiceRequest()
    request.ParseFromString(data)
    return MessageToDict(request, preserving_proto_field_name=False)


def otlp_proto_bytes_to_traces(
    data: bytes,
    *,
    agent_id: str | None = None,
    agent_version: str | None = None,
) -> list[dict[str, Any]]:
    """Same as `otlp_json_to_traces`, from the raw protobuf-encoded body an
    OTLP/gRPC or OTLP/HTTP+protobuf exporter sends
    (`Content-Type: application/x-protobuf`). Decodes to the identical JSON
    dict shape (`_otlp_proto_bytes_to_json_dict`) and then hands off to
    `otlp_json_to_traces` — no span-to-event mapping logic is duplicated
    here. Requires the optional `otel-proto` extra; see that helper's
    docstring."""
    return otlp_json_to_traces(
        _otlp_proto_bytes_to_json_dict(data), agent_id=agent_id, agent_version=agent_version
    )


def otlp_proto_bytes_to_trace(
    data: bytes,
    *,
    agent_id: str | None = None,
    agent_version: str | None = None,
) -> dict[str, Any]:
    """Same as `otlp_json_to_trace`, from a protobuf-encoded export body.
    Requires the optional `otel-proto` extra; see
    `_otlp_proto_bytes_to_json_dict`'s docstring."""
    return otlp_json_to_trace(
        _otlp_proto_bytes_to_json_dict(data), agent_id=agent_id, agent_version=agent_version
    )
