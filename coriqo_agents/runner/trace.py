"""The execution trace shape the validators evaluate: the OTLP-shaped agent
event model, as a plain-dict schema a validator can
read without any server-side import (this package ships standalone to
customer environments, same constraint documented in manifest.py).

An example event:

    {
      "event_id": "evt_9812",
      "trace_id": "trace_123",
      "agent_id": "agent_44",
      "agent_version": "2.4.1",
      "event_type": "TOOL_CALL",
      "tool": "loan_system",
      "action": "update_application",
      "actor": "agent",
      "authorization": "APPROVED",
      "timestamp": "...",
      "payload_hash": "sha256:..."
    }

One field is added beyond the wire shape: `payload` — the actual tool
args/result content a validator inspects locally (for PII detection, prompt-
injection markers, sequence checks). `payload_hash` is what would actually
leave the customer environment in an uploaded manifest/evidence record;
`payload` never does. Validators read `payload`, never invent a second
source of truth for it.

A trace is just `{"trace_id", "agent_id", "agent_version", "events": [...]}`
— no class needed on the validator side, since ValidatorContext.inputs is
already a free-form dict bag (contract.py) and every consumer here only
ever needs read access to plain dicts/lists.
"""
from __future__ import annotations

from typing import Any


class EventType:
    """The agent lifecycle stages, as a plain namespace (same pattern as
    contract.py's Status/Severity) — the wire format carries plain strings."""
    USER_REQUEST = "USER_REQUEST"
    LLM_CALL = "LLM_CALL"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    EXTERNAL_SIDE_EFFECT = "EXTERNAL_SIDE_EFFECT"
    FINAL_RESPONSE = "FINAL_RESPONSE"

    # --- kill-switch and containment events (consumed by the
    # six ks_* validators and the memory/override validators).
    #
    # Additive only. Existing validators filter by the event types they know
    # (events_of), so a trace carrying these is read exactly as before by the
    # five shipped validators — a new type can add a check, never weaken one.
    #
    # The kill-switch sequence a ks_* validator reads off a trace is:
    # TRIGGER_EVENT (the abnormal condition) → SHUTDOWN_SIGNAL (the switch is
    # pulled) → TASK_REJECTED / EXECUTION_HALTED / CREDENTIAL_REVOKED (it
    # took effect) → EXECUTION_RESUMED (recovery). Timing comes from the
    # events' own timestamps, so "how long between signal and halt" is
    # measured from the trace rather than asserted by the operator.
    TASK_REQUEST = "TASK_REQUEST"
    TASK_REJECTED = "TASK_REJECTED"
    SHUTDOWN_SIGNAL = "SHUTDOWN_SIGNAL"
    CREDENTIAL_REVOKED = "CREDENTIAL_REVOKED"
    MEMORY_READ = "MEMORY_READ"
    MEMORY_WRITE = "MEMORY_WRITE"
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"
    EXECUTION_HALTED = "EXECUTION_HALTED"
    # The abnormal condition that should provoke a shutdown — recorded
    # separately from SHUTDOWN_SIGNAL so a validator can tell "the trigger
    # fired and nothing happened" from "no trigger ever fired", which are
    # very different findings.
    TRIGGER_EVENT = "TRIGGER_EVENT"
    EXECUTION_RESUMED = "EXECUTION_RESUMED"

    # --- MCP events. Additive, same posture as the kill-switch
    # block above: existing validators filter by the event types they
    # already know, so a trace carrying these two new types is read exactly
    # as before by every validator that predates them.
    #
    # MCP_SERVER_CONNECT: `{server_id, url_hash, tools: [{name,
    # description_hash}]}` on `payload` — one per MCP server the agent
    # connects to in the run. `url_hash`/`description_hash` are what a
    # runner would put in an uploaded manifest (never the raw URL or
    # description text leaving the customer environment as identifying
    # material); the local `payload` additionally carries the raw
    # `description` text so `mcp_tool_description_injection` has something
    # to pattern-match against, same convention as `payload_text` already
    # documents for tool-result content.
    #
    # MCP_TOOL_LIST: `{server_id, tools: [{name, description_hash}]}` — a
    # server (re-)advertising its tool roster, e.g. after a mid-run
    # reconnect. `mcp_tool_shadowing` reads every MCP_SERVER_CONNECT and
    # MCP_TOOL_LIST event to build the name -> [server_id] map it checks
    # for collisions.
    #
    # A TOOL_CALL that targets an MCP-routed tool carries `payload.server_id`
    # naming which MCP server the call was routed to — the field the four
    # mcp_* validators below all key off. This is additive to TOOL_CALL's
    # existing shape (the example event has no `server_id`), so a
    # non-MCP TOOL_CALL is unaffected and simply has no `server_id`.
    MCP_SERVER_CONNECT = "MCP_SERVER_CONNECT"
    MCP_TOOL_LIST = "MCP_TOOL_LIST"

    # --- Multi-agent chain. Additive, same posture as the two blocks
    # above.
    #
    # AGENT_MESSAGE: one inter-agent hop, recorded in the trace of the
    # sender, the recipient, or both. Top-level fields (same convention as
    # the `on_behalf_of` identity field, not nested under `payload`):
    #
    #   {"event_type": "AGENT_MESSAGE", "message_id": "msg_1",
    #    "from": "agent_a", "to": "agent_b",
    #    "payload_hash": "sha256:...", "delegated_scope": ["loan_system"],
    #    "timestamp": "..."}
    #
    # `from`/`to` are agent ids matching the assured system's declared
    # roster. `message_id` is what lets the same hop recorded by both
    # endpoints be counted once. `payload_hash` is the hash of what crossed
    # — the raw payload never leaves the customer environment, same rule
    # this module's docstring already states. `delegated_scope` is the list
    # of tool names the sender is handing on for this hop.
    #
    # A hop that records no `delegated_scope` or no `payload_hash` is not a
    # violation on its own; the chain_* validators report it as an
    # INCONCLUSIVE link with a reason, never as a PASS.
    AGENT_MESSAGE = "AGENT_MESSAGE"

    ALL = (
        USER_REQUEST, LLM_CALL, TOOL_CALL, TOOL_RESULT, HUMAN_APPROVAL,
        EXTERNAL_SIDE_EFFECT, FINAL_RESPONSE,
        TASK_REQUEST, TASK_REJECTED, SHUTDOWN_SIGNAL, CREDENTIAL_REVOKED,
        MEMORY_READ, MEMORY_WRITE, HUMAN_OVERRIDE, EXECUTION_HALTED,
        TRIGGER_EVENT, EXECUTION_RESUMED,
        MCP_SERVER_CONNECT, MCP_TOOL_LIST,
        AGENT_MESSAGE,
    )


# --- Identity fields. Additive only, same posture as the MCP fields
# above: existing validators filter by the fields/types they already know,
# so a trace carrying these is read exactly as before by every validator
# that predates them.
#
# `credential_id` (TOOL_CALL, optional, top-level): the identity provider's
# id for the credential the call used. `identity_credential_ttl` groups
# TOOL_CALL events by this field and measures how long a single id spans
# the trace, compared against the agent's asserted `credential_ttl_seconds`.
# A TOOL_CALL with no `credential_id` is simply not part of any group — not
# a violation on its own; a validator that wanted to demand one on every
# call would be a policy decision, which belongs in a bank's own IdP/CASB,
# not in this SDK per design principle #2.
#
# `on_behalf_of` (TOOL_CALL, optional, top-level): the delegation chain of
# principal refs behind this call, ordered from the immediate actor to the
# ultimate requester, e.g. ["agent:loan-bot", "user:jsmith"]. A human
# principal ref is recognised by a configurable prefix (default "user:",
# `context.configuration["human_principal_prefix"]`).
# `identity_delegation_chain` FAILs a TOOL_CALL whose `on_behalf_of` is
# missing, empty, or does not terminate in that prefix.
#
# `principal_ref` (any event, optional, top-level): which principal (per
# the bank's IdP) emitted this event — distinct from `actor`
# ("agent"/"human"/"system"), which says WHAT kind of party acted, not
# WHICH one. `identity_zombie` compares every event's `principal_ref`
# against the retired agent's own asserted `principal_ref`
# (`context.configuration["principal_ref"]`) and the retirement timestamp
# (`context.configuration["retired_at"]`) to check for events emitted by
# that principal after retirement.


class Authorization:
    """The `authorization` field values on a TOOL_CALL/HUMAN_APPROVAL
    event — APPROVED is the only value action_integrity.py treats as
    satisfying a gate; anything else (DENIED, PENDING, absent) does not."""
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    PENDING = "PENDING"


def events_of(trace: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    """Every event of `event_type`, in trace order. `trace` is the
    {"trace_id", "agent_id", "agent_version", "events": [...]} shape above;
    a trace with no "events" key returns []."""
    return [e for e in (trace.get("events") or []) if e.get("event_type") == event_type]


def mcp_tool_listings(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Every `{"event", "index", "tool"}` triple off MCP_SERVER_CONNECT/
    MCP_TOOL_LIST events' `payload.tools[]`, in trace order — the shape both
    `mcp_tool_description_injection` and `mcp_tool_shadowing` need to walk
    "every tool any MCP server advertised, and which event/position it came
    from." Shared here rather than each validator keeping
    its own copy, since the two independently drifted on one detail already
    (whether a listing with no `server_id` should be skipped) before this
    existed — one traversal means they can't silently diverge on which
    listings they each consider again."""
    out: list[dict[str, Any]] = []
    for i, event in enumerate(trace.get("events") or []):
        if event.get("event_type") not in (EventType.MCP_SERVER_CONNECT, EventType.MCP_TOOL_LIST):
            continue
        for tool in (event.get("payload") or {}).get("tools") or []:
            if isinstance(tool, dict):
                out.append({"event": event, "index": i, "tool": tool})
    return out


def payload_text(event: dict[str, Any]) -> str:
    """Best-effort flattening of an event's `payload` to a string for
    pattern-matching (PII detection, injection-marker scanning). `payload`
    may be a string (tool result text), a dict (args), or absent."""
    payload = event.get("payload")
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        import json
        return json.dumps(payload, sort_keys=True, default=str)
    except TypeError:
        return str(payload)
