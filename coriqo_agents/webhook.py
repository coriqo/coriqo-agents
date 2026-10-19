"""Framework-agnostic webhook receiver helper.

For an agent platform that can only "call a webhook" (Zapier, Make, n8n, an
internal RPA/orchestration tool) rather than embed this SDK directly: stand
up ONE route in whatever web framework you already run, and have it call
`handle_webhook` with the parsed JSON body. This module ships no web server
of its own — the SDK stays free of any hard dependency on a particular web
framework.

FastAPI example:

    from fastapi import FastAPI, Request
    from coriqo_agents import CoriqoAgentClient
    from coriqo_agents.webhook import handle_webhook

    app = FastAPI()
    client = CoriqoAgentClient(base_url="...", api_key="...", tenant_slug="...")

    @app.post("/coriqo-webhook/{agent_id}")
    async def coriqo_webhook(agent_id: str, request: Request):
        return handle_webhook(client, agent_id, await request.json())

Flask example:

    @app.route("/coriqo-webhook/<agent_id>", methods=["POST"])
    def coriqo_webhook(agent_id):
        return handle_webhook(client, agent_id, request.get_json())
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .client import CoriqoAgentClient

_TRACE_FIELDS = (
    "inputs", "output", "tool_calls", "grounding_refs", "occurred_at",
    "subject_ref", "subject_type", "runtime",
)


def handle_webhook(client: CoriqoAgentClient, agent_id: str, payload: dict[str, Any]) -> dict:
    """Records one decision from a webhook payload already shaped like the
    /traces contract (inputs/output/tool_calls/grounding_refs/occurred_at/
    subject_ref/subject_type/runtime — any subset; other keys in the payload
    are ignored). A `subject_ref` in the payload is hashed by record_trace
    before the request leaves your route, so the raw identifier reaches your
    process from the platform but never reaches Coriqo. This is a thin
    pass-through, not a universal webhook-format translator: if your
    platform's payload looks different, reshape it to this contract first
    (or call CoriqoAgentClient.record_trace directly with whatever fields
    you already have)."""
    kwargs: dict[str, Any] = {
        k: payload[k] for k in _TRACE_FIELDS if payload.get(k) is not None
    }
    occurred_at = kwargs.get("occurred_at")
    if isinstance(occurred_at, str):
        # ISO 8601 string over the wire (JSON has no native datetime type).
        # record_trace still enforces timezone-awareness on the result —
        # an offset-less string here raises there, same as any other path.
        kwargs["occurred_at"] = datetime.fromisoformat(occurred_at)
    return client.record_trace(agent_id, **kwargs)
