"""Optional OpenAI Agents SDK integration: one sealed decision trace per run.

Importing `coriqo_agents` never requires the OpenAI Agents SDK; this submodule
does, and only raises if you actually import it. Install with:

    pip install coriqo-agents[openai-agents]

Register once, at start-up:

    from agents import add_trace_processor
    from coriqo_agents.openai_agents import CoriqoTracingProcessor

    add_trace_processor(CoriqoTracingProcessor(client, agent_id="credit-limit-agent"))

Every `Runner.run(...)` is one SDK trace, and becomes one Coriqo decision:

- each function-tool call becomes a tool call (name, args, result; args and
  result are hashed locally before anything leaves your process, as with
  `record_trace`), in the order the tools finished;
- a handoff becomes a tool call named `handoff:<to agent>`, and a guardrail
  that tripped becomes `guardrail:<name>` with result `"triggered"`, so a
  reviewer sees both on the record;
- the model that answered is the `runtime` ({"provider": "openai", "model": ...}),
  so a swapped model seals a runtime_changed event under an unchanged mandate;
- tokens are summed across the run, latency is trace start to trace end;
- the run's final text is the `output`. An error on a tool stays on that tool
  call (the SDK usually hands it back to the model and the run goes on); an
  error on any other span marks the run failed, and the trace is still
  recorded, as a failed decision is still a decision.

Several SDK agents can map to several Coriqo agents: pass `agent_ids={"Triage":
"triage-agent", ...}` and the run is recorded under the Coriqo agent mapped to
the first SDK agent that ran; unmapped runs fall back to `agent_id`.

Recording never raises into your agent: the SDK calls processors inline, and a
Coriqo outage must not fail a customer's loan decision. A failed send is logged
on the `coriqo_agents.openai_agents` logger and the trace is kept in the local
outage buffer (`coriqo_agents.buffer`), as with `@track_decision`. A suspended
agent is logged at error level; the trace ends after the run, so to stop a
suspended agent before it acts, call `client.get_agent()` or a deploy check first.

Verified against openai-agents 0.24.0 (TracingProcessor, FunctionSpanData,
GenerationSpanData, ResponseSpanData, HandoffSpanData, GuardrailSpanData).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

from .buffer import buffer_trace
from .client import AgentSuspendedError, CoriqoAgentClient

try:
    from agents.tracing import TracingProcessor
    _AGENTS_AVAILABLE = True
except ImportError:  # pragma: no cover: exercised only without openai-agents installed
    TracingProcessor = object
    _AGENTS_AVAILABLE = False

log = logging.getLogger(__name__)


def _final_text(data: Any) -> str | None:
    """The text a response or generation span ended on, or None."""
    response = getattr(data, "response", None)
    if response is not None:
        text = getattr(response, "output_text", None)
        if text:
            return str(text)
    output = getattr(data, "output", None)
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, str) and content:
                return content
    return None


def _tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    return sum(v for k, v in usage.items() if k in ("input_tokens", "output_tokens") and isinstance(v, int))


class _Run:
    __slots__ = ("started_at", "sdk_agent", "tool_calls", "model", "tokens", "output", "error")

    def __init__(self) -> None:
        self.started_at: datetime | None = None
        self.sdk_agent: str | None = None
        self.tool_calls: list[dict] = []
        self.model: str | None = None
        self.tokens = 0
        self.output: str | None = None
        self.error: str | None = None


class CoriqoTracingProcessor(TracingProcessor):  # type: ignore[misc]
    def __init__(
        self,
        client: CoriqoAgentClient,
        agent_id: str,
        *,
        agent_ids: dict[str, str] | None = None,
        subject_type: str | None = None,
    ):
        if not _AGENTS_AVAILABLE:
            raise ImportError(
                "CoriqoTracingProcessor requires the OpenAI Agents SDK: pip install openai-agents"
            )
        self._client = client
        self._agent_id = agent_id
        self._agent_ids = dict(agent_ids or {})
        self._subject_type = subject_type
        self._runs: dict[str, _Run] = {}
        # The SDK may run several traces at once on different threads or tasks.
        self._lock = threading.Lock()

    def _run(self, trace_id: str) -> _Run:
        with self._lock:
            return self._runs.setdefault(trace_id, _Run())

    def on_trace_start(self, trace: Any) -> None:
        self._run(trace.trace_id).started_at = datetime.now().astimezone()

    def on_span_start(self, span: Any) -> None:
        # An agent span ends after everything inside it, so the first agent is known only at start.
        if getattr(span.span_data, "type", None) == "agent":
            run = self._run(span.trace_id)
            if run.sdk_agent is None:
                run.sdk_agent = getattr(span.span_data, "name", None)

    def on_span_end(self, span: Any) -> None:
        run = self._run(span.trace_id)
        data = span.span_data
        kind = getattr(data, "type", None)
        message = str(span.error.get("message") or "error") if span.error else None
        if kind == "function":
            # A tool error is usually handed back to the model and the run goes on: it belongs on the call.
            call = {"step": len(run.tool_calls) + 1, "tool": data.name, "args": data.input, "result": data.output}
            if message:
                call["error"] = message
            run.tool_calls.append(call)
            return
        if message:
            run.error = message
        if kind == "handoff":
            run.tool_calls.append({"step": len(run.tool_calls) + 1, "tool": f"handoff:{data.to_agent or 'unknown'}",
                                   "args": data.from_agent, "result": data.to_agent})
        elif kind == "guardrail" and getattr(data, "triggered", False):
            run.tool_calls.append({"step": len(run.tool_calls) + 1, "tool": f"guardrail:{data.name}",
                                   "args": None, "result": "triggered"})
        elif kind in ("generation", "response"):
            model = getattr(data, "model", None) or getattr(getattr(data, "response", None), "model", None)
            if model:
                run.model = str(model)
            run.tokens += _tokens(getattr(data, "usage", None))
            text = _final_text(data)
            if text:
                run.output = text

    def on_trace_end(self, trace: Any) -> None:
        with self._lock:
            run = self._runs.pop(trace.trace_id, None)
        if run is None:
            return
        ended = datetime.now().astimezone()
        agent_id = self._agent_ids.get(run.sdk_agent or "", self._agent_id)
        output = f"Run failed: {run.error}" if run.error else run.output
        payload = {
            "output": output,
            "tool_calls": run.tool_calls or None,
            "runtime": {"provider": "openai", "model": run.model, "framework": "openai-agents"} if run.model else None,
            "token_count": run.tokens or None,
            "latency_ms": int((ended - run.started_at).total_seconds() * 1000) if run.started_at else None,
            "occurred_at": run.started_at,
            "subject_type": self._subject_type,
        }
        try:
            self._client.record_trace(agent_id, **payload)
        except AgentSuspendedError as exc:
            # The run has already finished by the time its trace ends, so there is nothing left to stop.
            # Say so loudly: the next run should be blocked by the caller's own deploy or mandate check.
            log.error("Coriqo: agent %s is suspended; Coriqo refused the trace for run %s: %s", agent_id, trace.trace_id, exc)
        except Exception as exc:  # an outage must not fail the run: keep the trace locally, as @track_decision does
            buffer_trace(agent_id, payload, exc)

    def shutdown(self) -> None:
        with self._lock:
            self._runs.clear()

    def force_flush(self) -> None:
        pass
