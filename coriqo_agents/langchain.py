"""Optional LangChain integration: one sealed decision trace per top-level run.

Importing `coriqo_agents` never requires langchain-core; this submodule does,
and only raises if you actually import it. Install with:

    pip install coriqo-agents[langchain]

Pass the handler in a run's config:

    handler = CoriqoCallbackHandler(client, agent_id="credit-limit-agent")
    agent.invoke({"input": ...}, config={"callbacks": [handler]})

What one top-level run records:

- every tool call, paired with its result by LangChain's `run_id`, so parallel
  tool calls cannot swap results; a tool that raised is recorded with its error
  instead of being dropped. Args and results are hashed locally before the
  request goes out, as with `record_trace`;
- the model that answered as the `runtime` ({"provider": ..., "model": ...}),
  read from the chat model's serialized config and `llm_output`, so a swapped
  model seals a runtime_changed event;
- tokens summed across every model call in the run, and wall-clock latency;
- the run's final output, or "Chain run failed: ..." when it raised. A failed
  run is still recorded: it is still a decision.

Nested runs (an AgentExecutor's planning sub-chains, a RunnableSequence's
steps) are folded into their top-level run by following `parent_run_id`, so one
`invoke` is one decision. One handler can serve concurrent invocations.

For precise control, for example setting `grounding_refs` (LangChain has no
notion of which sealed Coriqo record a decision relied on), call
`CoriqoAgentClient.record_trace` directly instead.

Verified against langchain-core 1.6.12 and 0.3.49 callback signatures.
"""
from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from .buffer import buffer_trace
from .client import AgentSuspendedError, CoriqoAgentClient

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover: exercised only without langchain-core installed
    BaseCallbackHandler = object
    _LANGCHAIN_AVAILABLE = False


def _model_from(serialized: dict | None, kwargs: dict) -> tuple[str | None, str | None]:
    """(provider, model) from a chat model's start event, as far as it says."""
    params = kwargs.get("invocation_params") or {}
    meta = kwargs.get("metadata") or {}
    model = params.get("model") or params.get("model_name") or meta.get("ls_model_name")
    provider = meta.get("ls_provider") or params.get("_type")
    if not model and serialized:
        kw = serialized.get("kwargs") or {}
        model = kw.get("model") or kw.get("model_name")
    return provider, model


def _usage_tokens(response: Any) -> int:
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    tokens = 0
    # Newer chat models carry usage on each message instead of llm_output.
    for gens in getattr(response, "generations", None) or []:
        for gen in gens:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
            if isinstance(meta.get("total_tokens"), int):
                tokens += meta["total_tokens"]
    return tokens


class _Run:
    __slots__ = ("started", "tool_calls", "pending", "provider", "model", "tokens")

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.tool_calls: list[dict] = []
        self.pending: dict[UUID, dict] = {}
        self.provider: str | None = None
        self.model: str | None = None
        self.tokens = 0


class CoriqoCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    # LangChain logs and swallows a callback's exception unless the handler opts
    # in, which would turn the kill switch into a log line. Every other failure
    # is caught below, so the only thing this lets through is AgentSuspendedError.
    raise_error = True

    def __init__(self, client: CoriqoAgentClient, agent_id: str, *, subject_type: str | None = None):
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "CoriqoCallbackHandler requires langchain-core: pip install langchain-core"
            )
        super().__init__()
        self._client = client
        self._agent_id = agent_id
        self._subject_type = subject_type
        self._runs: dict[UUID, _Run] = {}
        self._root: dict[UUID, UUID] = {}  # any run id -> its top-level run id
        self._lock = threading.Lock()

    # Every event finds its top-level run by walking parent_run_id once.
    def _enter(self, run_id: UUID, parent_run_id: UUID | None) -> _Run:
        with self._lock:
            root = self._root.get(parent_run_id, parent_run_id) if parent_run_id else run_id
            self._root[run_id] = root
            return self._runs.setdefault(root, _Run())

    def _find(self, run_id: UUID) -> _Run | None:
        with self._lock:
            root = self._root.get(run_id)
            return self._runs.get(root) if root else None

    def on_chain_start(self, serialized: dict, inputs: Any, *, run_id: UUID,
                       parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._enter(run_id, parent_run_id)

    def on_chat_model_start(self, serialized: dict, messages: Any, *, run_id: UUID,
                            parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        run = self._enter(run_id, parent_run_id)
        provider, model = _model_from(serialized, kwargs)
        run.provider, run.model = provider or run.provider, model or run.model

    def on_llm_start(self, serialized: dict, prompts: Any, *, run_id: UUID,
                     parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self.on_chat_model_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._find(run_id)
        if run is None:
            return
        run.tokens += _usage_tokens(response)
        name = (getattr(response, "llm_output", None) or {}).get("model_name")
        if name:
            run.model = name

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID,
                      parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        run = self._enter(run_id, parent_run_id)
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown_tool"
        run.pending[run_id] = {"tool": name, "args": kwargs.get("inputs") or input_str}

    def _tool_done(self, run_id: UUID, **fields: Any) -> None:
        run = self._find(run_id)
        if run is None:
            return
        call = run.pending.pop(run_id, {"tool": "unknown_tool", "args": None})
        run.tool_calls.append({"step": len(run.tool_calls) + 1, **call, **fields})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_done(run_id, result=output)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_done(run_id, result=None, error=str(error))

    def on_chain_end(self, outputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        if parent_run_id is None:
            self._record(run_id, output=str(outputs))

    def on_chain_error(self, error: BaseException, *, run_id: UUID,
                       parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        # An inner sub-chain's error isn't the agent's decision failing.
        if parent_run_id is None:
            self._record(run_id, output=f"Chain run failed: {error}")

    def _record(self, run_id: UUID, *, output: str) -> None:
        with self._lock:
            run = self._runs.pop(run_id, None)
            # Drop every child mapping of this run, so a reused handler holds nothing.
            for child in [k for k, v in self._root.items() if v == run_id]:
                del self._root[child]
        if run is None:
            return
        payload = {
            "output": output,
            "tool_calls": run.tool_calls or None,
            "runtime": {"provider": run.provider, "model": run.model, "framework": "langchain"} if run.model else None,
            "token_count": run.tokens or None,
            "latency_ms": int((time.monotonic() - run.started) * 1000),
            "subject_type": self._subject_type,
        }
        try:
            self._client.record_trace(self._agent_id, **payload)
        except AgentSuspendedError:
            raise  # the kill switch must stop the agent, not be logged and ignored
        except Exception as exc:  # an outage must not fail the run: keep the trace locally, as @track_decision does
            buffer_trace(self._agent_id, payload, exc)
