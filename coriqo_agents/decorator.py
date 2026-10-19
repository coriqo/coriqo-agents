"""Decorator for recording a plain Python function as one decision per call.

The simplest way to wire an agent that isn't built on LangChain: wrap the
function that makes the decision and every call is recorded automatically.
Captures the function's arguments as `inputs` and its return value as
`output` — nothing else. For tool_calls/grounding_refs, call
CoriqoAgentClient.record_trace directly instead (or from inside the
decorated function, alongside its own logic).
"""
from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, TypeVar

from .buffer import buffer_trace
from .client import AgentSuspendedError, CoriqoAgentClient

F = TypeVar("F", bound=Callable[..., Any])


def _bind_inputs(fn: Callable, args: tuple, kwargs: dict) -> dict:
    """Binds positional/keyword call args to the function's parameter names
    (e.g. {"applicant_id": "a-1"}), so `inputs` reads as named fields rather
    than an opaque positional tuple. Falls back to a raw args/kwargs dict if
    the signature can't be bound (e.g. a bare *args function)."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except TypeError:
        return {"args": args, "kwargs": kwargs}


def track_decision(
    client: CoriqoAgentClient,
    agent_id: str,
    *,
    cost_usd: float | None = None,
    token_count: int | None = None,
    subject_ref: str | Callable[[dict], str | None] | None = None,
    subject_type: str | None = None,
    runtime: dict | None = None,
) -> Callable[[F], F]:
    """`@track_decision(client, agent_id)` on a function that makes one
    agent decision: records its arguments as `inputs` and its return value
    as `output` on every call. A raised exception is recorded too (as the
    output, so the decision still gets a sealed trace) and then re-raised —
    this never swallows the original error. Works on both sync and async
    functions.

    Measures wall-clock time around the wrapped call and passes it through
    as `latency_ms` on every recorded trace — no code inside the wrapped
    function needs to time itself. `cost_usd`/`token_count` are optional and
    passed through unchanged on every call (Coriqo has no way to measure
    what a call actually cost or how many tokens it used — set these if
    your own accounting knows the per-call figure, e.g. a fixed-cost tool;
    for per-call-varying cost, call client.record_trace directly instead).

    `subject_ref` marks who the decision was about. Since that usually comes
    from the call itself, pass a function of the bound inputs —
    `subject_ref=lambda i: i["applicant_id"]` — or a plain string if every
    call concerns the same subject. Either way the value is hashed locally by
    record_trace before the request goes out, so the raw identifier stays in
    your process. `subject_type` and `runtime` are constants sent unchanged on
    every call; they are stored as plain text, so don't put identifiers in
    them.

    A Coriqo outage never raises into your code from here: if record_trace()
    itself fails (network error, 5xx, timeout — anything but a deliberate
    AgentSuspendedError), the failure is logged and the trace payload is
    appended to a local buffer file instead (see `coriqo_agents.buffer`) so it
    isn't silently dropped. AgentSuspendedError is a governance signal, not an
    outage, and still propagates so a caller checking for it can actually
    halt."""

    def _finish(inputs: dict, start: float, *, result: Any = None, exc: Exception | None = None) -> None:
        latency_ms = int((time.monotonic() - start) * 1000)
        output = f"Raised {type(exc).__name__}: {exc}" if exc is not None else str(result)
        ref = subject_ref(inputs) if callable(subject_ref) else subject_ref
        record_kwargs = dict(
            inputs=inputs, output=output,
            latency_ms=latency_ms, cost_usd=cost_usd, token_count=token_count,
            subject_ref=ref, subject_type=subject_type, runtime=runtime,
        )
        try:
            client.record_trace(agent_id, **record_kwargs)
        except AgentSuspendedError:
            # A deliberate governance decision, not an outage — the caller
            # may need to actually stop, so let it through. If the wrapped
            # function already raised, keep that exception as the one the
            # caller sees; the suspension still lands on the next call.
            if exc is not None:
                return
            raise
        except Exception as record_exc:  # noqa: BLE001 — outage must never break the caller's own code
            buffer_trace(agent_id, record_kwargs, record_exc)

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                inputs = _bind_inputs(fn, args, kwargs)
                start = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                except Exception as e:
                    _finish(inputs, start, exc=e)
                    raise
                _finish(inputs, start, result=result)
                return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            inputs = _bind_inputs(fn, args, kwargs)
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                _finish(inputs, start, exc=e)
                raise
            _finish(inputs, start, result=result)
            return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator
