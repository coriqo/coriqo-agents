"""Validator 4/5: is the agent going round in circles?

An agent that repeats the same tool call over and over burns budget, fills
logs, and is usually stuck rather than working. The spec calls for a WARNING,
never a FAIL: repetition is a design smell a human should look at, not a
control violation Coriqo is entitled to call. This validator therefore has no
FAIL branch at all — the same posture `chain_loop` takes.

## The signature

Two TOOL_CALL events are "the same call" when their signature matches:
`(tool, action, args_hash)`. `args_hash` comes from the event's own
`args_hash`/`payload_hash`, so two calls to the same tool with different
arguments are different work, not a loop. A call carrying no hash falls back
to `(tool, action)` — coarser, and disclosed as such in the observations,
because an emitter that records no argument hash cannot distinguish a retry
from a sweep and this validator must not pretend otherwise.

Repeats above `configuration["loop_repeat_threshold"]` (default 4) fire the
WARNING. Four identical calls is a retry policy; a fifth is a loop.

Determinism (D9): counting identical signatures.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..trace import EventType
from . import containment_common as cc


class LoopDetectionValidator(Validator):
    validator_id = "loop_detection"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        trace = cc.trace_of(context)
        raw = context.configuration.get("loop_repeat_threshold")
        try:
            threshold = int(raw) if raw is not None else cc.DEFAULT_LOOP_REPEAT_THRESHOLD
        except (TypeError, ValueError):
            threshold = cc.DEFAULT_LOOP_REPEAT_THRESHOLD
        calls = [e for e in cc.events(trace) if e.get("event_type") == EventType.TOOL_CALL]
        return {"trace": trace, "threshold": max(1, threshold), "calls": calls}

    def evaluate(self, result: Any) -> ValidatorResult:
        if result["trace"] is None:
            return cc.no_trace(self.validator_id, self.version)

        calls: list[dict[str, Any]] = result["calls"]
        if not calls:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["the trace records no tool call, so there is no repetition to count"],
            )

        threshold: int = result["threshold"]
        counts: dict[tuple[str, str, str], int] = {}
        coarse = 0
        for call in calls:
            signature, precise = _signature(call)
            counts[signature] = counts.get(signature, 0) + 1
            if not precise:
                coarse += 1

        repeated = sorted(
            ((sig, n) for sig, n in counts.items() if n > threshold),
            key=lambda pair: (-pair[1], pair[0]),
        )
        coarse_note = (
            [] if not coarse else
            [f"{coarse} of {len(calls)} tool call(s) record no argument hash, so their repeats "
             "are counted by tool and action alone — a sweep over different arguments would "
             "look the same as a retry"]
        )

        if repeated:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.WARNING,
                score=1 - (sum(n for _, n in repeated) / len(calls)),
                severity=Severity.MEDIUM,
                observations=[
                    f"{len(repeated)} tool-call signature(s) repeat more than {threshold} time(s) "
                    f"across {len(calls)} call(s)",
                    *[
                        f"{_describe(sig)} called {n} times, above the {threshold}-repeat threshold"
                        for sig, n in repeated
                    ],
                    *coarse_note,
                ],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[
                f"{len(calls)} tool call(s) across {len(counts)} distinct signature(s), none "
                f"repeating more than {threshold} time(s)",
                *coarse_note,
            ],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str, str], int] = {}
        for call in result["calls"]:
            signature, _ = _signature(call)
            counts[signature] = counts.get(signature, 0) + 1
        return [
            {"type": "tool_call_signature", "tool": tool, "action": action,
             "args_hash": args_hash or None, "count": n}
            for (tool, action, args_hash), n in sorted(counts.items())
        ]


def _signature(call: dict[str, Any]) -> tuple[tuple[str, str, str], bool]:
    """`((tool, action, args_hash), precise)` for one TOOL_CALL.

    `precise` is False when the event carries no argument hash — the caller
    discloses how many calls were counted that way rather than silently
    treating a coarse match as an exact one.
    """
    payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
    tool = str(call.get("tool") or cc.resource_of(call) or "")
    action = str(call.get("action") or "")
    for source in (call, payload):
        for key in ("args_hash", "payload_hash"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return (tool, action, value), True
    return (tool, action, ""), False


def _describe(signature: tuple[str, str, str]) -> str:
    tool, action, args_hash = signature
    name = f"{tool or '<unnamed tool>'}"
    if action:
        name += f".{action}"
    return name if not args_hash else f"{name} with the same arguments"
