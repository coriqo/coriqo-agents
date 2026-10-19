"""Identity validator 3/3: Zombie Identity ("automatic deprovisioning").

FAILs when the agent's principal (`context.configuration["principal_ref"]`)
still emits events in this trace AFTER the agent's own recorded
`retired_at` (`context.configuration["retired_at"]`) — a governed agent
Coriqo believes is retired but whose identity is still active in the
customer's own systems.

An event is attributed to the agent's principal via the trace event's own
`principal_ref` field (an identity field in trace.py), never inferred from
`actor` or any other field — an event with no `principal_ref` at all is not
counted as this principal, since attributing it would be a guess this
validator has no basis for (PROGRAM_TRACKER standing rule #8: an
unevaluated case must not collapse into either a clean or a violating one —
here it is disclosed as simply not matching, not silently assumed to
match).

NOT_APPLICABLE, not PASS, when the agent has no `retired_at` at all — this
validator only has something to assert about a retirement that has actually
happened; running it against a still-active agent and calling that a "pass"
would misrepresent what was checked.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .ks_common import parse_ts as _parse_ts


class IdentityZombieValidator(Validator):
    validator_id = "identity_zombie"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        trace = context.inputs.get("trace") or {}
        events: list[dict] = trace.get("events") or []
        principal_ref = context.configuration.get("principal_ref")
        retired_at = _parse_ts(context.configuration.get("retired_at"))

        post_retirement_events = []
        if principal_ref and retired_at is not None:
            for event in events:
                if event.get("principal_ref") != principal_ref:
                    continue
                ts = _parse_ts(event.get("timestamp"))
                if ts is not None and ts > retired_at:
                    post_retirement_events.append(event)

        return {
            "principal_ref": principal_ref,
            "retired_at": retired_at,
            "post_retirement_events": post_retirement_events,
        }

    def evaluate(self, result: Any) -> ValidatorResult:
        if not result["principal_ref"] or result["retired_at"] is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["agent has no recorded principal_ref and retired_at pair to check "
                              "post-retirement activity against"],
            )

        post = result["post_retirement_events"]
        if post:
            observations = [
                f"event {e.get('event_id')} at {e.get('timestamp')} was emitted by principal "
                f"{result['principal_ref']!r} after its recorded retirement at "
                f"{result['retired_at'].isoformat()}"
                for e in post
            ]
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL,
                score=0.0,
                severity=Severity.CRITICAL,
                observations=observations,
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[
                f"no event from principal {result['principal_ref']!r} after its recorded "
                f"retirement at {result['retired_at'].isoformat()}",
            ],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        return [
            {"type": "trace_event", "event_id": e.get("event_id"), "timestamp": e.get("timestamp"),
             "reason": "principal active after recorded retirement"}
            for e in result["post_retirement_events"]
        ]

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="identity_zombie",
            threat="A retired agent's identity remains active after Coriqo records it as deprovisioned",
            trigger="An event attributed to the agent's principal_ref is timestamped after its "
                    "recorded retired_at",
            expected_behavior="No event from a retired agent's principal after its retirement",
            forbidden_behavior="Agent's principal keeps emitting events after retirement",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Compare every event's principal_ref/timestamp against the agent's own "
                          "retired principal_ref and retired_at",
        )
