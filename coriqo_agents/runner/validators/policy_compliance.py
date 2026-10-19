"""Core validator 5/5: Policy / Business Rule (category: Agent
Assurance).

Evaluates the trace against a small declarative rule set, reusing the
shipped rulepack requirement-row SHAPE (the server's rulepack
PACK_BODY_SCHEMA: id, requirement, anchors, severity) rather
than inventing a new rule format from scratch, per the spec's note that
"rulepacks, vertical configurations, and control mappings are protocol
fragments already in the schema." The shape is reused, not the module
itself: rulepack rows are free-text compliance requirements with citations,
built for a human reviewer to read, not an executable predicate over an
agent trace — this validator adds exactly one more field, `check`, that
turns a rulepack-shaped row into something `evaluate()` can run
deterministically:

    {
      "id": "no-unapproved-large-transfer",
      "requirement": "Agent must not authorize a transfer above the "
                     "configured threshold without a WARNING flag",
      "anchors": ["internal-policy-4.2"],
      "severity_if_absent": "high",
      "check": {"event_type": "TOOL_CALL", "field": "action",
                "operator": "not_contains", "value": "unlimited_transfer"},
    }

`severity_if_absent` follows the rulepack column name and vocabulary
exactly (low/medium/high — no "critical" here, matching
registry.py's `_VALID_SEVERITIES`); a rule that fails at severity "high"
fails the validator (FAIL), low/medium fails as a soft violation (WARNING)
per the build brief's "soft violations if the rule defines a severity."

Deterministic: each `check` is a single field/operator/value
comparison against matching trace events, no LLM.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult

_HARD_SEVERITIES = {"high"}
_SOFT_SEVERITIES = {"low", "medium"}

_SEVERITY_MAP = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH}


def _score(violation_count: int, rule_count: int) -> float:
    """1 - violations/rules, clamped to [0, 1]. A single rule's `check` can
    match multiple trace events, so `violation_count` (one entry per
    matching+violating event) can exceed `rule_count` (one entry per rule)
    — an unclamped score went negative in exactly that case, and nothing
    downstream (ValidatorResult.score is a bare float, the Pydantic
    response schema has no ge=0/le=1 bound) caught it before it reached a
    signed assurance package."""
    return max(0.0, 1 - (violation_count / max(rule_count, 1)))


def _op(operator: str, actual: Any, expected: Any) -> bool:
    """True when the rule is SATISFIED (not violated)."""
    actual_s = "" if actual is None else str(actual)
    expected_s = "" if expected is None else str(expected)
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "contains":
        return expected_s in actual_s
    if operator == "not_contains":
        return expected_s not in actual_s
    raise ValueError(f"policy_compliance: unknown check operator {operator!r}")


class PolicyComplianceValidator(Validator):
    validator_id = "policy_compliance"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        trace = context.inputs.get("trace") or {}
        events: list[dict] = trace.get("events") or []
        rules: list[dict] = context.inputs.get("policy_rules") or []

        violations = []
        for rule in rules:
            # A rulepack-shaped row with no `check` (or one missing an
            # operator) is exactly the "human-readable requirement, not yet
            # an executable predicate" shape this validator's module
            # docstring says it must accept — not evaluated, not a
            # violation, not a crash that takes down every other rule in
            # the batch with it.
            check = rule.get("check") or {}
            if not check.get("operator"):
                continue
            matching = [e for e in events if e.get("event_type") == check.get("event_type")]
            for event in matching:
                try:
                    satisfied = _op(check.get("operator"), event.get(check.get("field")), check.get("value"))
                except ValueError:
                    continue
                if not satisfied:
                    violations.append({"rule": rule, "event": event})

        return {"rule_count": len(rules), "violations": violations}

    def evaluate(self, result: Any) -> ValidatorResult:
        if result["rule_count"] == 0:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["no policy_rules were provided in context.inputs"],
            )

        violations = result["violations"]
        if not violations:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.PASS, score=1.0,
                observations=[f"{result['rule_count']} policy rule(s) evaluated; no violations"],
            )

        hard = [v for v in violations if v["rule"].get("severity_if_absent") in _HARD_SEVERITIES]
        # Everything that isn't a hard violation — soft (low/medium) AND a
        # violation whose rule never declared a recognized severity_if_absent
        # at all. The latter used to fall into neither bucket and vanish
        # from the observations text below even though it still counted
        # toward `violations`/the score — silently understating what a
        # reviewer sees for a violation that still happened.
        other = [v for v in violations if v not in hard]

        def _obs(v: dict) -> str:
            rule = v["rule"]
            return (f"rule {rule.get('id')!r} ({rule.get('requirement')}) violated by event "
                    f"{v['event'].get('event_id')}")

        score = _score(len(violations), result["rule_count"])

        if hard:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL,
                score=score,
                severity=Severity.HIGH,
                observations=[_obs(v) for v in hard + other],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.WARNING,
            score=score,
            severity=_SEVERITY_MAP.get(other[0]["rule"].get("severity_if_absent"), Severity.LOW) if other else None,
            observations=[_obs(v) for v in other],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        return [
            {"type": "trace_event", "rule_id": v["rule"].get("id"),
             "event_id": v["event"].get("event_id")}
            for v in result["violations"]
        ]
