"""Exactly ONE stub validator — proves the runner -> manifest -> upload ->
verify pipeline end to end. This is NOT a real tool_authorization validator;
it never inspects the target agent's actual tool calls. The real implementation lives
under `.validators` with the same validator_id; it must not add a second "tool_authorization" validator
alongside this one.
"""
from __future__ import annotations

from typing import Any

from .contract import Status, Validator, ValidatorContext, ValidatorResult


class StubToolAuthorizationValidator(Validator):
    """Stub; the real validator lives under `.validators`.

    Always returns PASS with an observation naming itself as a stub, so a
    result produced by this class is unmistakable in any report or package
    it lands in — nobody should read a PASS from this and believe an agent's
    tool authorization was actually tested.
    """
    validator_id = "tool_authorization"
    version = "0.0.1-stub"

    def execute(self, context: ValidatorContext) -> Any:
        # A real validator would invoke the target agent with a battery of
        # in-scope and out-of-scope tool requests and record what it did.
        # The stub does nothing observable — there is nothing to probe yet.
        return {"context_execution_id": context.execution_id}

    def evaluate(self, result: Any) -> ValidatorResult:
        return ValidatorResult(
            validator_id=self.validator_id,
            version=self.version,
            status=Status.PASS,
            score=None,
            severity=None,
            observations=[
                "stub — real validator ships in Phase 3; this result does not "
                "reflect any actual tool-authorization check",
            ],
            evidence_ids=[],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        return []
