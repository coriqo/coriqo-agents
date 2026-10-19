"""MODEL validator 2/4: fairness parity (AC-016).

Grades the gap between the customer-computed metric's best and worst group
against a customer-stated tolerance. Groups are defined by the customer
(spec: "groups defined by the customer") — this validator never infers a
protected attribute, it only compares numbers already grouped upstream.

Deterministic: max-min gap vs tolerance, no LLM.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .model_common import model_input


class ModelFairnessParityValidator(Validator):
    validator_id = "model_fairness_parity"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        model = model_input(context) or {}
        return {"fairness": model.get("fairness")}

    def evaluate(self, result: Any) -> ValidatorResult:
        fairness = result["fairness"]
        if fairness is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no fairness report was provided in inputs.model "
                              "— group parity cannot be assessed"],
            )

        group_values: dict[str, float] = fairness.get("group_values") or {}
        if len(group_values) < 2:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["fewer than two groups reported — nothing to compare"],
            )

        metric = fairness.get("metric")
        tolerance = fairness.get("tolerance")
        bad_groups = sorted(g for g, v in group_values.items() if not isinstance(v, (int, float)))
        if bad_groups or not isinstance(tolerance, (int, float)):
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"group_values {bad_groups} or tolerance={tolerance!r} "
                              "is not a number — group parity cannot be assessed"],
            )
        best_group = max(group_values, key=lambda g: group_values[g])
        worst_group = min(group_values, key=lambda g: group_values[g])
        gap = group_values[best_group] - group_values[worst_group]

        if gap > tolerance:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=0.0,
                severity=Severity.HIGH,
                observations=[f"{metric} gap {gap} between {best_group!r} and {worst_group!r} "
                              f"exceeds tolerance {tolerance}"],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"{metric} gap {gap} between {best_group!r} and {worst_group!r} "
                          f"holds within tolerance {tolerance}"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        fairness = result["fairness"]
        if fairness is None:
            return []
        return [{
            "type": "model_fairness",
            "metric": fairness.get("metric"),
            "group_values": fairness.get("group_values"),
            "tolerance": fairness.get("tolerance"),
        }]

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="model_fairness_parity",
            threat="Disparate impact",
            trigger="The metric's gap between the best- and worst-performing group exceeds tolerance",
            expected_behavior="The metric holds within tolerance across every declared group",
            forbidden_behavior="The model is used in production with an unaddressed group disparity",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Compare the customer-computed group metrics' spread against a stated tolerance",
        )
