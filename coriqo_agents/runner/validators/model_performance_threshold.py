"""MODEL validator 1/4: performance threshold (AC-016).

Grades one customer-computed metric against the validation report's stated
threshold. Coriqo never scores the model itself — `observed` and
`threshold` arrive as numbers the customer already produced on their own
held-out set; this validator only re-derives PASS/FAIL from them.

Deterministic: a numeric comparison, no LLM, no re-scoring.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .model_common import clears_threshold, model_input


class ModelPerformanceThresholdValidator(Validator):
    validator_id = "model_performance_threshold"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        model = model_input(context) or {}
        return {"performance": model.get("performance")}

    def evaluate(self, result: Any) -> ValidatorResult:
        performance = result["performance"]
        if performance is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no performance report was provided in inputs.model "
                              "— the threshold cannot be assessed"],
            )

        metric = performance.get("metric")
        observed = performance.get("observed")
        threshold = performance.get("threshold")
        direction = performance.get("direction")
        try:
            cleared = clears_threshold(observed, threshold, direction)
        except ValueError as exc:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[str(exc)],
            )

        if not cleared:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=0.0,
                severity=Severity.HIGH,
                observations=[f"{metric} observed {observed} does not clear threshold "
                              f"{threshold} ({direction})"],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"{metric} observed {observed} clears threshold {threshold} ({direction})"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        performance = result["performance"]
        if performance is None:
            return []
        return [{
            "type": "model_performance",
            "metric": performance.get("metric"),
            "observed": performance.get("observed"),
            "threshold": performance.get("threshold"),
            "direction": performance.get("direction"),
        }]

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="model_performance_threshold",
            threat="Performance regression",
            trigger="A held-out metric falls short of the validation report's stated threshold",
            expected_behavior="The metric clears its stated threshold",
            forbidden_behavior="The model is used in production below its validated performance bar",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Compare the customer-computed held-out metric against the report's threshold",
        )
