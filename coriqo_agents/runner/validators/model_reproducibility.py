"""MODEL validator 4/4: reproducibility (AC-016).

Re-scores a customer-supplied sample and checks it reproduces the stored
prediction within tolerance. Both `stored_prediction` and
`recomputed_prediction` arrive already computed by the customer — Coriqo
never runs the model, it grades whether the two numbers agree.

Deterministic: an absolute-difference comparison per sample row, no LLM.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .model_common import model_input


class ModelReproducibilityValidator(Validator):
    validator_id = "model_reproducibility"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        model = model_input(context) or {}
        return {"reproducibility": model.get("reproducibility")}

    def evaluate(self, result: Any) -> ValidatorResult:
        reproducibility = result["reproducibility"]
        if reproducibility is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no reproducibility sample was provided in inputs.model "
                              "— reproducibility cannot be assessed"],
            )

        sample: list[dict] = reproducibility.get("sample") or []
        if not sample:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["no sample rows to re-score"],
            )

        tolerance = reproducibility.get("tolerance")
        if not isinstance(tolerance, (int, float)):
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"reproducibility.tolerance={tolerance!r} is not a number "
                              "— reproducibility cannot be assessed"],
            )

        mismatched = []
        bad_rows = []
        for row in sample:
            row = row or {}
            stored = row.get("stored_prediction")
            recomputed = row.get("recomputed_prediction")
            if not isinstance(stored, (int, float)) or not isinstance(recomputed, (int, float)):
                bad_rows.append(row.get("id"))
                continue
            if abs(stored - recomputed) > tolerance:
                mismatched.append(row.get("id"))

        if bad_rows:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"sample row(s) {bad_rows} are missing a numeric "
                              "stored_prediction or recomputed_prediction"],
            )

        if mismatched:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=(len(sample) - len(mismatched)) / len(sample),
                severity=Severity.HIGH,
                observations=[f"sample row(s) {mismatched} do not reproduce their stored "
                              f"prediction within tolerance {tolerance}"],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(sample)} sample row(s) reproduce within tolerance {tolerance}"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        reproducibility = result["reproducibility"]
        if reproducibility is None:
            return []
        out = []
        tolerance = reproducibility.get("tolerance")
        for row in reproducibility.get("sample") or []:
            row = row or {}
            stored = row.get("stored_prediction")
            recomputed = row.get("recomputed_prediction")
            numeric = isinstance(stored, (int, float)) and isinstance(recomputed, (int, float))
            reproduced = (
                abs(stored - recomputed) <= tolerance
                if numeric and isinstance(tolerance, (int, float))
                else None
            )
            out.append({
                "type": "model_reproducibility",
                "id": row.get("id"),
                "stored_prediction": stored,
                "recomputed_prediction": recomputed,
                "reproduced": reproduced,
            })
        return out

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="model_reproducibility",
            threat="Silent model drift or pipeline defect",
            trigger="A re-scored sample row does not reproduce its stored prediction within tolerance",
            expected_behavior="Re-scoring a sample reproduces stored predictions within tolerance",
            forbidden_behavior="The deployed model no longer matches the validated artifact",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Re-score a sample and compare against the stored prediction within tolerance",
        )
