"""MODEL validator 3/4: population stability index drift (AC-016).

Grades customer-computed PSI per feature against a stated threshold. Coriqo
never computes PSI itself — feature distributions never leave the customer
side, only the resulting index does (spec: "PSI over feature distributions
vs baseline hashes").

Deterministic: a numeric comparison per feature, no LLM.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .model_common import model_input


class ModelDriftPsiValidator(Validator):
    validator_id = "model_drift_psi"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        model = model_input(context) or {}
        return {"drift": model.get("drift")}

    def evaluate(self, result: Any) -> ValidatorResult:
        drift = result["drift"]
        if drift is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no drift report was provided in inputs.model "
                              "— PSI drift cannot be assessed"],
            )

        feature_psi: dict[str, float] = drift.get("feature_psi") or {}
        if not feature_psi:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["no feature PSI values reported"],
            )

        threshold = drift.get("threshold")
        if not isinstance(threshold, (int, float)):
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"drift.threshold={threshold!r} is not a number "
                              "— PSI drift cannot be assessed"],
            )

        bad_features = sorted(f for f, psi in feature_psi.items() if not isinstance(psi, (int, float)))
        if bad_features:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"feature(s) {bad_features} report a non-numeric PSI value"],
            )

        drifted = sorted(f for f, psi in feature_psi.items() if psi > threshold)

        if drifted:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=0.0,
                severity=Severity.HIGH,
                observations=[f"feature(s) {drifted} exceed PSI threshold {threshold}"],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(feature_psi)} feature(s) hold within PSI threshold {threshold}"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        drift = result["drift"]
        if drift is None:
            return []
        return [{
            "type": "model_drift",
            "feature_psi": drift.get("feature_psi"),
            "threshold": drift.get("threshold"),
        }]

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="model_drift_psi",
            threat="Population drift",
            trigger="A feature's population stability index exceeds the stated threshold",
            expected_behavior="Every feature's PSI holds within threshold against the baseline",
            forbidden_behavior="The model scores production traffic that has drifted from its baseline",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Compare customer-computed per-feature PSI against a stated threshold",
        )
