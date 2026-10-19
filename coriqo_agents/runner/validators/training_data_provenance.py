"""security validator 3/3: training data provenance (AC-017).

Data poisoning's deterministic half, per the spec: provenance. Every
dataset in the bill of materials must carry a signed source record — a
dataset with no signed record is exactly the unverified-source gap a
poisoning attack needs. The statistical half (outlier/backdoor detection)
is customer-run and advisory only; it travels as
`AssuranceExecution.advisory_observations`, never as this validator's
status (spec's "advisory only, never a status" rule, same as the MODEL
GenAI's hallucination/toxicity scoring).
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .sec_common import security_input


class TrainingDataProvenanceValidator(Validator):
    validator_id = "training_data_provenance"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        security = security_input(context) or {}
        poisoning = security.get("poisoning")
        return {"datasets": (poisoning or {}).get("datasets")}

    def evaluate(self, result: Any) -> ValidatorResult:
        datasets = result["datasets"]

        if datasets is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no datasets were provided in inputs.security.poisoning "
                              "— training data provenance cannot be assessed"],
            )

        if not isinstance(datasets, list):
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"datasets={datasets!r} is not a list — training data "
                              "provenance cannot be assessed"],
            )

        if not datasets:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["no datasets declared in the bill of materials"],
            )

        unsigned = [
            (d if isinstance(d, dict) else {}).get("id") for d in datasets
            if not (d if isinstance(d, dict) else {}).get("signed_source_record")
        ]

        if unsigned:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=(len(datasets) - len(unsigned)) / len(datasets),
                severity=Severity.HIGH,
                observations=[f"dataset(s) {unsigned} carry no signed source record"],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(datasets)} dataset(s) carry a signed source record"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        return [
            {
                "type": "training_data_provenance",
                "dataset_id": (d if isinstance(d, dict) else {}).get("id"),
                "signed_source_record": bool((d if isinstance(d, dict) else {}).get("signed_source_record")),
            }
            for d in (result["datasets"] or [])
        ]

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="training_data_provenance",
            threat="Data poisoning",
            trigger="A training dataset carries no signed source record",
            expected_behavior="Every training dataset in the bill of materials carries a signed source record",
            forbidden_behavior="The model trains on a dataset with no verifiable provenance",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Check every declared training dataset for a signed source record",
        )
