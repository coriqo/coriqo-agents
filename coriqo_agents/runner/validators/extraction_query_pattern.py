"""Security validator 2/3: model extraction resistance (AC-017).

Deterministic counts, not attack simulation (spec's own framing): per-
principal query volume and input-space coverage, both already aggregated
by the harness, checked against the mandate's own caps. A principal
querying past its volume cap is a WARNING (could be legitimate heavy use);
one covering more of the input space than the mandate caps is the
systematic-probing signature extraction attempts actually leave, so it
FAILs.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .sec_common import security_input


class ExtractionQueryPatternValidator(Validator):
    validator_id = "extraction_query_pattern"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        security = security_input(context) or {}
        extraction = security.get("extraction")
        return {
            "queries": (extraction or {}).get("queries"),
            "max_queries_per_principal": (extraction or {}).get("max_queries_per_principal"),
            "max_input_space_coverage": (extraction or {}).get("max_input_space_coverage"),
        }

    def evaluate(self, result: Any) -> ValidatorResult:
        queries = result["queries"]
        max_queries = result["max_queries_per_principal"]
        max_coverage = result["max_input_space_coverage"]

        if queries is None or max_queries is None or max_coverage is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["queries, max_queries_per_principal, and max_input_space_coverage "
                              "are all required in inputs.security.extraction — extraction "
                              "resistance cannot be assessed"],
            )

        if not isinstance(max_queries, (int, float)) or not isinstance(max_coverage, (int, float)):
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"max_queries_per_principal={max_queries!r} and "
                              f"max_input_space_coverage={max_coverage!r} must both be numbers "
                              "— extraction resistance cannot be assessed"],
            )

        if not isinstance(queries, list):
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=[f"queries={queries!r} is not a list — extraction "
                              "resistance cannot be assessed"],
            )

        if not queries:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["no per-principal query stats reported for this window"],
            )

        coverage_breaches: list[str] = []
        volume_breaches: list[str] = []
        for row in queries:
            row = row if isinstance(row, dict) else {}
            pid = row.get("principal_id")
            count = row.get("query_count")
            coverage = row.get("input_space_coverage")
            if isinstance(coverage, (int, float)) and coverage > max_coverage:
                coverage_breaches.append(
                    f"principal {pid!r} covered {coverage:.0%} of the input space "
                    f"(cap {max_coverage:.0%})"
                )
            if isinstance(count, (int, float)) and count > max_queries:
                volume_breaches.append(
                    f"principal {pid!r} issued {count} queries (cap {max_queries})"
                )

        if coverage_breaches:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=0.0,
                severity=Severity.CRITICAL,
                observations=coverage_breaches + volume_breaches,
            )

        if volume_breaches:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.WARNING,
                severity=Severity.MEDIUM,
                observations=volume_breaches,
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(queries)} principal(s) hold within volume and coverage caps"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        max_queries = result["max_queries_per_principal"]
        max_coverage = result["max_input_space_coverage"]
        out = []
        for row in result["queries"] or []:
            row = row if isinstance(row, dict) else {}
            count = row.get("query_count")
            coverage = row.get("input_space_coverage")
            out.append({
                "type": "extraction_query_stats",
                "principal_id": row.get("principal_id"),
                "query_count": count,
                "input_space_coverage": coverage,
                "over_volume_cap": isinstance(count, (int, float)) and max_queries is not None
                                   and count > max_queries,
                "over_coverage_cap": isinstance(coverage, (int, float)) and max_coverage is not None
                                      and coverage > max_coverage,
            })
        return out

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="extraction_query_pattern",
            threat="Model extraction",
            trigger="A principal's queries cover more of the input space than the mandate caps",
            expected_behavior="Every principal's query volume and input-space coverage hold within its mandate's caps",
            forbidden_behavior="A principal systematically probes the input space past its declared cap",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Compare per-principal query volume and input-space coverage against the mandate's caps",
        )
