"""RAG validator 1/5: retrieval completeness (AC-015).

Every document id the task calls for must appear among the retrieved
chunks' document ids. A miss means the assessment ran against the wrong
evidence: downstream validators would grade an answer to a different
corpus, so this FAIL is CRITICAL and the run's other verdicts cannot be
read as clean on their own.

Deterministic: pure set-subset over caller-supplied ids, no LLM.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .rag_common import chunks_by_doc, rag_input


class RagRetrievalCompletenessValidator(Validator):
    validator_id = "rag_retrieval_completeness"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        rag = rag_input(context) or {}
        chunks = rag.get("retrieved_chunks") or []
        return {
            "expected_doc_ids": rag.get("expected_doc_ids"),
            "retrieved_doc_ids": sorted(chunks_by_doc(chunks)),
        }

    def evaluate(self, result: Any) -> ValidatorResult:
        expected = result["expected_doc_ids"]
        retrieved = result["retrieved_doc_ids"]

        if expected is None:
            # Mirrors tool_authorization's unset-roster rule: no expected
            # set means "unrestricted," which cannot be assessed — reporting
            # PASS would manufacture certainty the contract forbids.
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no expected_doc_ids were provided in inputs.rag "
                              "— retrieval completeness cannot be assessed"],
            )

        if not expected:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["expected_doc_ids is empty — nothing to retrieve"],
            )

        missing = [doc for doc in expected if doc not in set(retrieved)]
        if missing:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL,
                score=(len(expected) - len(missing)) / len(expected),
                severity=Severity.CRITICAL,
                observations=[
                    f"expected document {doc!r} was not retrieved "
                    f"(retrieved: {retrieved or 'nothing'})"
                    for doc in missing
                ],
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(expected)} expected document(s) retrieved"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        expected = result["expected_doc_ids"] or []
        retrieved = set(result["retrieved_doc_ids"])
        return [
            {"type": "rag_retrieval", "document_id": doc, "retrieved": doc in retrieved}
            for doc in expected
        ]

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="rag_retrieval_completeness",
            threat="Incomplete retrieval",
            trigger="Retriever returned chunks missing documents the task calls for",
            expected_behavior="Every expected document id appears among the retrieved chunks",
            forbidden_behavior="The answer is assessed against a corpus missing expected documents",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Compare retrieved chunk document ids against the caller-supplied expected set",
        )
