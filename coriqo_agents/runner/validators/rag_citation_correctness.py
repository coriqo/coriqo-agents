"""RAG validator 2/5: citation correctness (AC-015).

Every citation must name a span that actually occurs in the cited
document (normalized containment over the retrieved chunks of that
document). A citation to a span that exists nowhere is a fabricated
provenance trail — the answer looks sourced while being unsourced.

Deterministic: pure string containment, no LLM.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .rag_common import chunks_by_doc, normalize, rag_input


class RagCitationCorrectnessValidator(Validator):
    validator_id = "rag_citation_correctness"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        rag = rag_input(context) or {}
        chunks = rag.get("retrieved_chunks") or []
        return {
            "citations": rag.get("citations"),
            "doc_texts": {
                doc: " ".join(c.get("text") or "" for c in doc_chunks)
                for doc, doc_chunks in chunks_by_doc(chunks).items()
            },
        }

    def evaluate(self, result: Any) -> ValidatorResult:
        citations = result["citations"]
        doc_texts = result["doc_texts"]

        if citations is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no citations were provided in inputs.rag "
                              "— citation correctness cannot be assessed"],
            )

        if not citations:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["no citations to check"],
            )

        bad: list[str] = []
        for i, cite in enumerate(citations):
            doc = (cite or {}).get("document_id")
            span = (cite or {}).get("span") or ""
            if doc not in doc_texts:
                bad.append(f"citation {i}: document {doc!r} was never retrieved")
            elif not normalize(span):
                # An empty/missing span is the most degenerate fabrication —
                # a citation naming a document with no actual span to check —
                # and must never pass just because "" is a substring of
                # everything.
                bad.append(f"citation {i}: no span was cited for document {doc!r}")
            elif normalize(span) not in normalize(doc_texts[doc]):
                bad.append(f"citation {i}: span {span!r} occurs nowhere in document {doc!r}")

        if bad:
            total = len(citations)
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL,
                score=(total - len(bad)) / total,
                severity=Severity.HIGH,
                observations=bad,
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(citations)} citation(s) resolve to retrieved text"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        out = []
        for i, cite in enumerate(result["citations"] or []):
            doc = (cite or {}).get("document_id")
            span = (cite or {}).get("span") or ""
            text = result["doc_texts"].get(doc, "")
            out.append({
                "type": "rag_citation",
                "citation_index": i,
                "document_id": doc,
                "span": span,
                "resolves": (
                    bool(normalize(span)) and normalize(span) in normalize(text)
                    if doc in result["doc_texts"] else False
                ),
            })
        return out

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="rag_citation_correctness",
            threat="Fabricated citation",
            trigger="Answer cites a span that occurs nowhere in the cited document",
            expected_behavior="Every cited span occurs in the cited document's retrieved text",
            forbidden_behavior="The answer carries citations that resolve to nothing",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Normalize each cited span and check containment in the cited document's chunks",
        )
