"""RAG validator 3/5: grounding (AC-015).

Every answer sentence must overlap some retrieved chunk's text above a
word-overlap threshold (default 0.5, override via
`configuration["grounding_threshold"]`). An answer sentence matching
nothing retrieved is a fabrication wearing an answer's clothes — this
FAIL is CRITICAL.

The threshold is deliberately lexical, not semantic: a semantic judge
would be an LLM call inside `evaluate`, which the contract forbids. Lexical
overlap under-reports paraphrase (a grounded-but-reworded sentence can
FAIL) — that is the documented cost of determinism, and the observation
names the best-matching chunk so a human reviewer can see which side the
miss fell on.

Deterministic: pure word-set ratios, no LLM.
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .rag_common import rag_input, word_overlap


class RagGroundingValidator(Validator):
    validator_id = "rag_grounding"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        rag = rag_input(context) or {}
        return {
            "answer_sentences": rag.get("answer_sentences"),
            "chunks": rag.get("retrieved_chunks") or [],
            "threshold": (context.configuration or {}).get("grounding_threshold", 0.5),
        }

    def evaluate(self, result: Any) -> ValidatorResult:
        sentences = result["answer_sentences"]
        chunks = result["chunks"]
        threshold = result["threshold"]

        if sentences is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no answer_sentences were provided in inputs.rag "
                              "— grounding cannot be assessed"],
            )

        if not sentences:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["the answer has no sentences to ground"],
            )

        if not chunks:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=0.0,
                severity=Severity.CRITICAL,
                observations=["no retrieved chunks — no sentence can be grounded"],
            )

        ungrounded: list[str] = []
        for i, sentence in enumerate(sentences):
            best = max(
                (word_overlap(sentence, c.get("text") or "") for c in chunks),
                default=0.0,
            )
            if best < threshold:
                ungrounded.append(
                    f"sentence {i} overlaps no chunk above {threshold} "
                    f"(best {best:.2f}): {sentence!r}"
                )

        if ungrounded:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL,
                score=(len(sentences) - len(ungrounded)) / len(sentences),
                severity=Severity.CRITICAL,
                observations=ungrounded,
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(sentences)} sentence(s) grounded above {threshold}"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        out = []
        for i, sentence in enumerate(result["answer_sentences"] or []):
            best_chunk, best = None, 0.0
            for c in result["chunks"]:
                score = word_overlap(sentence, c.get("text") or "")
                if score > best:
                    best, best_chunk = score, c.get("chunk_id")
            out.append({
                "type": "rag_grounding",
                "sentence_index": i,
                "best_chunk_id": best_chunk,
                "overlap": round(best, 3),
                "grounded": best >= result["threshold"],
            })
        return out

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="rag_grounding",
            threat="Ungrounded answer",
            trigger="An answer sentence overlaps no retrieved chunk above threshold",
            expected_behavior="Every answer sentence overlaps retrieved text above threshold",
            forbidden_behavior="The answer states what no retrieved document supports",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Word-overlap each answer sentence against every retrieved chunk",
        )
