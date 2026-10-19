"""RAG slice: shared input shape and text helpers.

Every RAG validator reads `context.inputs["rag"]`, one answered question:

    {
      "question": str,                       # what was asked
      "expected_doc_ids": [str],             # docs the task calls for
      "retrieved_chunks": [                  # what the retriever returned
        {"chunk_id": str, "document_id": str, "text": str},
      ],
      "answer_sentences": [str],             # the answer, split by the caller
      "citations": [                         # one per cited sentence
        {"sentence_index": int, "document_id": str, "span": str},
      ],
      "document_allowlist": [str],           # doc ids permitted for this task
      "injection_markers": [str],            # planted forbidden instructions
      "forbidden_claims": [str],             # claims the answer must not make
    }

The reference data itself never leaves the customer side — only its hashes
land in the manifest's `input_dataset_hashes` (contract.py). Validators see
whatever the local runner was given; a missing key means "not supplied",
never an empty stand-in, so every validator below distinguishes "nothing to
check" (NOT_APPLICABLE) from "cannot assess" (INCONCLUSIVE) instead of
collapsing the two into a clean verdict.
"""
from __future__ import annotations

import re
import string
from typing import Any


def rag_input(context) -> dict[str, Any] | None:
    """The `inputs["rag"]` dict, or None when the run carried none."""
    rag = (context.inputs or {}).get("rag")
    return rag if isinstance(rag, dict) else None


def normalize(text: str) -> str:
    """Lowercase, punctuation-stripped, whitespace-collapsed — the single
    normalization every RAG comparison uses, so "contains" means the same
    thing in citation checks and grounding checks."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return re.sub(r"\s+", " ", text).strip()


def word_overlap(sentence: str, chunk_text: str) -> float:
    """Fraction of the sentence's distinct words appearing in the chunk.
    0.0 for an empty sentence rather than a division error reading as a
    verdict."""
    words = set(normalize(sentence).split())
    if not words:
        return 0.0
    chunk_words = set(normalize(chunk_text).split())
    return len(words & chunk_words) / len(words)


def chunks_by_doc(chunks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.get("document_id") or "", []).append(chunk)
    return grouped
