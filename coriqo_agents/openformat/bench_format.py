"""Scoring a benchmark run, as pure functions over plain dicts.

Split from the harness that runs the validators (`runner/bench.py`) so the
scoring rules can be tested without instantiating a validator, and so a
third party who ran the corpus some other way can score their own
observations with the same code that produced Coriqo's published numbers.

D9: there is no model call anywhere in this file or in anything it imports.
A benchmark verdict is a string comparison between what the corpus says and
what a deterministic validator returned.

Two rules do the real work here, both of them the "unevaluated must not
collapse into clean" rule from the programme's standing list:

1. `classify` compares only statuses in `COMPARABLE_STATUSES` — the GOOD
   list, read off the published schema. Everything else, known
   (ERROR/INCONCLUSIVE) or not (a status invented by a third-party
   validator, a typo, an empty string), is `unevaluated`.
2. `score` divides by matched + mismatched, never by the total. Unevaluated
   checks are reported next to the ratio, never inside it, and a run with
   nothing scorable gets `ratio: None` rather than 1.0.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .formats import (
    BENCHMARK_RESULT_FORMAT,
    BENCHMARK_RESULT_VERSION,
    COMPARABLE_STATUSES,
    OUTCOME_MATCHED,
    OUTCOME_MISMATCHED,
    OUTCOME_UNEVALUATED,
)


def classify(expected: str, observed: Any) -> str:
    """The outcome of one (expected, observed) pair.

    `observed` is deliberately typed `Any`: it is whatever a validator put
    in its result, and a validator outside this repo can put anything
    there. Anything that is not a string in the comparable set is
    unevaluated — including `None`, which is what a result with no status
    at all looks like.
    """
    if not isinstance(observed, str) or observed not in COMPARABLE_STATUSES:
        return OUTCOME_UNEVALUATED
    if expected not in COMPARABLE_STATUSES:
        # A corpus label outside the published enum. The schema forbids it,
        # but this function is also reachable from a caller that skipped
        # validation, and "the label is not one we can compare" is exactly
        # an unevaluated check — not a mismatch, which would report the
        # validator as wrong about a case that was never askable.
        return OUTCOME_UNEVALUATED
    return OUTCOME_MATCHED if observed == expected else OUTCOME_MISMATCHED


def score(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """The score block, from the per-case check lists."""
    matched = mismatched = unevaluated = 0
    for case in cases:
        for check in case["checks"]:
            outcome = check["outcome"]
            if outcome == OUTCOME_MATCHED:
                matched += 1
            elif outcome == OUTCOME_MISMATCHED:
                mismatched += 1
            else:
                # Every outcome that is not one of the two comparable ones.
                # Enumerating the two good ones rather than testing for
                # `== OUTCOME_UNEVALUATED` means a third outcome added to a
                # later format version lands here, outside the ratio,
                # instead of being silently dropped from all three counters
                # and inflating nothing while accounting for nothing.
                unevaluated += 1
    scored = matched + mismatched
    return {
        "scored": scored,
        "matched": matched,
        "unevaluated": unevaluated,
        # None, not 1.0 and not 0.0: a run that could not evaluate a single
        # check has no score. 1.0 would read as perfect and 0.0 as a total
        # failure; both are claims about an agent that was never measured.
        "ratio": (matched / scored) if scored else None,
    }


def corpus_sha256(corpus_bytes: bytes) -> str:
    """Over the file's bytes exactly as read — not over a re-serialization
    of the parsed document, which would hash a normalisation of the corpus
    rather than the corpus, and let two byte-different files claim the same
    digest."""
    return hashlib.sha256(corpus_bytes).hexdigest()


def build_result(
    *,
    corpus: dict[str, Any],
    corpus_bytes: bytes,
    cases: list[dict[str, Any]],
    sdk_version: str,
    validator_versions: dict[str, str],
) -> dict[str, Any]:
    """Assemble the published `coriqo-benchmark-result` document.

    Carries no timestamp and no host identity, by design: the acceptance
    criterion for this format is that two machines running the same corpus
    with the same validator versions produce byte-identical output. A
    `generated_at` would break that for no benefit a consumer can use — the
    corpus digest and the validator versions already say what was run, and
    when it ran is something the publisher of a score can state alongside
    it without it being inside the signed-over document.
    """
    total_checks = sum(len(case["checks"]) for case in cases)
    computed = score(cases)
    return {
        "format": BENCHMARK_RESULT_FORMAT,
        "format_version": BENCHMARK_RESULT_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["corpus_version"],
        "corpus_sha256": corpus_sha256(corpus_bytes),
        "harness": {
            "sdk_version": sdk_version,
            "validator_versions": dict(sorted(validator_versions.items())),
        },
        "cases": cases,
        "totals": {
            "cases": len(cases),
            "checks": total_checks,
            "matched": computed["matched"],
            "mismatched": total_checks - computed["matched"] - computed["unevaluated"],
            "unevaluated": computed["unevaluated"],
        },
        "score": computed,
    }
