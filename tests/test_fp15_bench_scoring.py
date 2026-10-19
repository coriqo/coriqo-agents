"""The benchmark scoring rules.

The assertions that matter here are all one rule: an unevaluated check must
never turn into a clean one. A benchmark format that scored ERROR as a pass,
or reported 1.0 for a run where nothing could be evaluated, would be a way
to launder an incomplete result into a published score — and because the
format is public, it would be a way for anyone, not just us.
"""
from __future__ import annotations

import pytest

from coriqo_agents.openformat.bench_format import build_result, classify, score
from coriqo_agents.openformat.formats import (
    COMPARABLE_STATUSES,
    OUTCOME_MATCHED,
    OUTCOME_MISMATCHED,
    OUTCOME_UNEVALUATED,
)


def _case(*outcomes: str) -> dict:
    return {
        "case_id": "c", "domain": "loan_servicing",
        "checks": [
            {"validator_id": f"v{i}", "validator_version": "1.0.0", "expected": "PASS",
             "observed": "PASS", "outcome": outcome}
            for i, outcome in enumerate(outcomes)
        ],
    }


def test_the_comparable_statuses_come_from_the_published_schema():
    """`COMPARABLE_STATUSES` is read out of the corpus schema at import
    time rather than typed in Python. If someone retypes it, this fails and
    the code and the published format can drift apart again."""
    assert COMPARABLE_STATUSES == ("PASS", "FAIL", "WARNING", "NOT_APPLICABLE")


def test_a_matching_status_matches():
    assert classify("FAIL", "FAIL") == OUTCOME_MATCHED


def test_a_different_comparable_status_is_a_mismatch():
    assert classify("FAIL", "PASS") == OUTCOME_MISMATCHED


@pytest.mark.parametrize("observed", ["ERROR", "INCONCLUSIVE"])
def test_the_two_statuses_that_mean_no_verdict_are_unevaluated(observed):
    """Not a mismatch: a validator that errored has not disagreed with the
    label, it has failed to answer. Counting it as a mismatch would make a
    broken harness look like a wrong validator."""
    assert classify("PASS", observed) == OUTCOME_UNEVALUATED
    assert classify("FAIL", observed) == OUTCOME_UNEVALUATED


@pytest.mark.parametrize("observed", [
    "PASSED",           # a near-miss spelling
    "pass",             # the right word, the wrong case
    "MOSTLY_FINE",      # a status a third-party validator invented
    "",
    None,
    True,
    ["PASS"],
])
def test_anything_outside_the_good_list_is_unevaluated(observed):
    """The whole point of enumerating the GOOD statuses. A bad-list would
    have to be extended every time somebody invents a status, and would
    silently score the next one as a match."""
    assert classify("PASS", observed) == OUTCOME_UNEVALUATED


def test_an_unlabellable_expectation_is_unevaluated_not_a_mismatch():
    """A corpus label the schema forbids reached us anyway (a caller that
    skipped validation). The validator was never asked a question it could
    answer, so it must not be recorded as having got it wrong."""
    assert classify("ERROR", "PASS") == OUTCOME_UNEVALUATED


def test_unevaluated_checks_stay_out_of_the_denominator():
    computed = score([_case(OUTCOME_MATCHED, OUTCOME_MISMATCHED, OUTCOME_UNEVALUATED)])
    assert computed == {"scored": 2, "matched": 1, "unevaluated": 1, "ratio": 0.5}


def test_a_run_with_nothing_evaluable_has_no_score_rather_than_a_perfect_one():
    computed = score([_case(OUTCOME_UNEVALUATED, OUTCOME_UNEVALUATED)])
    assert computed["ratio"] is None, (
        "a run where no check could be evaluated must report no score — 1.0 would read as "
        "a perfect result for an agent nobody measured"
    )
    assert computed["scored"] == 0
    assert computed["unevaluated"] == 2


def test_an_outcome_this_version_does_not_know_lands_outside_the_ratio():
    """Rule 8 applied to the outcome vocabulary itself, not just to
    statuses: `score` enumerates matched and mismatched and puts everything
    else in unevaluated, so an outcome added by a later format version
    cannot vanish from all three counters."""
    computed = score([_case(OUTCOME_MATCHED, "partially_matched")])
    assert computed["scored"] == 1
    assert computed["unevaluated"] == 1
    assert computed["ratio"] == 1.0


def test_the_totals_and_the_score_cannot_disagree():
    cases = [_case(OUTCOME_MATCHED, OUTCOME_MISMATCHED), _case(OUTCOME_UNEVALUATED)]
    result = build_result(
        corpus={"corpus_id": "c", "corpus_version": "1"},
        corpus_bytes=b"{}", cases=cases, sdk_version="0.0.0", validator_versions={},
    )
    totals = result["totals"]
    assert totals["checks"] == totals["matched"] + totals["mismatched"] + totals["unevaluated"]
    assert result["score"]["scored"] == totals["matched"] + totals["mismatched"]
    assert result["score"]["unevaluated"] == totals["unevaluated"]


def test_the_result_document_carries_no_timestamp_or_host_identity():
    """Reproducibility is the format's only real guarantee. Any field that
    varies between two machines running the same corpus breaks it, so the
    absence of one is worth an assertion rather than a comment."""
    result = build_result(
        corpus={"corpus_id": "c", "corpus_version": "1"},
        corpus_bytes=b"{}", cases=[_case(OUTCOME_MATCHED)],
        sdk_version="0.0.0", validator_versions={"v0": "1.0.0"},
    )
    forbidden = {"generated_at", "timestamp", "run_id", "hostname", "started_at", "duration_ms"}
    assert not (forbidden & set(result)), (
        f"the result carries {forbidden & set(result)}, which differs between two runs of the "
        "same corpus and makes a published score unreproducible"
    )


def test_the_corpus_digest_is_over_the_bytes_not_the_parsed_document():
    """Two files that parse to the same document but differ in whitespace
    are different corpora as far as a reader checking a published digest is
    concerned, and must hash differently."""
    a = build_result(corpus={"corpus_id": "c", "corpus_version": "1"}, corpus_bytes=b'{"a": 1}',
                     cases=[_case(OUTCOME_MATCHED)], sdk_version="0", validator_versions={})
    b = build_result(corpus={"corpus_id": "c", "corpus_version": "1"}, corpus_bytes=b'{"a":1}',
                     cases=[_case(OUTCOME_MATCHED)], sdk_version="0", validator_versions={})
    assert a["corpus_sha256"] != b["corpus_sha256"]
