"""The offline assurance-package verifier and its CLI.

The assertions that matter are about what the verifier refuses to say. It
must never print `verified` for a package it could not fully check, and it
must never imply it checked a signature, because it cannot.
"""
from __future__ import annotations

import json

import pytest

from coriqo_agents.openformat import cli
from coriqo_agents.openformat.formats import UnsupportedFormatError
from coriqo_agents.openformat.package_verify import (
    CHECK_FAIL,
    CHECK_PASS,
    CHECK_UNEVALUATED,
    VERDICT_FAILED,
    VERDICT_INCOMPLETE,
    VERDICT_VERIFIED,
    canonical_content_hash,
    verify_package,
)


def _package(**overrides) -> dict:
    package = {
        "schema_version": "1.1",
        "execution_id": "ex_1",
        "generated_at": "2026-01-02T03:04:05Z",
        "executive_summary": "Two validators ran against the servicing agent.",
        "scope": {"environment": "STAGING"},
        "ai_system": {"model_id": "m_1"},
        "protocol": {"name": "high_risk_agent_assurance_v1", "version": "1.0"},
        "validators": [
            {"validator_id": "tool_authorization", "version": "1.0.0", "status": "PASS",
             "evidence_ids": ["ev_1"]},
        ],
        "findings": [{"validator_id": "tool_authorization", "title": "none"}],
        "evidence_manifest": ["ev_1"],
        "runner_attestation": {"engine": "native"},
        "decision": {"conclusion": "EFFECTIVE"},
    }
    package.update(overrides)
    return package


def _status(report, check_id):
    return next(c["status"] for c in report["checks"] if c["check_id"] == check_id)


# ------------------------------------------------------------- version refusal

def test_an_unknown_schema_version_is_refused_rather_than_read():
    with pytest.raises(UnsupportedFormatError, match="schema_version"):
        verify_package(_package(schema_version="2.0"))


def test_a_package_with_no_schema_version_is_refused():
    package = _package()
    del package["schema_version"]
    with pytest.raises(UnsupportedFormatError):
        verify_package(package)


def test_something_that_is_not_an_object_is_refused():
    with pytest.raises(UnsupportedFormatError):
        verify_package(["not", "a", "package"])


# ------------------------------------------------------- the verdict vocabulary

def test_a_package_with_no_content_hash_supplied_is_incomplete_not_verified():
    """The single most important assertion in this file. Without a hash to
    compare against, nothing ties the document in front of you to anything
    Coriqo sealed, and a tool that printed `verified` on that basis would
    be worse than no tool."""
    report = verify_package(_package())
    assert _status(report, "CONTENT_HASH") == CHECK_UNEVALUATED
    assert report["verdict"] == VERDICT_INCOMPLETE


def test_a_fully_checked_package_verifies():
    package = _package()
    report = verify_package(package, expected_content_hash=canonical_content_hash(package))
    assert report["verdict"] == VERDICT_VERIFIED, report
    assert all(c["status"] == CHECK_PASS for c in report["checks"])


def test_an_edited_package_fails_the_hash_check():
    package = _package()
    sealed = canonical_content_hash(package)
    package["executive_summary"] = "Two validators ran against the servicing agent. All good."
    report = verify_package(package, expected_content_hash=sealed)
    assert _status(report, "CONTENT_HASH") == CHECK_FAIL
    assert report["verdict"] == VERDICT_FAILED


def test_a_failed_check_outranks_an_unevaluated_one():
    """A package that both fails a check and could not run another is
    `failed`, not `incomplete` — the reader needs to see the failure."""
    report = verify_package(_package(validators=[{"validator_id": "x", "evidence_ids": ["ev_9"]}]))
    assert _status(report, "REFERENCES") == CHECK_FAIL
    assert report["verdict"] == VERDICT_FAILED


def test_a_v1_0_package_is_readable_but_its_structure_is_not_checked():
    """1.0 is a real version consumers still hold, and no schema was ever
    published for it. Reporting the structural check as unevaluated is the
    honest answer; a silent PASS would claim we validated a document
    against a schema that does not exist."""
    report = verify_package(_package(schema_version="1.0"))
    assert _status(report, "SCHEMA") == CHECK_UNEVALUATED
    assert report["verdict"] == VERDICT_INCOMPLETE


def test_the_report_says_in_so_many_words_that_no_signature_was_checked():
    report = verify_package(_package())
    assert report["signature_checked"] is False
    assert "does not check a signature" in report["note"]


# ---------------------------------------------------------------- references

def test_a_finding_citing_a_validator_that_did_not_run_is_a_reference_failure():
    report = verify_package(_package(findings=[{"validator_id": "never_ran"}]))
    assert _status(report, "REFERENCES") == CHECK_FAIL


def test_a_finding_with_no_validator_id_is_not_a_reference_failure():
    """A reviewer-raised finding legitimately names no validator. Treating
    that as a broken reference would fail every package carrying manual
    review notes."""
    report = verify_package(_package(findings=[{"title": "raised by the reviewer"}]))
    assert _status(report, "REFERENCES") == CHECK_PASS


def test_evidence_cited_but_not_listed_in_the_manifest_is_a_reference_failure():
    report = verify_package(_package(evidence_manifest=[]))
    assert _status(report, "REFERENCES") == CHECK_FAIL


# ---------------------------------------------------------------- schema check

def test_a_package_missing_a_required_field_fails_the_schema_check():
    package = _package()
    del package["decision"]
    report = verify_package(package, expected_content_hash=canonical_content_hash(package))
    assert _status(report, "SCHEMA") == CHECK_FAIL
    assert report["verdict"] == VERDICT_FAILED


def test_the_schema_check_is_not_shadowed_by_the_reference_check():
    """A break caught by an outer layer proves nothing
    about the inner one.

    The first attempt at this test set `validators` to a string, and both
    checks failed — the reference walk iterated the string's characters and
    reported them as non-objects, so the SCHEMA failure proved nothing.
    Widened here to a break only the schema can see: `generated_at` is
    typed `string` in the schema and the reference walk never reads it, so
    exactly one check may fail."""
    package = _package(generated_at=17520000)
    report = verify_package(package)
    assert _status(report, "SCHEMA") == CHECK_FAIL
    assert _status(report, "REFERENCES") == CHECK_PASS


# --------------------------------------------------------------------- the CLI

def test_the_cli_exits_incomplete_when_no_hash_is_supplied(tmp_path, capsys):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_package()))
    assert cli.main([str(path)]) == cli.EXIT_INCOMPLETE
    assert "verdict: incomplete" in capsys.readouterr().out


def test_the_cli_exits_zero_only_on_a_full_verification(tmp_path):
    package = _package()
    path = tmp_path / "p.json"
    path.write_text(json.dumps(package))
    assert cli.main([str(path), "--content-hash", canonical_content_hash(package)]) == (
        cli.EXIT_VERIFIED
    )


def test_the_cli_refuses_an_unknown_version_with_its_own_exit_code(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_package(schema_version="9.9")))
    assert cli.main([str(path)]) == cli.EXIT_UNREADABLE


def test_the_cli_reports_a_missing_file_as_unreadable_not_as_a_failed_check(tmp_path):
    assert cli.main([str(tmp_path / "absent.json")]) == cli.EXIT_UNREADABLE


def test_the_cli_has_no_pubkey_flag():
    """A `--pubkey` would read as an authenticity check to anyone running
    it, and there is nothing offline for it to verify against. Its absence
    is a decision, so it gets an assertion rather than a comment."""
    actions = {action.dest for action in cli.build_parser()._actions}
    assert "pubkey" not in actions
    assert "public_key" not in actions


@pytest.mark.parametrize("declared", [[1, 1], {"major": 1}, 1.1, None])
def test_an_unhashable_or_non_string_schema_version_is_refused_not_a_crash(declared):
    """`declared not in KNOWN_SCHEMA_VERSIONS` is a hash lookup, so a list
    or an object raises `TypeError: unhashable type` before the fix — a raw
    traceback out of the CLI instead of the documented refusal every other
    malformed version gets."""
    package = _package()
    package["schema_version"] = declared
    with pytest.raises(UnsupportedFormatError, match="schema_version"):
        verify_package(package)


def test_the_cli_reports_a_list_schema_version_rather_than_traceback(tmp_path, capsys):
    package = _package()
    package["schema_version"] = [1, 1]
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package), encoding="utf-8")

    code = cli.main([str(path)])

    assert code != 0
    combined = capsys.readouterr()
    assert "schema_version" in (combined.err + combined.out)
