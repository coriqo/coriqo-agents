"""`coriqo_agents.runner.ci`'s pure logic: the gate decision,
the JUnit report, and trace-file loading.

Pure-stdlib, no app/DB dependency — run standalone:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_ci_gate_logic.py

Deliberately does NOT exercise `ci.main()`'s network calls — that needs a
running server or a full mock of it, which
test_ci_no_secrets.py does with a fake transport. This file is only the
functions `main()` calls that contain actual decision logic: `_gate` (the
one place a --fail-on threshold is applied — see its own docstring for why
it is not "scoring") and `_junit_xml`, plus `_load_trace`'s file/directory
handling.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

from coriqo_agents.runner.ci import CiUsageError, _gate, _junit_xml, _load_trace, build_parser


def _otlp_payload(agent_id: str = "agent_1", agent_version: str = "1.0.0") -> dict:
    """A batch with no spans — enough to resolve one agent identity via the
    resource attributes, with no event content this suite needs."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [
                    {"key": "coriqo.agent_id", "value": {"stringValue": agent_id}},
                    {"key": "coriqo.agent_version", "value": {"stringValue": agent_version}},
                ]},
                "scopeSpans": [{"spans": []}],
            }
        ]
    }


# ===========================================================================
# _gate — reads server-computed diff output only, never scores anything.
# ===========================================================================

def test_gate_passes_clean_diff():
    diff = {"counts": {"regressed": 0}, "control_determinations": []}
    failed, reasons = _gate(diff, ("regressed", "ineffective"))
    assert failed is False
    assert reasons == []
    print("  ✓ a clean diff (no regressions, no ineffective controls) passes the gate")


def test_gate_fails_on_regression_when_asked():
    diff = {"counts": {"regressed": 2}, "control_determinations": None}
    failed, reasons = _gate(diff, ("regressed",))
    assert failed is True
    assert "2 validator" in reasons[0]
    print("  ✓ regressed count > 0 fails the gate when --fail-on regressed")


def test_gate_ignores_regression_when_not_asked():
    diff = {"counts": {"regressed": 3}, "control_determinations": None}
    failed, reasons = _gate(diff, ("ineffective",))
    assert failed is False
    print("  ✓ --fail-on ineffective alone does not fail on a validator regression")


def test_gate_fails_on_ineffective_control():
    diff = {
        "counts": {"regressed": 0},
        "control_determinations": [
            {"control_id": "AC-002", "before": "EFFECTIVE", "after": "INEFFECTIVE"},
            {"control_id": "AC-003", "before": "EFFECTIVE", "after": "EFFECTIVE"},
        ],
    }
    failed, reasons = _gate(diff, ("ineffective",))
    assert failed is True
    assert "AC-002" in reasons[0]
    assert "AC-003" not in reasons[0]
    print("  ✓ an INEFFECTIVE control fails the gate, an EFFECTIVE one does not")


def test_gate_handles_missing_control_determinations():
    """No diff available, or nothing to compare — diff_service
    returns `None`, not an empty list (see ExecutionDiffOut's own
    docstring); the gate must treat that as 'nothing to report', not
    crash."""
    diff = {"counts": {"regressed": 0}, "control_determinations": None}
    failed, reasons = _gate(diff, ("regressed", "ineffective"))
    assert failed is False
    print("  ✓ control_determinations=None (not merged/not comparable) does not crash the gate")


def test_default_fail_on_is_both_conditions():
    args = build_parser().parse_args([
        "--campaign", "c1", "--agent-version", "v1", "--bundle", "b.json", "--trace", "t.json",
        "--base-url", "https://x", "--tenant", "acme", "--api-key", "k", "--device-key", "d.pem",
    ])
    assert args.fail_on is None, "None means 'apply the default', resolved in main() as both conditions"
    print("  ✓ --fail-on defaults to unset (main() resolves that to both conditions)")


# ===========================================================================
# _junit_xml — one testcase per validator the sealed diff compares.
# ===========================================================================

def test_junit_reports_a_regressed_validator_as_a_failure():
    diff = {
        "against_execution_id": "ex_before",
        "validators": [
            {"validator_id": "tool_authorization", "before": "PASS", "after": "FAIL", "change": "regressed"},
            {"validator_id": "data_exfiltration", "before": "PASS", "after": "PASS", "change": "unchanged"},
        ],
    }
    xml_text = _junit_xml(execution_id="ex_after", campaign_id="c1", diff=diff, failed=True, reasons=["1 regressed"])
    root = ET.fromstring(xml_text)
    assert root.tag == "testsuite"
    assert root.get("tests") == "2"
    assert root.get("failures") == "1"
    cases = {c.get("name"): c for c in root.findall("testcase")}
    assert cases["tool_authorization"].find("failure") is not None
    assert cases["data_exfiltration"].find("failure") is None
    print("  ✓ JUnit XML marks exactly the regressed validator as a <failure>")


def test_junit_without_a_diff_is_still_valid_xml():
    xml_text = _junit_xml(execution_id="ex1", campaign_id="c1", diff=None, failed=False, reasons=["no baseline yet"])
    root = ET.fromstring(xml_text)  # must not raise
    assert root.get("tests") == "0"
    assert root.find("system-out").text == "no baseline yet"
    print("  ✓ a missing diff (no baseline) still produces valid, non-empty JUnit XML")


# ===========================================================================
# _load_trace — a single file, a directory of files, and the same-identity
# guard across files.
# ===========================================================================

def test_load_trace_single_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        path.write_text(json.dumps(_otlp_payload("agent_1", "1.0.0")), encoding="utf-8")
        trace = _load_trace(path)
        assert trace["agent_id"] == "agent_1"
        assert trace["agent_version"] == "1.0.0"
    print("  ✓ --trace pointing at one file loads that file's trace")


def test_load_trace_directory_merges_same_agent():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.json").write_text(json.dumps(_otlp_payload("agent_1")), encoding="utf-8")
        (d / "b.json").write_text(json.dumps(_otlp_payload("agent_1")), encoding="utf-8")
        trace = _load_trace(d)
        assert trace["agent_id"] == "agent_1"
        assert trace["events"] == []
    print("  ✓ --trace pointing at a directory merges every file's events")


def test_load_trace_directory_rejects_mixed_agents():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.json").write_text(json.dumps(_otlp_payload("agent_1")), encoding="utf-8")
        (d / "b.json").write_text(json.dumps(_otlp_payload("agent_2")), encoding="utf-8")
        try:
            _load_trace(d)
        except CiUsageError as exc:
            assert "agent_1" in str(exc) and "agent_2" in str(exc)
        else:
            raise AssertionError("expected CiUsageError for a directory mixing two agent identities")
    print("  ✓ a trace directory mixing two agents' exports is rejected, not silently merged")


def test_load_trace_missing_file_is_a_usage_error():
    try:
        _load_trace(Path("/does/not/exist/trace.json"))
    except CiUsageError:
        pass
    else:
        raise AssertionError("expected CiUsageError for a nonexistent trace path")
    print("  ✓ a nonexistent --trace path is a clean usage error, not a raw traceback")


if __name__ == "__main__":
    import sys

    failures = 0
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  ✗ {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
