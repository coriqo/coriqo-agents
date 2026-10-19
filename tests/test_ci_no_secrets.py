"""No secret may appear in anything
`coriqo-agents runner ci` prints or writes.

Runs the CLI's real `main()` end to end with a fake API key and a real
(freshly generated, throwaway) device key, against a fully monkeypatched
`httpx` — no network, no server needed — then asserts the fake secret
string never appears in captured stdout, stderr, the JSON payload, or the
JUnit file it writes. This is the explicit automated test for that rule.

Requires the `runner` extra (`cryptography`) — skips cleanly if it is not
installed, the same posture `_load_device_key` takes in ci.py itself.

Run standalone:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_ci_no_secrets.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

from coriqo_agents.runner import ci as ci_module

from _ci_test_helpers import (
    HAVE_CRYPTOGRAPHY, FakeResponse, Monkeypatch, bundle, install_fake_transport, write_device_key,
)

FAKE_API_KEY = "sk-fp06-super-secret-do-not-leak-8f92a1"
FAKE_PIPELINE_ID = "run-777"
EXECUTION_ID = "ex_ci_1"


def _regressed_diff(url: str) -> FakeResponse:
    return FakeResponse(200, {
        "execution_id": EXECUTION_ID, "against_execution_id": "ex_before",
        "validators": [
            {"validator_id": "tool_authorization", "before": "PASS", "after": "FAIL", "change": "regressed"},
        ],
        "counts": {"regressed": 1, "improved": 0, "unchanged": 0, "new": 0, "dropped": 0},
        "control_determinations": None,
    })


def test_secrets_never_appear_in_stdout_stderr_json_or_junit():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    mp = Monkeypatch()
    try:
        calls = install_fake_transport(mp, ci_module, execution_id=EXECUTION_ID, diff_behavior=_regressed_diff)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            bundle_path = d / "bundle.json"
            bundle_path.write_text(json.dumps(bundle()), encoding="utf-8")
            trace_path = d / "trace.json"
            trace_path.write_text(json.dumps({"resourceSpans": []}), encoding="utf-8")
            device_key_path = d / "device.pem"
            write_device_key(device_key_path)
            junit_path = d / "results.xml"

            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = ci_module.main([
                    "--campaign", "c1", "--agent-version", "v1",
                    "--bundle", str(bundle_path), "--trace", str(trace_path),
                    "--base-url", "https://coriqo.example", "--tenant", "acme_bank",
                    "--api-key", FAKE_API_KEY, "--device-key", str(device_key_path),
                    "--pipeline-id", FAKE_PIPELINE_ID,
                    "--junit", str(junit_path), "--json",
                ])

            captured = stdout.getvalue() + stderr.getvalue()
            assert FAKE_API_KEY not in captured, "the API key leaked into stdout/stderr"
            junit_text = junit_path.read_text(encoding="utf-8")
            assert FAKE_API_KEY not in junit_text, "the API key leaked into the JUnit report"

            device_key_pem_body = device_key_path.read_text(encoding="utf-8")
            assert device_key_pem_body not in captured, "the device key PEM leaked into stdout/stderr"
            assert device_key_pem_body not in junit_text, "the device key PEM leaked into the JUnit report"

            # The gate genuinely ran (regression -> exit 1) and produced
            # real output — a no-op that printed nothing would pass this
            # test for the wrong reason.
            assert exit_code == 1
            payload = json.loads(captured)
            assert payload["gate_failed"] is True
            assert ("GET", "https://coriqo.example/api/v1/assurance/campaigns/c1") in calls
            assert any(kind == "UPLOAD" for kind, _ in calls)
    finally:
        mp.undo()
    print("  ✓ the API key and device key never appear in stdout, stderr, JSON, or JUnit output")


if __name__ == "__main__":
    from _ci_test_helpers import run_tests

    sys.exit(run_tests(globals()))
