"""`--execution`: attach the runner to a run Coriqo already created (the id in a
run-trigger message) instead of creating a new one.

Runs standalone, same convention as its siblings in this directory:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_ci_attach_execution.py
"""
from __future__ import annotations

import sys

from coriqo_agents.runner import ci as ci_module

from _ci_test_helpers import HAVE_CRYPTOGRAPHY, FakeResponse, run_ci

EXECUTION_ID = "ex_ci_attach_1"


def test_attaches_to_a_waiting_run_without_creating_another():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    seen = {"n": 0}

    def poll(url):
        seen["n"] += 1
        status = "pending" if seen["n"] == 1 else "completed"
        return FakeResponse(200, {"id": EXECUTION_ID, "status": status})

    exit_code, output, calls = run_ci(
        ci_module, execution_id=EXECUTION_ID, poll_behavior=poll, extra_args=["--execution", EXECUTION_ID],
    )
    assert ("POST", "https://coriqo.example/api/v1/assurance/campaigns/c1/executions") not in calls, calls
    assert any(kind == "UPLOAD" for kind, _ in calls), "results must still be uploaded to the existing run"
    assert exit_code != ci_module.EXIT_USAGE_ERROR, output
    print("  ✓ --execution reuses the waiting run and uploads to it, no second run is created")


def test_refuses_a_run_that_already_finished():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, calls = run_ci(
        ci_module, execution_id=EXECUTION_ID, extra_args=["--execution", EXECUTION_ID],
    )  # default poll answers "completed"
    assert exit_code == ci_module.EXIT_USAGE_ERROR, f"{exit_code}: {output}"
    assert "only a run that is still waiting" in output
    assert not any(kind in ("POST", "UPLOAD") for kind, _ in calls), calls
    print("  ✓ --execution on a finished run is refused before anything is sent")


def test_without_the_flag_a_new_run_is_still_created():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    _, _, calls = run_ci(ci_module, execution_id=EXECUTION_ID)
    assert ("POST", "https://coriqo.example/api/v1/assurance/campaigns/c1/executions") in calls
    print("  ✓ unchanged default: no flag, the runner creates its own run")


if __name__ == "__main__":
    from _ci_test_helpers import run_tests

    sys.exit(run_tests(globals()))
