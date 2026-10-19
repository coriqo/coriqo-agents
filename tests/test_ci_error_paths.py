"""Every exception the HTTP layer or diff
payload can raise must land on the documented exit-code contract instead of
a raw Python traceback (default exit code 1, no "GATE PASSED" guarantee).

Three cases, one per finding:

  1. `CiApiError` raised while polling `GET /assurance/{id}` for completion
     (a rotated key, a transient 500, or a 404 mid-poll) used to escape
     `main()`'s exception handling entirely — nothing in `main()` caught
     `CiApiError` around `_wait_for_completion`. Now it does, and routes to
     `EXIT_GATE_UNAVAILABLE`, same as every other "verdict never retrieved"
     path.
  2. A 2xx response whose body isn't JSON (a proxy/WAF challenge page)
     raised `json.JSONDecodeError` out of `_CiApiClient._parse`, which is
     not a `CiApiError` and so also escaped `main()`. `_parse` now converts
     that into a `CiApiError` carrying the status code, so it flows through
     the same path as any other API failure.
  3. A control determination missing `control_id` raised a bare `KeyError`
     out of `_gate()`. It's now treated as an unresolved/failing
     determination (fail closed — a determination that can't be tied to a
     control isn't trusted) rather than crashing.

Runs standalone, same convention as its siblings in this directory:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_ci_error_paths.py
"""
from __future__ import annotations

import sys

from coriqo_agents.runner import ci as ci_module

from _ci_test_helpers import HAVE_CRYPTOGRAPHY, FakeResponse, run_ci

EXECUTION_ID = "ex_ci_errpath_1"


def test_api_error_during_polling_fails_closed():
    """Finding 2: a `CiApiError` raised out of `_wait_for_completion` (a
    401/500/404 mid-poll) must not escape `main()` as a raw traceback."""
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, _ = run_ci(
        ci_module, execution_id=EXECUTION_ID,
        poll_behavior=lambda url: FakeResponse(401, {"detail": "invalid API key"}),
    )
    assert exit_code == ci_module.EXIT_GATE_UNAVAILABLE, f"expected EXIT_GATE_UNAVAILABLE, got {exit_code}: {output}"
    assert "GATE PASSED" not in output
    print("  ✓ a CiApiError while polling for completion fails closed, never prints GATE PASSED")


def test_non_json_2xx_response_fails_closed():
    """Finding 3: a 2xx diff response whose body isn't JSON must not raise
    `json.JSONDecodeError` straight out of `main()`."""
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, _ = run_ci(
        ci_module, execution_id=EXECUTION_ID,
        diff_behavior=lambda url: FakeResponse(200, raw_text="<html>proxy challenge page</html>"),
    )
    assert exit_code == ci_module.EXIT_GATE_UNAVAILABLE, f"expected EXIT_GATE_UNAVAILABLE, got {exit_code}: {output}"
    assert "GATE PASSED" not in output
    print("  ✓ a non-JSON 2xx diff response fails closed, never prints GATE PASSED")


def test_missing_control_id_fails_the_gate_not_crash():
    """Finding 4: a determination with no `control_id` must not raise
    `KeyError` out of `_gate()` — it's treated as an unresolved/failing
    determination instead."""
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, _ = run_ci(
        ci_module, execution_id=EXECUTION_ID,
        diff_behavior=lambda url: FakeResponse(200, {
            "against_execution_id": "ex_prev", "counts": {"regressed": 0},
            "control_determinations": [{"after": "INEFFECTIVE"}],  # no control_id
            "validators": [],
        }),
    )
    assert exit_code == ci_module.EXIT_GATE_FAILED, f"expected EXIT_GATE_FAILED, got {exit_code}: {output}"
    assert "GATE PASSED" not in output
    assert "GATE FAILED" in output
    print("  ✓ a determination missing control_id fails the gate instead of raising KeyError")


def test_missing_counts_field_fails_closed_not_crash():
    """Final review round finding: a 2xx diff body with no usable `counts`
    field must route to `EXIT_GATE_UNAVAILABLE`, not raise `KeyError` out of
    `_gate()`."""
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, _ = run_ci(
        ci_module, execution_id=EXECUTION_ID,
        diff_behavior=lambda url: FakeResponse(200, {
            "against_execution_id": "ex_prev", "control_determinations": [], "validators": [],
        }),
    )
    assert exit_code == ci_module.EXIT_GATE_UNAVAILABLE, f"expected EXIT_GATE_UNAVAILABLE, got {exit_code}: {output}"
    assert "GATE PASSED" not in output
    print("  ✓ a diff body missing counts fails closed instead of raising KeyError")


if __name__ == "__main__":
    from _ci_test_helpers import run_tests

    sys.exit(run_tests(globals()))
