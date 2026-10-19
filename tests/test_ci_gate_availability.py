"""The sealed diff fetch (`GET /assurance/{id}/diff`) must
fail CLOSED on anything other than the documented 409 "no baseline yet"
shape.

Before this fix, `_CiApiClient._parse` raised a generic `RuntimeError` for
every 4xx/5xx response, and `ci.main()` caught that generic `RuntimeError`
around the diff fetch and always treated it as "no baseline yet" — so a
rotated API key (401), a transient server error (500), or a network blip
all silently printed "GATE PASSED" with exit 0. `CiApiError` now carries
the real status code, and `main()` branches explicitly: 409 keeps the
documented no-baseline pass-through, everything else returns
`EXIT_GATE_UNAVAILABLE` and never prints "GATE PASSED".

Runs standalone, same convention as its sibling files in this directory:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_ci_gate_availability.py
"""
from __future__ import annotations

import sys

import httpx

from coriqo_agents.runner import ci as ci_module

from _ci_test_helpers import HAVE_CRYPTOGRAPHY, FakeResponse, run_ci

EXECUTION_ID = "ex_ci_avail_1"


def test_409_is_the_documented_no_baseline_pass_through():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, _ = run_ci(
        ci_module, execution_id=EXECUTION_ID,
        diff_behavior=lambda url: FakeResponse(409, {"detail": "no baseline yet"}),
    )
    assert exit_code == ci_module.EXIT_OK, f"expected EXIT_OK, got {exit_code}: {output}"
    assert "GATE PASSED" in output
    print("  ✓ 409 on the diff endpoint stays the documented no-baseline pass-through")


def test_401_fails_closed_not_gate_passed():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, _ = run_ci(
        ci_module, execution_id=EXECUTION_ID,
        diff_behavior=lambda url: FakeResponse(401, {"detail": "invalid API key"}),
    )
    assert exit_code == ci_module.EXIT_GATE_UNAVAILABLE, f"expected EXIT_GATE_UNAVAILABLE, got {exit_code}: {output}"
    assert "GATE PASSED" not in output
    print("  ✓ 401 on the diff endpoint fails closed, never prints GATE PASSED")


def test_500_fails_closed_not_gate_passed():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return
    exit_code, output, _ = run_ci(
        ci_module, execution_id=EXECUTION_ID,
        diff_behavior=lambda url: FakeResponse(500, {"detail": "internal error"}),
    )
    assert exit_code == ci_module.EXIT_GATE_UNAVAILABLE, f"expected EXIT_GATE_UNAVAILABLE, got {exit_code}: {output}"
    assert "GATE PASSED" not in output
    print("  ✓ 500 on the diff endpoint fails closed, never prints GATE PASSED")


def test_network_error_fails_closed_not_gate_passed():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped — cryptography not installed)")
        return

    def _raise(url):
        raise httpx.ConnectTimeout("connect timed out")

    exit_code, output, _ = run_ci(ci_module, execution_id=EXECUTION_ID, diff_behavior=_raise)
    assert exit_code == ci_module.EXIT_GATE_UNAVAILABLE, f"expected EXIT_GATE_UNAVAILABLE, got {exit_code}: {output}"
    assert "GATE PASSED" not in output
    print("  ✓ a network/timeout error on the diff endpoint fails closed, never prints GATE PASSED")


if __name__ == "__main__":
    from _ci_test_helpers import run_tests

    sys.exit(run_tests(globals()))
