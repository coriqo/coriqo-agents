"""FailureScenario payload builder — the customer-side half of the
FailureScenario wiring. This package has no database access (it runs in the
customer environment; the `failure_scenarios` table lives in Coriqo's own
Postgres), so a validator cannot create a FailureScenario ROW itself. What
it CAN do, and what the server actually needs from the runner, is produce the same
fields that table stores — category/threat/trigger/expected_behavior/
forbidden_behavior/observed_behavior/severity — as a plain dict alongside
its ValidatorResult, so the ingestion path (server-side, where uploaded
results already become ValidatorResult rows) has everything it needs to create the FailureScenario row without
the validator re-deriving that narrative from the raw result later.

Kept as one small builder function, not a class, because every validator
that produces one needs exactly the same seven fields and nothing else.
"""
from __future__ import annotations

from typing import Any


def build_failure_scenario(
    *,
    category: str,
    threat: str,
    trigger: str,
    expected_behavior: str,
    forbidden_behavior: str,
    observed_behavior: str,
    severity: str | None,
    test_strategy: str | None = None,
    reproducibility: str = "always",
) -> dict[str, Any]:
    """The FailureScenario field set, as a dict. `execution_id`,
    `adversarial_run_id`, and `validator_result_id` are NOT included here —
    those are ids the runner either doesn't have (validator_result_id is
    allocated server-side on ingestion) or that belong to caller context
    (execution_id/adversarial_run_id), not to the validator's own finding."""
    return {
        "category": category,
        "threat": threat,
        "trigger": trigger,
        "expected_behavior": expected_behavior,
        "forbidden_behavior": forbidden_behavior,
        "observed_behavior": observed_behavior,
        "severity": severity,
        "test_strategy": test_strategy,
        "reproducibility": reproducibility,
    }
