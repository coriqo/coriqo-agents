"""MODEL slice: shared input shape and helpers.

Every MODEL validator reads `context.inputs["model"]`, results the customer
computed on their own held-out set and validation report — Coriqo never
scores a model or holds the data, only re-derives a verdict from numbers the
customer already produced:

    {
      "performance": {                 # model_performance_threshold
        "metric": str, "observed": float, "threshold": float,
        "direction": "gte" | "lte",     # observed must clear threshold this way
      },
      "fairness": {                     # model_fairness_parity
        "metric": str,
        "group_values": {group_name: float},
        "tolerance": float,             # max allowed gap between any two groups
      },
      "drift": {                        # model_drift_psi
        "feature_psi": {feature_name: float},   # customer-computed PSI per feature
        "threshold": float,
      },
      "reproducibility": {              # model_reproducibility
        "sample": [{"id": str, "stored_prediction": float, "recomputed_prediction": float}],
        "tolerance": float,
      },
    }

A missing top-level key means "not supplied" (INCONCLUSIVE — cannot assess
that check), distinct from a supplied-but-empty collection (NOT_APPLICABLE —
nothing to check), the same split every other MODEL/RAG/GenAI validator draws.
"""
from __future__ import annotations

from typing import Any


def model_input(context) -> dict[str, Any] | None:
    """The `inputs["model"]` dict, or None when the run carried none."""
    model = (context.inputs or {}).get("model")
    return model if isinstance(model, dict) else None


def clears_threshold(observed: float, threshold: float, direction: str) -> bool:
    """True when `observed` clears `threshold` in the stated `direction`.
    "gte" for a metric where higher is better (accuracy, AUC), "lte" for one
    where lower is better (error rate, loss). Raises ValueError — never a
    bare TypeError — for anything a malformed rule could produce: a
    missing/non-numeric observed or threshold, or an unrecognized
    direction, so every caller can catch the one exception type and read
    it as INCONCLUSIVE rather than crashing the run."""
    if not isinstance(observed, (int, float)) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"model_performance_threshold: observed={observed!r} and threshold={threshold!r} "
            "must both be numbers"
        )
    if direction == "gte":
        return observed >= threshold
    if direction == "lte":
        return observed <= threshold
    raise ValueError(f"model_performance_threshold: unknown direction {direction!r}")
