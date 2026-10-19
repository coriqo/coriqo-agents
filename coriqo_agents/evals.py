"""Eval-run and adversarial-run helpers: local artifact hashing, and the
declared-unit metric shape the server accepts.

Why local hashing is the default
--------------------------------
Coriqo never stores the raw inputs of an eval run — the dataset, the prompt
set, the attack corpus. It stores a sha256 of them and nothing else. A harness
that POSTs its whole dataset so the server can hash it has shipped that dataset
across the network, through whatever sits in front of the API, for a result it
could have computed in one line locally. `hash_eval_artifact` computes the same
digest the server would, so the request carries 64 hex characters instead.
"""
from __future__ import annotations

from typing import Any

from .hashing import hash_value

_SHA256_PREFIX = "sha256:"

METRIC_DIRECTIONS = ("higher_is_better", "lower_is_better")


def hash_eval_artifact(artifact_inputs: Any) -> str:
    """The sha256 the server would derive from the same `artifact_inputs`.

    Byte-identical to the server's canonical hash: both
    serialize with `json.dumps(obj, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False, default=str)` and sha256 the UTF-8 encoding. This
    returns the BARE hex digest, not the "sha256:"-prefixed form the
    tool-call hashes use, because `artifact_hash` is capped at 64 characters
    server-side and is compared as bare hex.

    tests/test_eval_runs.py asserts this equals the hash the server computes
    from the same object, so a change to either canonicalisation fails a test
    rather than silently producing two digests for one dataset.
    """
    digest = hash_value(artifact_inputs)
    if digest is None:
        raise ValueError("hash_eval_artifact needs something to hash — got None")
    return digest.removeprefix(_SHA256_PREFIX)


def eval_metric(
    metric: str,
    value: float,
    *,
    direction: str | None = None,
    scale: tuple[float, float] | dict | None = None,
    threshold: float | None = None,
    threshold_direction: str | None = None,
) -> dict:
    """One entry for `record_eval_run(metrics=[...])`.

    `metric` and `value` are all the server requires. The rest say what the
    number MEANS, and they matter more than they look: a harness reporting
    accuracy as 87 on a 0-100 scale, with no scale declared, is read as a
    score of 87 on a 0..1 metric and goes unjudged. Declare the unit and the
    eval-readiness check can compare it.

    - `direction` — "higher_is_better" or "lower_is_better". Overrides the
      server's guess from the metric name. A direction is not a unit; it does
      not on its own make a value outside 0..1 judgeable.
    - `scale` — the bounds you report in, as `(min, max)` or
      `{"min": ..., "max": ...}`. `(0, 100)` for a percentage.
    - `threshold` — an explicit bar in this metric's own units. Compared
      directly, with no normalization and no tenant floor. Wins over `scale`.
    - `threshold_direction` — only when the bar points the other way from
      `direction`. Falls back to `direction`.

    Keys you leave unset are omitted from the request rather than sent as
    null, so nothing here changes what a bare {metric, value} entry stores.
    """
    for name, given in (("direction", direction), ("threshold_direction", threshold_direction)):
        if given is not None and given not in METRIC_DIRECTIONS:
            raise ValueError(
                f"{name} must be one of {', '.join(METRIC_DIRECTIONS)} — got {given!r}"
            )
    entry: dict[str, Any] = {"metric": metric, "value": value}
    if direction is not None:
        entry["direction"] = direction
    if scale is not None:
        entry["scale"] = _normalize_scale(scale)
    if threshold is not None:
        entry["threshold"] = threshold
    if threshold_direction is not None:
        entry["threshold_direction"] = threshold_direction
    return entry


def _normalize_scale(scale: tuple[float, float] | dict) -> dict:
    """Accepts `(min, max)` or `{"min": ..., "max": ...}`; always sends the
    dict the server's EvalMetricScale expects. The ordering check runs here
    as well as server-side so a reversed pair costs a ValueError, not a 422
    on a request that already carried a whole eval run's results."""
    if isinstance(scale, dict):
        missing = {"min", "max"} - set(scale)
        if missing:
            raise ValueError(f"scale needs both min and max — missing {', '.join(sorted(missing))}")
        extra = set(scale) - {"min", "max"}
        if extra:
            raise ValueError(f"scale takes only min and max — remove {', '.join(sorted(extra))}")
        low, high = scale["min"], scale["max"]
    else:
        try:
            low, high = scale
        except (TypeError, ValueError):
            raise ValueError(
                'scale must be (min, max) or {"min": ..., "max": ...} — got ' + repr(scale)
            ) from None
    if high <= low:
        raise ValueError(f"scale max must be greater than min — got min={low}, max={high}")
    return {"min": low, "max": high}
