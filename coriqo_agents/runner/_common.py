"""Helpers shared between `runner run` (`__main__.py`) and `runner ci`
(`ci.py`). Kept in one place after the two copies of `_load_device_key`
drifted — one raised `CiUsageError`, the other called `SystemExit` — so a
fix to one no longer silently misses the other.
"""
from __future__ import annotations

from pathlib import Path


class DeviceKeyError(Exception):
    """The device key at a given path could not be loaded. Each caller
    catches this and converts it to its own preferred error path
    (`CiUsageError` in `ci.py`, `SystemExit` in `__main__.py`) rather than
    letting every entrypoint share one exception type end-to-end."""


def load_device_key(path: Path):
    """Load the Ed25519 private key issued at device enrollment.

    Needs the `runner` extra (`pip install coriqo-agents[runner]`), which is
    where `cryptography` lives — the import is inside the function so
    `--dry-run` (and any other path that never signs anything) works on a
    host that never installed it.

    Raises `DeviceKeyError` on any failure; never raises `SystemExit` or a
    tool-specific exception itself, since both callers need to convert the
    failure into their own idiom.
    """
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except ImportError as exc:
        raise DeviceKeyError(
            "signing needs the runner extra: pip install 'coriqo-agents[runner]'"
        ) from exc
    try:
        return load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError) as exc:
        raise DeviceKeyError(f"the device key at {path} could not be loaded: {exc}") from exc
