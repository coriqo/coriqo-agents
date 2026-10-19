"""What this runner host can actually execute.

The campaign builder's Engines step has to tell three states apart, and
before this module it could only tell one:

  * the engine is installed on the device that runs this agent's campaigns;
  * the device has reported in, and this engine is NOT installed there;
  * no device has reported at all, so nothing is known either way.

Only the runner can answer the first two, because only the runner is on the
customer's host. So it reports them, in the manifest it already signs and
uploads, and the server reads them back out (`GET
/api/v1/assurance/runner-capabilities`). Nothing here is a determination:
this is a description of a machine, and the server stores it as one.

Deliberately probe-only. `report()` does not import PyRIT, does not run
`promptfoo --version`, and does not touch the network: a capability report
is collected on every run, including runs of an unrelated engine, and a
report that imported every optional dependency would make each of those
runs pay for dependencies it never uses. `importlib.util.find_spec` answers
"is it installed" without executing the package, and `shutil.which` answers
the same for a binary without starting it.

Pure stdlib, like every module in this package.
"""
from __future__ import annotations

import importlib.util
import shutil
from typing import Any

#: Optional distributions this SDK's extras install, and the extra that
#: installs each. Reported by NAME only — never a version, never a path:
#: the report ends up in a signed manifest and then in a tenant's database,
#: and a filesystem path from a customer's build host is not something a
#: campaign builder needs in order to grey out a checkbox.
_EXTRA_MODULES: dict[str, str] = {
    # `pip install coriqo-agents[runner]` — Ed25519 signing for the manifest.
    "runner": "cryptography",
    # `pip install coriqo-agents[pyrit]` — the PyRIT engine adapter.
    "pyrit": "pyrit",
    # The Inspect engine adapter. Garak has no entry here — it is a
    # customer-installed CLI, probed by `shutil.which` below, not a Python
    # distribution `find_spec` could answer for.
    "inspect": "inspect_ai",
}


def _module_installed(module: str) -> bool:
    """True when `module` is importable without importing it.

    `find_spec` itself raises on a package whose parent is missing or whose
    `__init__` is broken; both mean "not usable here", which is what the
    caller asked, so they are answered rather than propagated. A capability
    probe must never be the thing that fails a run.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def engine_availability() -> dict[str, dict[str, Any]]:
    """Per engine id, whether this host can run it and why/why not.

    The reason string is the point. "PyRIT: false" on its own reads, in a
    UI, as a broken checkbox; "the pyrit extra is not installed on this
    device" tells the reader what to do about it.
    """
    promptfoo_binary = shutil.which("promptfoo")
    pyrit_installed = _module_installed("pyrit")
    garak_binary = shutil.which("garak")
    inspect_installed = _module_installed("inspect_ai")
    return {
        "native": {
            "available": True,
            "reason": "the shipped validators run with no extra install",
        },
        "manual": {
            "available": True,
            "reason": "manual evidence needs no tool on this device",
        },
        "promptfoo": {
            "available": promptfoo_binary is not None,
            "reason": (
                "the promptfoo binary is on this device's PATH"
                if promptfoo_binary
                else "the promptfoo binary is not on this device's PATH. "
                     "Install the Node package to run it here."
            ),
        },
        "pyrit": {
            "available": pyrit_installed,
            "reason": (
                "the pyrit package is installed on this device (probed, not imported "
                "— a broken install can still fail at run time, and says so in the results)"
                if pyrit_installed
                else "pyrit is not installed on this device. "
                     "Install it with: pip install 'coriqo-agents[pyrit]'"
            ),
        },
        # `garak`/`inspect` joined the server's ENGINE_IDS in this same
        # branch (migration 0305 — see engines/__init__.py's module
        # docstring) — the comment used to say these two keys were dropped
        # by `capabilities_router.py._engines()` because the server didn't
        # know the ids yet, which was already stale by the time 0305 landed
        # in the same PR. Confirmed instead of assumed: `_engines()`
        # iterates `ENGINE_IDS` and looks up each id in this dict, so now
        # that both ids are in `ENGINE_IDS`, these two blocks ARE surfaced
        # to `GET /assurance/runner-capabilities` and the Engines step reads
        # them the same way it already reads `pyrit`'s.
        "garak": {
            "available": garak_binary is not None,
            "reason": (
                "the garak binary is on this device's PATH"
                if garak_binary
                else "the garak binary is not on this device's PATH. "
                     "Install it with: pip install garak"
            ),
        },
        "inspect": {
            "available": inspect_installed,
            "reason": (
                "the inspect-ai package is installed on this device (probed, not imported — "
                "a broken install can still fail at run time, and says so in the results)"
                if inspect_installed
                else "inspect-ai is not installed on this device. "
                     "Install it with: pip install 'coriqo-agents[inspect]'"
            ),
        },
    }


def report() -> dict[str, Any]:
    """The capability block embedded in the signed execution manifest.

    Shape is fixed and small: engines with a reason each, the validator ids
    this SDK ships, and which optional extras are installed. Any field a
    reader could mistake for a claim about the RUN (what executed, what
    passed) is deliberately absent — this describes the host, and the
    execution's own results describe the run.
    """
    from .validators import VALIDATOR_REGISTRY

    return {
        "engines": engine_availability(),
        "validators": sorted(VALIDATOR_REGISTRY),
        "extras": {
            extra: _module_installed(module) for extra, module in sorted(_EXTRA_MODULES.items())
        },
    }
