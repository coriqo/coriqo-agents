"""`EnginePlan` — what an adapter's `prepare()` hands to its `run()`, and the
only description of the run a reviewer ever gets.

The adapter contract fixes `EngineAdapter.prepare` as returning `Any`, so each adapter is
free to return its own plan object. Promptfoo and manual both return
this one, because both need the same three things and the manifest needs
them in the same shape:

  * the engine-native configuration, so `plan_hash` means something;
  * the validators the run intends to cover, so a crashed run can still
    report one result per intended validator instead of reporting nothing;
  * the out-of-scope list, which is a disclosure obligation whether or not
    the engine can enforce it (see `PromptfooAdapter.prepare`).

`plan_hash` is what goes into the signed manifest. It hashes the config the
adapter actually wrote, not a summary of it: the claim a reader needs to be
able to check is "this run was configured this way", and a hash over a
prettied-up description of the config cannot support that.

Pure stdlib, like every module in this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..manifest import canonical_bytes, sha256_hex


@dataclass
class EnginePlan:
    """One prepared engine run."""

    engine_id: str
    #: [{"id": ..., "version": ...}] — the validators this run intends to
    #: produce a result for, in order.
    validators: list[dict[str, str]] = field(default_factory=list)
    #: The engine's own configuration, as the structure that was written to
    #: disk. Never contains a secret: a target's credentials are referenced
    #: by the NAME of an environment variable the customer sets, and the
    #: name is what is written here (a campaign target never holds a
    #: secret).
    config: dict[str, Any] = field(default_factory=dict)
    #: Absolute path of the config file written into the workdir, if any.
    config_path: str | None = None
    workdir: str | None = None
    #: Scope exclusions carried from the campaign. Always recorded, whether
    #: or not the engine can act on them — an examiner reading the package
    #: needs to see what was declared off limits, not only what the tool
    #: happened to support.
    out_of_scope: list[str] = field(default_factory=list)

    def plan_hash(self) -> str:
        """sha256 over the canonical bytes of the config, validator list and
        exclusions. Goes into the manifest as `engine.plan_hash`."""
        return sha256_hex(canonical_bytes({
            "engine_id": self.engine_id,
            "validators": self.validators,
            "config": self.config,
            "out_of_scope": sorted(self.out_of_scope),
        }))

    def disclosure(self) -> dict[str, Any]:
        """The subset of the plan that is safe to show a reviewer verbatim.
        Used in observations when a run fails before it produces output."""
        return {
            "engine_id": self.engine_id,
            "validators": [v.get("id") for v in self.validators],
            "out_of_scope": sorted(self.out_of_scope),
            "plan_hash": self.plan_hash(),
        }
