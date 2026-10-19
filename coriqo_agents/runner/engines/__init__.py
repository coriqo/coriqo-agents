"""Engine adapters — third-party red-team tools plugged into the runner.

The adapters: `PromptfooAdapter` and `ManualAdapter`; `PyritAdapter`, the
first library-shaped engine (multi-turn, in-process, optional dependency);
`GarakAdapter`
(CLI subprocess, shaped like Promptfoo) and `InspectAdapter` (in-process,
shaped like PyRIT but a trace source rather than a verdict source — see its
module docstring). A third candidate was evaluated, a third candidate, Agent-Opfor
(`@keyvaluesystems/agent-opfor-sdk`, Apache-2.0, actively maintained) and
deliberately shipped no adapter for it: its evaluator families fit
`tool_authorization`/`privilege_escalation` well on paper, but this
research could not confirm the evaluator's actual runtime output
JSON shape from the current source — a `normalize()` guessing at unconfirmed
keys is exactly the "wrapper that can't work" an adapter must not be: read the
actual API, or skip. Building it is a
follow-up once that shape is verified, not abandoned.

`GarakAdapter` and `InspectAdapter` are importable whether or not their
underlying tool/library is installed — neither module imports its optional
dependency at import time, so `ADAPTERS` is the same table on every host and
a missing dependency surfaces as a disclosed ERROR result at run time rather
than an ImportError here (the same property `PyritAdapter` already holds).

Neither `garak` nor `inspect` is in the server's `ENGINE_IDS`
yet. Both adapters are complete, tested and importable now, but a campaign
cannot select either engine server-side until Coriqo's API accepts them.

`ADAPTERS` is the CLI's `--engine` table. A name that is not in it is not
runnable, which is the point.
"""
from __future__ import annotations

from .contract import EngineAdapter, NormalizedResult, RawRun
from .garak import PINNED_GARAK_VERSION, PROBE_MAP as GARAK_PROBE_MAP, GarakAdapter
from .inspect import PINNED_INSPECT_VERSION, InspectAdapter, InspectUnavailableError
from .manual import ManualAdapter
from .plan import EnginePlan
from .promptfoo import PINNED_PROMPTFOO_VERSION, PLUGIN_MAP, PromptfooAdapter
from .pyrit import PINNED_PYRIT_VERSION, PyritAdapter, PyritUnavailableError

ADAPTERS = {
    PromptfooAdapter.engine_id: PromptfooAdapter,
    ManualAdapter.engine_id: ManualAdapter,
    PyritAdapter.engine_id: PyritAdapter,
    GarakAdapter.engine_id: GarakAdapter,
    InspectAdapter.engine_id: InspectAdapter,
}

__all__ = [
    "ADAPTERS",
    "EngineAdapter",
    "EnginePlan",
    "GARAK_PROBE_MAP",
    "GarakAdapter",
    "InspectAdapter",
    "InspectUnavailableError",
    "ManualAdapter",
    "NormalizedResult",
    "PINNED_GARAK_VERSION",
    "PINNED_INSPECT_VERSION",
    "PINNED_PROMPTFOO_VERSION",
    "PINNED_PYRIT_VERSION",
    "PLUGIN_MAP",
    "PromptfooAdapter",
    "PyritAdapter",
    "PyritUnavailableError",
    "RawRun",
]
