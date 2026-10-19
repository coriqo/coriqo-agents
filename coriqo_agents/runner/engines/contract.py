"""The engine adapter contract — how a third-party red-team tool plugs into
the runner (Promptfoo was the first adapter).

Where this runs, and why it matters
-----------------------------------
Customer-side, inside the runner, never on Coriqo's servers.
Banks will not let a vendor's SaaS fire at their AI, so Coriqo never invokes
Promptfoo or any other engine. What crosses the wire is only what `normalize`
produced plus a hash of what `raw_artifact` returned — the normalized results,
the engine id and version, and a raw artifact stored as evidence.

The four methods exist as four methods because they have four different
trust properties, and collapsing them would hide that:

    prepare(campaign_bundle) -> plan   translate a signed bundle into the
                                       engine's own config. Reads the bundle
                                       as data.
    run(plan) -> RawRun                execute. Arbitrary side effects, network
                                       calls, subprocesses.
    normalize(raw) -> [NormalizedResult]
                                       turn engine output into the standard
                                       result shape. MUST be deterministic:
                                       the same RawRun yields the same
                                       results, every time. This is where an
                                       adapter would be tempted to ask a model
                                       "did this pass?" — it must not (master
                                       plan rule 9). An engine that reports
                                       nothing decidable produces INCONCLUSIVE
                                       results, not a guess.
    raw_artifact(raw) -> bytes         the engine's own output, verbatim, for
                                       hashing and upload as evidence. Not
                                       reformatted, not summarised: the point
                                       is that a reviewer can compare the
                                       normalized results against what the
                                       tool actually emitted.

No implementation lives here. This module is the fixed interface every
adapter codes against, so a second adapter (PyRIT, Garak, Agent-Opfor) is
a new class rather than a change to the runner.

Pure stdlib, like every module in this package — it ships standalone to
customer environments and imports nothing from the server.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawRun:
    """Whatever the engine produced, in the engine's own terms.

    `output` is the parsed structure an adapter's `normalize` reads;
    `artifact` is the same content as the bytes that will be hashed and
    uploaded. Kept as two fields rather than one because the artifact must be
    byte-exact for its hash to mean anything, while `output` is free to be a
    convenience view of it.
    """
    engine_id: str
    engine_version: str
    output: Any = None
    artifact: bytes = b""
    # Command line, exit code, duration, engine config digest — enough for a
    # reviewer to see how the tool was invoked. Never credentials.
    invocation: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class NormalizedResult:
    """One engine result in Coriqo's terms — `ValidatorResult`'s fields plus
    the three the campaign layer needs.

    Mirrors contract.ValidatorResult rather than subclassing it because a
    normalized result carries provenance a native validator's result does not
    (`engine`) and campaign context a validator has no access to
    (`control_ids`, `failure_scenario`). `status` and `severity` take the same
    values as `contract.Status` / `contract.Severity`; the server's
    ck_validator_results_status remains the single source of truth.
    """
    validator_id: str
    version: str
    status: str
    score: float | None = None
    severity: str | None = None
    observations: list[Any] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    # Which engine produced this result — 'promptfoo', 'native', 'manual'.
    # Denormalized onto the result row server-side so a report can say "these
    # three results are unattested" per result, not per execution.
    engine: str = ""
    # Control ids this result bears on. An adapter MAY leave this empty: the
    # authoritative validator→control map is server-side (D5), and a
    # customer-side adapter cannot be trusted to decide control coverage.
    control_ids: list[str] = field(default_factory=list)
    # FailureScenario payload for a FAIL, None otherwise — same shape
    # `Validator.failure_scenario` returns.
    failure_scenario: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "version": self.version,
            "status": self.status,
            "score": self.score,
            "severity": self.severity,
            "observations": self.observations,
            "evidence_ids": self.evidence_ids,
            "engine": self.engine,
            "control_ids": self.control_ids,
            "failure_scenario": self.failure_scenario,
        }


class EngineAdapter(abc.ABC):
    """Base class every engine adapter implements.

    An ABC, not a duck-typed protocol: an adapter that silently skips
    `raw_artifact` would produce results with no re-checkable source, and a
    TypeError at construction is a much better place to find that out than a
    sealed package with nothing behind it.
    """

    #: Must be one of the engine ids Coriqo's API accepts. An unknown value is rejected at
    #: ingest by the executions' engine CHECK constraint.
    engine_id: str = ""
    #: The version of the underlying tool this adapter drove, resolved at
    #: runtime where possible (e.g. `promptfoo --version`) rather than
    #: hardcoded — it goes into the manifest as provenance.
    engine_version: str = "0.0.0"

    def __init__(self) -> None:
        # Same argument as the ABC itself: an adapter that never set
        # `engine_id` would run a whole campaign and fail at the far end,
        # when the server rejects the execution against
        # ck_assurance_executions_engine. Construction is a much better place
        # to find that out than a customer's terminal after the run.
        if not self.engine_id:
            raise ValueError(
                f"{type(self).__name__} must set engine_id to one of the "
                "server's ENGINE_IDS"
            )

    @abc.abstractmethod
    def prepare(self, campaign_bundle: dict[str, Any]) -> Any:
        """Translate the signed campaign/protocol bundle into whatever plan
        this engine executes (a Promptfoo config, a test-case list).

        Reads the bundle as DATA. It may supply payloads and targets; it must
        never supply logic, and nothing in it can turn a check off."""

    @abc.abstractmethod
    def run(self, plan: Any) -> RawRun:
        """Execute the plan against the target and return the raw output.

        The only method allowed side effects."""

    @abc.abstractmethod
    def normalize(self, raw: RawRun) -> list[NormalizedResult]:
        """Turn raw engine output into standard results. Deterministic — the
        same RawRun must always produce the same list. No LLM call belongs in
        this method's implementation."""

    @abc.abstractmethod
    def raw_artifact(self, raw: RawRun) -> bytes:
        """The engine's own output, verbatim, for hashing and upload as
        evidence."""


#: Default length for a one-line summary of a prompt, response or transcript
#: turn. Adapters override it where their material is longer — a chat turn
#: carries more context than a single test case's grading reason.
DEFAULT_SUMMARY_CHARS = 160


def summarize(text: str, limit: int = DEFAULT_SUMMARY_CHARS) -> str:
    """Collapse whitespace and cut to `limit`, with an ellipsis when cut.

    Lives here, in the module both adapters already import, because both
    needed exactly this and each kept a copy differing only in the constant.
    Two copies of a formatter that feeds OBSERVATIONS is two places for the
    manifest's per-result hashes to start disagreeing between adapters about
    what the same text summarises to.

    A length limit, not a redaction: observations land in the sealed package
    and in the manifest's per-result hash, while the full text stays in the
    raw artifact, which is stored separately and can be access-controlled.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def declared_version(raw: RawRun, validator_id: str) -> str:
    """The bundle's declared version for this validator.

    `NormalizedResult.version` is the VALIDATOR's version — the server
    stores it as `validator_results.validator_version` and cross-checks
    nothing, so stamping an engine's own release there would make every
    result contradict the same manifest's own `validators` list. Falls back
    to `"0.0.0"` rather than to the tool's version: an unknown validator
    version is better recorded as unknown than as a number that means
    something else.

    Moved here  after four adapters — promptfoo, pyrit, garak,
    inspect — had each kept a byte-identical copy differing in nothing at
    all, the same duplication `summarize` above was already created to stop
    for `_truncate`.
    """
    versions = raw.invocation.get("validator_versions")
    if isinstance(versions, dict):
        value = versions.get(validator_id)
        if isinstance(value, str) and value:
            return value
    return "0.0.0"
