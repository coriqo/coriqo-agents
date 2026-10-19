"""The validator contract.

    class Validator:
        def execute(self, context): ...
        def evaluate(self, result): ...
        def produce_evidence(self, result): ...

`execute` runs the probe against the target system and returns whatever raw
observation the validator needs (tool-call traces, response transcripts —
validator-specific, hence `Any`). `evaluate` turns that observation into a
`ValidatorResult` — the standardized shape every validator produces
regardless of what it tests. `produce_evidence` returns the evidence
payloads (dicts) this result should be backed by; the engine is responsible
for getting those uploaded/hashed, not the validator.

Kept as three separate methods (not one `run()`) because the deterministic-
evaluation principle draws its line exactly at the execute/evaluate boundary:
`execute` may do anything (including, for a future content-quality
validator, call an LLM) but `evaluate` must be deterministic code, no
exceptions, for every agent-behavioral validator.
Splitting the methods makes that boundary something a reviewer can see in
the class, not just something documented in prose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Status:
    """The six result statuses. A plain namespace of string constants
    rather than an enum.Enum: the wire format (JSON manifest, JSON result)
    carries plain strings, and the server's CheckConstraint
    (ck_validator_results_status) is the single source of truth for which
    values are valid — this class exists so a validator author gets
    autocomplete and a typo shows up as an AttributeError here rather than
    a 400 at upload time three steps later."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"

    ALL = (PASS, FAIL, WARNING, NOT_APPLICABLE, INCONCLUSIVE, ERROR)


class Severity:
    """The severity scale, present only on FAIL/WARNING results in
    practice — PASS and NOT_APPLICABLE results normally leave this None."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    ALL = (LOW, MEDIUM, HIGH, CRITICAL)


@dataclass
class ValidatorContext:
    """What a validator needs to probe the target agent. Deliberately thin:
    validators may need to extend this with whatever agent-invocation handle their probes actually use
    (a live agent endpoint, a recorded trajectory to replay against,
    credentials scoped to the assurance run). Fields here are the ones
    every validator needs regardless of what it tests.
    """
    execution_id: str
    agent_model_id: str
    agent_version_ref: str | None
    environment: str | None
    configuration: dict[str, Any] = field(default_factory=dict)
    # Free-form bag for a validator's own inputs (test datasets, attack
    # payload sets) — hashed by the engine into the manifest's
    # `input_dataset_hashes`, never sent to Coriqo's servers raw.
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatorResult:
    """The standardized result object, field for field:

        {
          "validator_id": "tool_authorization",
          "version": "1.2.0",
          "status": "FAIL",
          "score": 0.42,
          "severity": "CRITICAL",
          "observations": [],
          "evidence_ids": ["ev_123", "ev_124"]
        }

    `evidence_ids` is populated by the engine after it uploads whatever
    `produce_evidence` returned — a validator itself has no evidence-id
    namespace to allocate from, since that only exists once Coriqo has
    ingested the evidence. A validator's own `evaluate()` should leave it
    empty and let the engine fill it in.
    """
    validator_id: str
    version: str
    status: str
    score: float | None = None
    severity: str | None = None
    observations: list[Any] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "version": self.version,
            "status": self.status,
            "score": self.score,
            "severity": self.severity,
            "observations": self.observations,
            "evidence_ids": self.evidence_ids,
        }


class Validator:
    """Base class every validator implements. See module docstring for why
    the three methods stay separate."""

    #: Stable identity for this validator, e.g. "tool_authorization" — must
    #: match one of the protocol bundle's declared validator ids for the
    #: engine to run it (RunnerEngine.run raises otherwise).
    validator_id: str = ""
    #: Semantic version of THIS validator's logic, independent of the
    #: protocol version it ships under (per-validator versioning).
    version: str = "0.0.0"

    def __init__(self, bundle: dict | None = None) -> None:
        """`bundle` is the compiled protocol bundle this validator is running
        under, so a validator can read declarative test material the server
        pinned into it — currently the generated attack-scenario corpus
        (`template.protocol.generated_attack_scenarios`), produced
        server-side at compile time so the runner keeps zero LLM
        dependencies.

        Optional and defaulted, so `Cls()` still constructs every validator
        that has no use for it, and no existing caller changes.

        Boundary: bundle content is DATA — payload strings and keyword
        material a validator's existing matching logic consumes. It never
        supplies logic, never changes how a status is decided, and cannot
        turn a check off. `evaluate()` stays deterministic code either way.
        """
        self.bundle = bundle or {}

    def execute(self, context: ValidatorContext) -> Any:
        """Run the probe. Returns whatever raw observation `evaluate` needs.
        Must be implemented by every concrete validator."""
        raise NotImplementedError

    def evaluate(self, result: Any) -> ValidatorResult:
        """Turn `execute`'s raw observation into the standardized result.
        Deterministic for every agent-behavioral validator: no LLM
        call belongs in this method's implementation."""
        raise NotImplementedError

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        """Evidence payloads (as dicts) backing this result. May return an
        empty list — not every validator run produces standalone evidence
        beyond the result itself."""
        return []

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        """FailureScenario payload (see failure_scenario.py's
        build_failure_scenario) for a FAIL result, None otherwise. Base
        implementation returns None — not every validator's failure mode
        maps to one named FailureScenario category; a validator opts in by
        overriding this (tool_authorization, prompt_injection,
        data_exfiltration currently do). The engine calls this
        unconditionally for every result and only forwards a non-None
        return, so a validator that doesn't override it costs nothing."""
        return None
