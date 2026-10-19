"""The execution engine — runs a compiled protocol bundle's validators
against a target and builds the signed manifest. This is the
customer-side half of "protocol compiled server-side, executed
customer-side": the engine never talks to Coriqo's servers to run a
validator, only to fetch the bundle beforehand and upload results after.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .contract import Validator, ValidatorContext, ValidatorResult
from .manifest import (
    build_evidence_envelope,
    build_manifest,
    evidence_content_id,
    raw_engine_artifact_item,
    sha256_hex,
)

from .capabilities import report as capability_report

RUNNER_VERSION = "0.1.0"  # this package's own version — recorded in every manifest


class UnknownValidatorError(ValueError):
    """A validator the caller passed in isn't declared in the bundle it
    claims to execute — running it would produce a manifest the server
    refuses (bundle/validators mismatch), so this fails fast, locally,
    before any network call."""


class InvalidBundleError(ValueError):
    """`--bundle` (or whatever dict was handed to `RunnerEngine`) isn't
    shaped like a compiled protocol bundle.

    Raised instead of letting a raw `KeyError`/`TypeError` propagate out of
    `__init__` when `bundle["template"]["protocol"]["validators"]` is
    missing or malformed, so `__main__.py`'s usage-error path (exit code 1,
    a clear message — see its module docstring) can catch this the same way
    it already catches every other bad-input case, rather than crashing with
    an unhandled traceback for what is just a bad `--bundle` file."""


class RunnerEngine:
    """Runs validators sequentially (no parallel execution needed at this
    scale) and produces the signed manifest alongside the results.

    `bundle` is what GET /api/v1/assurance/protocols/{id} (or the compile
    response) returns: {id, name, version, template, bundle_hash, ...}.
    """

    def __init__(self, bundle: dict, *, runner_version: str = RUNNER_VERSION):
        self.bundle = bundle
        self.runner_version = runner_version
        try:
            validators = bundle["template"]["protocol"]["validators"]
            self._declared_ids = {v["id"] for v in validators}
            # id/version/bundle_hash are read later by run()/run_engine()
            # to build the manifest — validate them here too, so a
            # malformed bundle fails once, up front, with the same clean
            # InvalidBundleError, instead of a raw KeyError deep inside
            # manifest-building after validators have already executed.
            bundle["id"]
            bundle["version"]
            bundle["bundle_hash"]
        except (KeyError, TypeError) as exc:
            raise InvalidBundleError(
                "the protocol bundle is missing "
                "template.protocol.validators[].id, id, version, or "
                "bundle_hash — is --bundle pointing at a compiled protocol "
                f"bundle? ({type(exc).__name__}: {exc})"
            ) from exc

    def run(
        self, validators: list[Validator], context: ValidatorContext,
    ) -> tuple[list[ValidatorResult], dict, dict[str, dict], list[dict]]:
        """Execute every validator in order, evaluate each, collect its
        evidence and any FailureScenario payload, and build the signed
        manifest. Raises UnknownValidatorError before running anything if a
        validator's id isn't part of this bundle.

        Returns `(results, manifest, evidence, failure_scenarios)`:
          - `evidence` maps each content-addressed evidence id (also
            embedded in the matching result's `evidence_ids`) to the
            envelope ExecutionUploader.upload_results should send alongside
            the results array. A validator that produces no evidence
            contributes nothing to this dict; that is normal, not an error
            (produce_evidence "may return an empty list").
          - `failure_scenarios` is a list of {"result_index": i, **payload}
            dicts — `result_index` is the position in `results` this
            scenario belongs to, since the runner has no ValidatorResult id
            to link to yet (allocated server-side on ingestion, per
            failure_scenario.py's own docstring). Only results whose
            validator overrides `failure_scenario()` and returns non-None
            appear here.

        A validator whose execute()/evaluate() raises is NOT swallowed —
        per the validator contract, an INCONCLUSIVE/ERROR result belongs in the standardized
        result object with the reason as an observation, not as an
        uncaught exception that aborts the whole run and reports nothing
        for the validators that already succeeded. Concrete validators
        are expected to catch their own probe failures and
        return ERROR/INCONCLUSIVE results; this engine only guards against
        a validator that skips that discipline entirely.
        """
        unknown = [v.validator_id for v in validators if v.validator_id not in self._declared_ids]
        if unknown:
            raise UnknownValidatorError(
                f"validator(s) {unknown} are not part of bundle {self.bundle.get('id')!r} "
                f"(declares: {sorted(self._declared_ids)})"
            )

        started_at = datetime.now(timezone.utc)
        results: list[ValidatorResult] = []
        evidence: dict[str, dict] = {}
        failure_scenarios: list[dict] = []
        for index, validator in enumerate(validators):
            try:
                raw = validator.execute(context)
                result = validator.evaluate(raw)
            except Exception as exc:  # noqa: BLE001 - deliberately broad: a crashing
                # validator must still produce a reportable result, not abort the run.
                from .contract import Status
                result = ValidatorResult(
                    validator_id=validator.validator_id,
                    version=validator.version,
                    status=Status.ERROR,
                    observations=[f"validator raised {type(exc).__name__}: {exc}"],
                )
                results.append(result)
                continue

            # Evidence/failure-scenario collection is deliberately OUTSIDE
            # the execute/evaluate try block above: a real, already-decided
            # result (a CRITICAL FAIL, say) must survive a downstream
            # produce_evidence()/failure_scenario() crash intact — the
            # opposite bug used to discard a genuine security finding
            # (downgrading it to a bare ERROR) AND could leave partial
            # evidence envelopes already written into `evidence` for the
            # same validator with no result actually claiming them, since
            # the crash could happen mid-loop after some envelopes were
            # already recorded. Collecting into LOCAL containers first and
            # only merging them into `evidence`/`result.evidence_ids` after
            # this validator's collection succeeds in full keeps both
            # failure modes from happening independent of each other.
            try:
                new_evidence: dict[str, dict] = {}
                evidence_ids: list[str] = []
                for raw_item in validator.produce_evidence(raw):
                    envelope = build_evidence_envelope(context.execution_id, validator.validator_id, raw_item)
                    evidence_id = evidence_content_id(envelope)
                    new_evidence[evidence_id] = envelope
                    evidence_ids.append(evidence_id)
                scenario = validator.failure_scenario(result)
            except Exception as exc:  # noqa: BLE001 - see comment above: don't lose `result`.
                result.observations.append(
                    f"evidence/failure_scenario collection raised {type(exc).__name__}: {exc} "
                    "— result stands, no evidence recorded for this validator"
                )
            else:
                evidence.update(new_evidence)
                result.evidence_ids.extend(evidence_ids)
                if scenario is not None:
                    failure_scenarios.append({"result_index": index, **scenario})
            results.append(result)
        completed_at = datetime.now(timezone.utc)

        manifest = build_manifest(
            runner_version=self.runner_version,
            protocol_id=self.bundle["id"],
            protocol_version=self.bundle["version"],
            bundle_hash=self.bundle["bundle_hash"],
            validators=[{"id": v.validator_id, "version": v.version} for v in validators],
            system_version_ref=context.agent_version_ref,
            configuration_snapshot=context.configuration,
            input_datasets=context.inputs,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
            trace_provenance=_trace_provenance(context),
            capabilities=_capabilities(),
        )
        return results, manifest, evidence, failure_scenarios

    def run_engine(
        self, adapter, context: ValidatorContext,
    ) -> tuple[list[ValidatorResult], dict, dict[str, dict], list[dict]]:
        """Run a third-party engine adapter and produce the same four
        things `run()` produces, so the uploader, the manifest and the server
        ingest path do not care which one ran.

        The four adapter methods are called once each, in order, and the
        engine does no interpreting between them: `normalize` decides every
        status, and nothing here second-guesses it.

        What this method adds on top of the adapter:

        * **A bundle check before anything is uploaded.** An adapter that
          normalized a result for a validator the bundle never declared
          would produce a manifest the server refuses; raising
          `UnknownValidatorError` here fails locally instead, exactly as
          `run()` does for native validators.
        * **The raw artifact as evidence.** `raw_artifact()`'s bytes are
          hashed, the hash goes into the manifest's `engine` block, and the
          bytes themselves go up as one evidence envelope of kind
          `raw_engine_artifact`. Both, not either: the hash is what a sealed
          manifest can attest to, and the bytes are what lets a reviewer
          check the normalized results against what the tool actually said.
        * **Order.** The artifact's evidence id is derived from its envelope
          the same content-addressed way every other evidence id is, so it is
          computable before any network round trip.

        A crash inside the adapter is NOT swallowed here. `run()` catches
        validator exceptions because a native run has many validators and one
        of them failing must not lose the others; an engine run is a single
        subprocess, and an adapter whose `run`/`normalize` raises has produced
        nothing to salvage. Adapters disclose their own failures as ERROR
        results instead (`PromptfooAdapter` does this for a missing binary,
        a timeout and a non-zero exit) — a raised exception here means the
        adapter itself is broken, which is not something to paper over with a
        signed manifest.
        """
        plan = adapter.prepare(self.bundle)
        # Fail fast, before any traffic reaches the target — mirroring
        # run()'s own order. Checking `plan.validators` (what the adapter
        # INTENDS to cover) rather than waiting for `normalize()`'s output
        # means a bundle mismatch is caught before `adapter.run(plan)` has
        # already fired real requests at the target; the post-normalize
        # check further below stays as a second guard for an adapter whose
        # actual output drifts from its own declared plan.
        planned_unknown = sorted(
            {v.get("id") for v in plan.validators} - self._declared_ids
        )
        if planned_unknown:
            raise UnknownValidatorError(
                f"engine {adapter.engine_id!r} planned validator(s) {planned_unknown}, "
                f"which are not part of bundle {self.bundle.get('id')!r} "
                f"(declares: {sorted(self._declared_ids)})"
            )

        started_at = datetime.now(timezone.utc)
        raw = adapter.run(plan)
        # What the tool REPORTED it was, not what the adapter was pinned
        # against. `PromptfooAdapter` goes to trouble to return None from
        # `promptfoo --version` rather than assume the pinned constant, and
        # reading `adapter.engine_version` here threw that away — a host
        # running a different release, or none at all, would sign a manifest
        # asserting the pinned version, which is a fabricated fact in a
        # signed record.
        engine_version = raw.engine_version or adapter.engine_version
        normalized = adapter.normalize(raw)
        artifact = adapter.raw_artifact(raw)
        completed_at = datetime.now(timezone.utc)

        unknown = sorted({r.validator_id for r in normalized} - self._declared_ids)
        if unknown:
            raise UnknownValidatorError(
                f"engine {adapter.engine_id!r} normalized results for validator(s) {unknown}, "
                f"which are not part of bundle {self.bundle.get('id')!r} "
                f"(declares: {sorted(self._declared_ids)})"
            )

        # Evidence the adapter derived from `raw`. Optional: a third-party
        # adapter implementing only the four abstract methods still works,
        # it just backs its results with the raw artifact alone.
        evidence: dict[str, dict] = {}
        envelopes = getattr(adapter, "evidence_envelopes", None)
        if callable(envelopes):
            evidence.update(envelopes(raw))

        artifact_envelope = build_evidence_envelope(
            context.execution_id,
            f"engine:{adapter.engine_id}",
            raw_engine_artifact_item(
                artifact,
                engine_id=adapter.engine_id,
                engine_version=engine_version,
                plan_hash=plan.plan_hash(),
            ),
        )
        artifact_evidence_id = evidence_content_id(artifact_envelope)
        evidence[artifact_evidence_id] = artifact_envelope

        results: list[ValidatorResult] = []
        failure_scenarios: list[dict] = []
        for index, item in enumerate(normalized):
            results.append(ValidatorResult(
                validator_id=item.validator_id,
                version=item.version,
                status=item.status,
                score=item.score,
                severity=item.severity,
                observations=list(item.observations),
                evidence_ids=list(item.evidence_ids),
            ))
            if item.failure_scenario:
                failure_scenarios.append({"result_index": index, **item.failure_scenario})

        manifest = build_manifest(
            runner_version=self.runner_version,
            protocol_id=self.bundle["id"],
            protocol_version=self.bundle["version"],
            bundle_hash=self.bundle["bundle_hash"],
            validators=plan.validators or [
                {"id": r.validator_id, "version": r.version} for r in results
            ],
            system_version_ref=context.agent_version_ref,
            configuration_snapshot=context.configuration,
            input_datasets=context.inputs,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
            trace_provenance=_trace_provenance(context),
            engine={
                "id": adapter.engine_id,
                "version": engine_version,
                "plan_hash": plan.plan_hash(),
                "raw_artifact_sha256": sha256_hex(artifact),
                "raw_artifact_evidence_id": artifact_evidence_id,
            },
            capabilities=_capabilities(),
        )
        return results, manifest, evidence, failure_scenarios


def _capabilities() -> dict:
    """`capabilities.report()`, or a disclosed failure instead of one.

    This runs at manifest-build time, AFTER the campaign has executed. An
    exception from a probe — `find_spec` on a broken install, an unreadable
    PATH entry, the validator registry import — would destroy a completed
    run over a descriptive field nothing depends on. The failure is recorded
    where a reader can see it rather than swallowed to `None`, which would
    read as a runner too old to report at all.
    """
    try:
        return capability_report()
    except Exception as exc:  # noqa: BLE001 - a probe must never fail a run
        return {"error": f"{type(exc).__name__}: {exc}", "engines": {}, "validators": [], "extras": {}}


def _trace_provenance(context: ValidatorContext) -> dict | None:
    """`{"source": ..., "event_count": ...}` when `context.inputs["trace"]`
    is a trace dict carrying `otel_ingest.py`'s `"source"` marker, else
    `None`. The engine deliberately doesn't import `otel_ingest` to check
    this — that would make the OTLP converter a hard dependency of every
    runner invocation, including the many validators that will never touch
    OTel. Duck-typed instead: any trace-shaped dict with a `source` key is
    disclosed the same way, whatever produced it."""
    trace = context.inputs.get("trace")
    if not isinstance(trace, dict) or "source" not in trace:
        return None
    events = trace.get("events") or []
    provenance: dict = {
        "source": trace["source"],
        "event_count": len(events),
    }
    provenance.update(_timing_signals(events))
    return provenance


def _parse_event_time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _timing_signals(events: list) -> dict:
    """Completeness signals read off event timestamps, never off content:
    `first_event_at`, `last_event_at` and `max_gap_seconds` (the longest
    silence between two consecutive events, in time order). A key is left out
    when no event carries a readable timestamp, so a manifest from a trace
    without times reads as one from an older runner rather than as a claim of
    zero silence. Only dates and numbers leave the host: no tool name, no
    payload, no id."""
    times = sorted(
        t for t in (_parse_event_time(e.get("timestamp")) for e in events if isinstance(e, dict))
        if t is not None
    )
    if not times:
        return {}
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    return {
        "first_event_at": times[0].isoformat(),
        "last_event_at": times[-1].isoformat(),
        "max_gap_seconds": round(max(gaps), 3) if gaps else 0.0,
    }
