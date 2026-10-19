"""The Inspect (UK AISI) engine adapter.

Fourth adapter, and shaped differently from every one before it: Inspect is
a Python evaluation library the runner drives *in-process* (like PyRIT), but
this adapter is not itself a verdict source. Its job is to turn an Inspect
eval log into Coriqo trace events and hand them to the SAME native validator
CLASSES the runner already ships (`..validators.VALIDATOR_REGISTRY`), rather
than re-implementing any matcher of its own. Where PyRIT's `normalize`
imports a validator's RULE TABLE and re-applies it to a transcript, this
adapter imports and RUNS the validator itself against a trace it built — the
spec's own framing: "the adapter here is a trace source more than a verdict
source".

Pinned against **inspect-ai 0.3.263** (PyPI, UKGovernmentBEIS/inspect_ai,
2026-09-04). Confirmed from that release's own source/docs:
`inspect_ai.eval(tasks, model=..., model_base_url=..., model_args=...,
log_format=...)` returning `list[EvalLog]`; `EvalLog.status` (`"success"` /
`"error"` / `"cancelled"` / `"started"`) and `.error`; `EvalSample.messages`
and `.events`; `ToolEvent` (`function`, `arguments`, `result`, `error`,
`timestamp`) and `ModelEvent` (`input`, `output`, `error`, `timestamp`); the
`openai-api/<provider>/<model>` convention for pointing Inspect at an
arbitrary REST/OpenAI-compatible endpoint via `model_base_url` and a
provider-scoped API key. What is NOT confirmed and is therefore handled
defensively rather than assumed: the exact discriminator key Inspect's
pydantic model dump uses to tell `ToolEvent`/`ModelEvent`/other event kinds
apart in a plain dict — `_events_from_sample` duck-types on field presence
(`function`+`arguments` for a tool event, `input`+`output`+`model` for a
model event) instead, so a wrong guess at that one field name degrades to
"event not recognised", not a KeyError.

**Scope note, stated rather than implied:** this adapter does NOT author
Inspect `Task`/`Solver`/`Scorer` code — the solver-authoring surface
(`basic_agent`, `use_tools`, sandboxed tool execution) was outside what
could be verified against the current release in the time available, and
master-plan rule 3 (read the actual API, or skip) applies to code this
adapter would need to get right on the first try just as much as it applies
to a probe table. Coding a solver from memory would be exactly the "wrapper
that can't work" the acceptance rule warns against. Instead, the CUSTOMER's
own already-authored Inspect tasks — dotted import paths named on the
campaign target (`target["inspect_tasks"]`) — are what this adapter runs; it
contributes the eval-log-to-trace conversion and the re-use of Coriqo's own
validators, not the task content. This also means "sandboxed runs
without E2B" is only partly delivered: Inspect's own `sandbox=` mechanism is
available to whatever task the customer points at, but this adapter neither
configures nor verifies it — that is the task author's responsibility, and a
task using it is unaffected by anything below.

Three rules from the PyRIT adapter, applied here:

1. **Inspect's own task scorer is not Coriqo's verdict.** A scored Inspect
   task's `EvalLog.results` is kept as an advisory observation, never as a
   status — every validator's own PASS/FAIL/... comes from running that
   validator's real `execute`/`evaluate` against the trace this adapter
   built, the same code path a native run uses.
2. **A missing dependency, a missing task list, or a failed eval log is a
   disclosed ERROR per intended validator, never NOT_APPLICABLE and never a
   crash.** `_load_inspect()` confines the import; `run()` also deletes the
   previous run's event log FIRST, so a run that never started cannot
   normalize a stale file into fresh verdicts.
3. **inspect-ai is its own optional extra**, never in the base install or
   `runner` — `pip install coriqo-agents[inspect]`.

Pure stdlib at import time. `inspect_ai` is imported only inside `run()`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contract import Status
from .contract import EngineAdapter, NormalizedResult, RawRun, declared_version as _declared_version
from .manual import declared_validators
from .plan import EnginePlan

ENGINE_ID = "inspect"

#: The inspect-ai release this adapter's imports and eval-log shape were
#: read off. Recorded separately from what the host reports at run time.
PINNED_INSPECT_VERSION = "0.3.263"

#: Every trace event this run produced, one JSON object per line — the raw
#: artifact, hashed and uploaded as evidence, and what a reviewer re-reads to
#: check the validators' verdicts against what Inspect actually logged.
EVENTS_FILENAME = "inspect_events.jsonl"

_DEFAULT_TIMEOUT_SECONDS = 3600


class InspectUnavailableError(RuntimeError):
    """inspect-ai could not be imported on this host.

    Raised only inside `_load_inspect`, caught in `run()`, and turned into a
    disclosed ERROR result. It never reaches a caller.
    """


def _load_inspect() -> dict[str, Any]:
    """Import inspect-ai, or say precisely why it could not be imported.

    `Exception`, not just `ImportError` — same reasoning `pyrit._load_pyrit`
    documents: a half-installed optional dependency can raise something else
    entirely at import time, and none of it may escape as a raw traceback.
    """
    try:
        from inspect_ai import eval as inspect_eval  # noqa: PLC0415
        import inspect_ai  # noqa: PLC0415
    except ImportError as exc:
        raise InspectUnavailableError(
            "inspect-ai is not installed on this host — it is an optional extra "
            "(pip install 'coriqo-agents[inspect]'), never part of the base runner install. "
            f"Import failed with: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - see docstring
        raise InspectUnavailableError(
            f"inspect-ai is installed but could not be imported: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "version": getattr(inspect_ai, "__version__", None),
        "eval": inspect_eval,
    }


class InspectAdapter(EngineAdapter):
    """Runs the customer's own Inspect tasks against a REST target and
    converts the resulting eval log into a trace Coriqo's native validators
    can judge.

    `target["inspect_tasks"]` is a non-empty list of dotted import paths to
    `@task`-decorated functions ALREADY INSTALLED in the customer's
    environment (their own eval suite, or a published one such as
    `inspect_evals`) — this adapter runs them, it does not author them. An
    empty or missing list is disclosed at `prepare` time via the plan and at
    `run` time via one ERROR result per intended validator, the same
    "nothing to normalize" shape a missing binary produces in the other
    adapters.

    `target["model"]` names the model identifier Inspect should record for
    the run (cosmetic — it is not one of Coriqo's connector types); the
    actual traffic goes to `target["endpoint_ref"]` via Inspect's
    `openai-api/<provider>/<model>` compatible-provider convention.
    `target["auth_env"]` is, as with every other adapter, the NAME of an
    environment variable the customer set — never a secret. It is read once,
    at run time, only because Inspect's provider convention ties the API key
    to a FIXED env var name derived from the provider slug rather than
    accepting an arbitrary one; the value is passed straight to Inspect's
    `model_args` and is never written to the plan, the plan hash, or the
    manifest — only `auth_env`'s NAME is, exactly like every other target
    field.
    """

    engine_id = ENGINE_ID
    _PROVIDER = "coriqo"

    def __init__(
        self,
        *,
        execution_id: str,
        target: dict[str, Any],
        scope: dict[str, Any] | None = None,
        workdir: str | Path,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        env: dict[str, str] | None = None,
    ) -> None:
        self.engine_version = PINNED_INSPECT_VERSION
        super().__init__()
        self.execution_id = execution_id
        self.target = target or {}
        self.scope = scope or {}
        self.workdir = Path(workdir)
        self.timeout_seconds = timeout_seconds
        self._env = env
        # Default so a test can call `normalize` directly against a
        # hand-built `RawRun`, the same shortcut `test_pyrit_adapter.py`'s
        # `normalizer()` helper takes, without going through `prepare` first.
        self._bundle: dict[str, Any] = {}

    # -- prepare ---------------------------------------------------------
    def prepare(self, campaign_bundle: dict[str, Any]) -> EnginePlan:
        # Kept for `normalize`, which constructs the SAME validator classes
        # `resolve_validators` would, and those take the compiled bundle at
        # construction so a validator with declarative test material (the
        # generated attack-scenario corpus) can read it — see
        # `contract.Validator.__init__`. `EngineAdapter.normalize(raw)` has
        # no bundle parameter of its own (the adapter contract fixes that signature), so an
        # adapter that needs the bundle after `prepare` has to keep it,
        # exactly as `PromptfooAdapter`/`PyritAdapter` keep `self.target`.
        self._bundle = campaign_bundle or {}
        validators = declared_validators(campaign_bundle)
        out_of_scope = [str(x) for x in (self.scope.get("out_of_scope") or [])]
        tasks = [str(t) for t in (self.target.get("inspect_tasks") or []) if str(t).strip()]

        self.workdir.mkdir(parents=True, exist_ok=True)
        return EnginePlan(
            engine_id=self.engine_id,
            validators=validators,
            config={
                "tasks": tasks,
                "model": str(self.target.get("model") or "target"),
                "endpoint_ref": str(self.target.get("endpoint_ref") or ""),
            },
            workdir=str(self.workdir),
            out_of_scope=out_of_scope,
        )

    # -- run -------------------------------------------------------------
    def run(self, plan: EnginePlan) -> RawRun:
        """Run the configured Inspect tasks and convert their eval logs to
        one JSON-per-line trace-event file — the raw artifact.

        Every failure mode is a disclosed ERROR on `invocation`, never a
        raised exception: a missing dependency, an empty task list, and an
        Inspect run that raised or reported `status != "success"` all leave
        `invocation["error"]` set and `output` as `None`, which `normalize`
        turns into one ERROR result per intended validator — never
        NOT_APPLICABLE, per the lesson `pyrit.py`'s round-1 review paid for.
        """
        started = datetime.now(timezone.utc)
        events_path = self.workdir / EVENTS_FILENAME
        events_path.unlink(missing_ok=True)

        tasks: list[str] = list(plan.config.get("tasks") or [])
        invocation: dict[str, Any] = {
            "plan_hash": plan.plan_hash(),
            "intended_validators": [v["id"] for v in plan.validators],
            "validator_versions": {v["id"]: v["version"] for v in plan.validators},
            "tasks": tasks,
            "inspect_version_pinned": PINNED_INSPECT_VERSION,
            "inspect_version_reported": None,
            "exit_code": 0,
            "error": None,
            "log_statuses": [],
            "task_results_advisory": [],
        }

        if not tasks:
            invocation["exit_code"] = 2
            invocation["error"] = (
                "no inspect_tasks configured on the campaign target — this adapter runs the "
                "customer's own already-authored Inspect tasks, named by dotted import path, "
                "and none were named"
            )
            return RawRun(
                engine_id=self.engine_id, engine_version=self.engine_version,
                output=None, artifact=b"", invocation=invocation,
                started_at=started.isoformat(), completed_at=datetime.now(timezone.utc).isoformat(),
            )

        try:
            loaded = _load_inspect()
        except InspectUnavailableError as exc:
            invocation["exit_code"] = 127
            invocation["error"] = str(exc)
            return RawRun(
                engine_id=self.engine_id, engine_version=self.engine_version,
                output=None, artifact=b"", invocation=invocation,
                started_at=started.isoformat(), completed_at=datetime.now(timezone.utc).isoformat(),
            )

        invocation["inspect_version_reported"] = loaded["version"]
        model_name = str(plan.config.get("model") or "target")
        model = f"openai-api/{self._PROVIDER}/{model_name}"
        model_base_url = str(plan.config.get("endpoint_ref") or "")
        model_args: dict[str, Any] = {}
        auth_env = self.target.get("auth_env")
        env = self._env if self._env is not None else os.environ
        if isinstance(auth_env, str) and auth_env.strip() and env.get(auth_env.strip()):
            # See the class docstring: Inspect's provider convention ties the
            # key to a fixed slug, not an arbitrary env var name, so this is
            # the one adapter that has to read the VALUE rather than only
            # forward the NAME. It goes straight to Inspect's own call and is
            # never written to `invocation`, the plan, or the manifest.
            model_args["api_key"] = env[auth_env.strip()]

        try:
            logs = loaded["eval"](
                tasks=tasks,
                model=model,
                model_base_url=model_base_url or None,
                model_args=model_args,
                log_dir=str(self.workdir / "inspect_logs"),
                log_format="json",
                fail_on_error=False,
            )
        except Exception as exc:  # noqa: BLE001 - an Inspect crash is a disclosed ERROR, not a
            # traceback in a customer's terminal, same discipline as PromptfooAdapter/PyritAdapter.
            invocation["exit_code"] = 1
            invocation["error"] = f"inspect_ai.eval raised {type(exc).__name__}: {exc}"
            return RawRun(
                engine_id=self.engine_id, engine_version=loaded["version"] or self.engine_version,
                output=None, artifact=b"", invocation=invocation,
                started_at=started.isoformat(), completed_at=datetime.now(timezone.utc).isoformat(),
            )

        events: list[dict[str, Any]] = []
        for log in logs or []:
            status = getattr(log, "status", None)
            invocation["log_statuses"].append(status)
            results = getattr(log, "results", None)
            if results is not None:
                invocation["task_results_advisory"].append(_advisory_results(log))
            if status != "success":
                error = getattr(log, "error", None)
                invocation["error"] = (
                    (invocation["error"] + "; " if invocation["error"] else "")
                    + f"Inspect task log status={status!r}: {error}"
                )
                continue
            log_events, sample_errors = _events_from_log(log)
            events.extend(log_events)
            for sample_id, sample_error in sample_errors:
                # A log-level status of "success" says nothing about whether
                # EVERY sample in it completed — `fail_on_error=False` keeps
                # the log "successful" while one sample's tool call/sandbox
                # crashes and its `events` come back empty. That sample's
                # trace is missing, not clean, and folding it into the same
                # `invocation["error"]` field `normalize` already treats as
                # fatal (below) is what stops the missing evidence from
                # reading as an absence of findings.
                invocation["error"] = (
                    (invocation["error"] + "; " if invocation["error"] else "")
                    + f"Inspect sample {sample_id!r} did not complete: {sample_error}"
                )

        if invocation["error"] and not events:
            invocation["exit_code"] = 1

        artifact = "".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in events).encode("utf-8")
        events_path.write_bytes(artifact)

        return RawRun(
            engine_id=self.engine_id,
            engine_version=loaded["version"] or self.engine_version,
            output={"events": events} if events or not invocation["error"] else None,
            artifact=artifact,
            invocation=invocation,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    # -- normalize -------------------------------------------------------
    def normalize(self, raw: RawRun) -> list[NormalizedResult]:
        """Build one trace from every event the run produced, then run each
        intended validator's REAL implementation against it — imported from
        `..validators.VALIDATOR_REGISTRY`, never reimplemented. No model, and
        Inspect's own task scorer never decides a status; its result is carried as `task_results_advisory` on every result
        instead.
        """
        from ..validators import VALIDATOR_REGISTRY
        from ..contract import ValidatorContext

        intended = list(raw.invocation.get("intended_validators") or [])
        advisory = raw.invocation.get("task_results_advisory") or []
        advisory_note = (
            f"engine score (advisory), not used for this status: {json.dumps(advisory, sort_keys=True, default=str)}"
            if advisory else None
        )

        # ANY recorded failure — a total failure (`raw.output is None`), one
        # task log that errored while another produced events, or one
        # SAMPLE within an otherwise-"success" log that never completed
        # (`_events_from_log`'s `sample_errors`, folded into this same
        # field by `run()`) — makes the trace's completeness unknown for
        # this run. Mandated-review finding: this adapter runs the REAL
        # validators against a trace it built, which means the trace's
        # completeness is load-bearing for verdict correctness in a way it
        # was not for the other adapters' re-derivation approach. A
        # validator judging an incomplete trace can only underreport, never
        # overreport, what happened — so every intended validator is ERROR
        # here, not just the ones whose window happened to be missing.
        # "We could not establish what happened" must never collapse into
        # "nothing happened" (PASS) or "there was nothing to check"
        # (NOT_APPLICABLE) — both would report an attack resisted or
        # untested when it may never have been delivered at all.
        if raw.output is None or raw.invocation.get("error"):
            reason = raw.invocation.get("error") or "Inspect produced no readable eval log"
            return [
                NormalizedResult(
                    validator_id=vid, version=_declared_version(raw, vid),
                    status=Status.ERROR, observations=[reason], engine=self.engine_id,
                )
                for vid in intended
            ]

        trace = {"source": "inspect", "events": raw.output.get("events") or []}
        results: list[NormalizedResult] = []
        for validator_id in intended:
            cls = VALIDATOR_REGISTRY.get(validator_id)
            if cls is None:
                # A validator id this SDK has no implementation for at all —
                # distinct from "Inspect produced no evidence for it", which
                # the validator's own NOT_APPLICABLE branch already covers.
                results.append(NormalizedResult(
                    validator_id=validator_id, version=_declared_version(raw, validator_id),
                    status=Status.ERROR, engine=self.engine_id,
                    observations=[f"no registered implementation for validator {validator_id!r}"],
                ))
                continue
            validator = cls(self._bundle)
            try:
                context = ValidatorContext(
                    execution_id=self.execution_id, agent_model_id="", agent_version_ref=None,
                    environment=None, inputs={"trace": trace},
                )
                observation = validator.execute(context)
                result = validator.evaluate(observation)
            except Exception as exc:  # noqa: BLE001 - a crashing validator is a disclosed ERROR,
                # matching RunnerEngine.run's own guard for native validators.
                results.append(NormalizedResult(
                    validator_id=validator_id, version=cls.version, status=Status.ERROR,
                    engine=self.engine_id,
                    observations=[f"validator raised {type(exc).__name__}: {exc}"],
                ))
                continue
            observations = list(result.observations)
            if advisory_note:
                observations.append(advisory_note)
            scenario = None
            try:
                scenario = validator.failure_scenario(result)
            except Exception:  # noqa: BLE001 - the result stands even if the scenario builder fails.
                pass
            results.append(NormalizedResult(
                validator_id=result.validator_id, version=result.version, status=result.status,
                score=result.score, severity=result.severity, observations=observations,
                engine=self.engine_id, failure_scenario=scenario,
            ))
        return results

    # -- raw_artifact ----------------------------------------------------
    def raw_artifact(self, raw: RawRun) -> bytes:
        return raw.artifact


# --------------------------------------------------------------------------
# Reading an Inspect EvalLog. Duck-typed rather than keyed to one exact
# discriminator field name — see the module docstring's "not confirmed" note.
# --------------------------------------------------------------------------

def _advisory_results(log: Any) -> dict[str, Any]:
    results = getattr(log, "results", None)
    try:
        return results.model_dump() if hasattr(results, "model_dump") else dict(results or {})
    except Exception:  # noqa: BLE001 - advisory only; never block a real result on this
        return {"unreadable": True}


def _events_from_log(log: Any) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]]]:
    """`(events, sample_errors)` for one eval log.

    A log-level `status == "success"` does not mean every SAMPLE in it
    completed: `EvalSample.error` is set independently when one sample's
    tool call, sandbox, or model turn crashed, and `EvalSample.events` for
    that sample comes back empty or partial either way. Reading only
    `sample.events` (as an earlier version of this function did) let a
    crashed sample's absence read as "the agent had nothing to hide" rather
    than "this attempt was never delivered" — reviewer-confirmed finding:
    a validator run against a trace silently missing that sample returns
    PASS or NOT_APPLICABLE, which is the worst way an assurance product can
    be wrong. `sample_errors` is returned separately so the caller can make
    that failure fatal to the whole run's verdicts, not just a footnote.
    """
    events: list[dict[str, Any]] = []
    sample_errors: list[tuple[Any, Any]] = []
    samples = getattr(log, "samples", None) or []
    for sample in samples:
        sample_id = getattr(sample, "id", None)
        sample_error = getattr(sample, "error", None)
        if sample_error:
            sample_errors.append((sample_id, sample_error))
        for index, raw_event in enumerate(getattr(sample, "events", None) or []):
            events.extend(_trace_events_for(raw_event, sample_id, fallback_index=index))
    return events, sample_errors


def _as_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # noqa: BLE001 - fall through to "not recognised"
            return None
    return None


def _trace_events_for(raw_event: Any, sample_id: Any, fallback_index: int = 0) -> list[dict[str, Any]]:
    """One Inspect event -> zero or more Coriqo trace events.

    `fallback_index` should be the raw event's position within its
    sample's event list (see `_events_from_log`) — it is only used when
    the event itself carries no `id`, and is what keeps two id-less events
    in the same sample from colliding on the same `event_id`.

    A `ToolEvent`-shaped object (has `function` and `arguments`) becomes a
    TOOL_CALL, plus a TOOL_RESULT when it carries a `result`. A
    `ModelEvent`-shaped object (has `input`, `output`, `model`) becomes a
    FINAL_RESPONSE carrying the model's output text — the only event kind
    the agent-behavioral validators can read without a live tool-call
    channel. Anything else is skipped, not guessed at: an unrecognised event
    shape must not silently vanish INTO a wrong bucket, only out of the
    trace entirely, where `contract.py`'s own NOT_APPLICABLE branches
    already handle "nothing to check.\""""
    data = _as_dict(raw_event)
    if data is None:
        return []
    timestamp = data.get("timestamp")
    out: list[dict[str, Any]] = []

    if "function" in data and "arguments" in data:
        # `fallback_index` — the raw event's position within its sample's
        # event list, passed in by `_events_from_log` — not `len(out)`.
        # `out` starts empty on every call to this function, so `len(out)`
        # BEFORE any append is always 0: every id-less ToolEvent in a
        # sample got the SAME fallback id (`<sample_id>:0:tool_call`), and
        # anything keying or deduplicating by `event_id` downstream
        # (governance chain storage, manifest evidence linking, a
        # validator indexing events) collapsed a sample's second and later
        # tool calls into its first — evidence of additional attack
        # attempts silently vanishing from the signed record. A per-sample
        # positional index is unique across the whole sample regardless of
        # how many trace events any one raw event expands into, without
        # needing any module-level counter state.
        event_id = f"{sample_id}:{data.get('id') if data.get('id') is not None else fallback_index}:tool_call"
        out.append({
            "event_id": event_id,
            "event_type": "TOOL_CALL",
            "tool": data.get("function"),
            "action": json.dumps(data.get("arguments"), sort_keys=True, default=str),
            "payload": data.get("arguments"),
            "timestamp": timestamp,
        })
        if data.get("result") is not None:
            out.append({
                "event_id": f"{event_id}:tool_result",
                "event_type": "TOOL_RESULT",
                "tool": data.get("function"),
                "payload": data.get("result"),
                "timestamp": timestamp,
            })
        if data.get("error"):
            out.append({
                "event_id": f"{event_id}:tool_error",
                "event_type": "EXTERNAL_SIDE_EFFECT",
                "tool": data.get("function"),
                "payload": data.get("error"),
                "timestamp": timestamp,
            })
        return out

    if "input" in data and "output" in data and "model" in data:
        output = data.get("output")
        text = _output_text(output)
        if text:
            out.append({
                "event_id": f"{sample_id}:model:{timestamp}",
                "event_type": "FINAL_RESPONSE",
                "tool": None,
                "payload": text,
                "timestamp": timestamp,
            })
        return out

    return []


def _output_text(output: Any) -> str:
    data = _as_dict(output)
    if data is None:
        return str(output) if isinstance(output, str) else ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = _as_dict(choices[0]) or {}
        content = (message.get("message") or {}).get("content") if isinstance(message.get("message"), dict) else message.get("content")
        if isinstance(content, str):
            return content
    completion = data.get("completion")
    return completion if isinstance(completion, str) else ""
