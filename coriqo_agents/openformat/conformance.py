"""The conformance suite a validator must pass before anyone should
run it, whether Coriqo wrote it or not.

Six checks. They are not a security boundary and this module does not
pretend to be one: a validator is arbitrary Python running inside the
customer's own environment, and nothing here stops it doing whatever that
code wants. What the suite establishes is narrower and still worth having —
that a validator's *verdict* is a property of the trace and not of the run:

  IDENTITY     validator_id is present and in the id shape the wire format uses
  VERSION      version is a semantic version, so results can be compared
  SHAPE        evaluate() returns a ValidatorResult that serializes to JSON
  STATUS       the status is one of the six the result contract defines
  DETERMINISM  the same trace evaluated twice gives byte-identical results
  PURITY       evaluating a trace does not modify the trace

DETERMINISM is the one the benchmark depends on: a published score is only
reproducible by a stranger if the same trace produces the same verdict, and
that is not something a docstring saying "deterministic" establishes — it is
what this check runs twice to find out.

A report is passing only when EVERY check passed. `passed` enumerates the
good status and requires all of them to be it, so a check that ends up in
some third state later (skipped, errored) makes a report fail rather than
disappearing from the count.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..runner.contract import Status, Validator, ValidatorContext

#: The six check ids, in the order `check_conformance` runs them.
CHECK_IDS: tuple[str, ...] = ("IDENTITY", "VERSION", "SHAPE", "STATUS", "DETERMINISM", "PURITY")

CHECK_PASS = "PASS"
CHECK_FAIL = "FAIL"

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass
class CheckOutcome:
    check_id: str
    status: str
    detail: str = ""


@dataclass
class ConformanceReport:
    validator_id: str
    checks: list[CheckOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # Every check must be PASS, and there must be one outcome per known
        # check id — a suite that silently ran four of six checks would
        # otherwise report a passing validator on the strength of the four.
        if {c.check_id for c in self.checks} != set(CHECK_IDS):
            return False
        return all(c.status == CHECK_PASS for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "passed": self.passed,
            "checks": [
                {"check_id": c.check_id, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr)


def check_conformance(
    validator_cls: type[Validator],
    *,
    trace: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> ConformanceReport:
    """Run the six checks against `validator_cls` using `trace` as the
    single sample input. Never raises for a validator that misbehaves: a
    validator that throws is a FAIL with the exception in the detail, which
    is the report a third-party author needs, not a traceback out of the
    conformance runner."""
    # Same crash-vs-report contract `_run` applies to evaluate(), applied to
    # the constructor: a third-party validator whose __init__ raises is a
    # failing validator, and the author needs the six-check report with the
    # exception in the detail — not a traceback out of the conformance
    # runner, which tells them nothing about which check they failed.
    try:
        instance = validator_cls(bundle or {})
    except Exception as exc:  # noqa: BLE001 - reported, never propagated
        detail = f"__init__ raised {type(exc).__name__}: {exc}"
        report = ConformanceReport(validator_id="")
        # Every check, not just IDENTITY: none of the six can be established
        # about a validator that cannot be constructed, and `passed` requires
        # one outcome per known check id.
        report.checks = [CheckOutcome(check_id, CHECK_FAIL, detail) for check_id in CHECK_IDS]
        return report

    report = ConformanceReport(validator_id=getattr(instance, "validator_id", "") or "")
    checks: list[CheckOutcome] = []

    vid = getattr(instance, "validator_id", "")
    if isinstance(vid, str) and _ID_PATTERN.match(vid):
        checks.append(CheckOutcome("IDENTITY", CHECK_PASS))
    else:
        checks.append(CheckOutcome(
            "IDENTITY", CHECK_FAIL,
            f"validator_id {vid!r} must match {_ID_PATTERN.pattern}",
        ))

    version = getattr(instance, "version", "")
    if isinstance(version, str) and _SEMVER_PATTERN.match(version):
        checks.append(CheckOutcome("VERSION", CHECK_PASS))
    else:
        checks.append(CheckOutcome(
            "VERSION", CHECK_FAIL, f"version {version!r} must be MAJOR.MINOR.PATCH",
        ))

    def _run(input_trace: dict[str, Any]) -> tuple[Any, str | None]:
        context = ValidatorContext(
            execution_id="conformance",
            agent_model_id=input_trace.get("agent_id", ""),
            agent_version_ref=input_trace.get("agent_version"),
            environment="CONFORMANCE",
            inputs={"trace": input_trace},
        )
        # ONE instance per run: `execute` and `evaluate` are two halves of
        # the same call and a validator is entitled to carry state between
        # them. Two instances would hand `evaluate` an observation produced
        # by a different object, which is not how the engine runs it — and
        # the difference would show up as a spurious failure in somebody
        # else's validator, in our suite, with no way for them to see why.
        # A FRESH instance per run, though, is the point: reusing one across
        # the two determinism runs would hide exactly the per-instance state
        # that check exists to find.
        validator = validator_cls(bundle or {})
        try:
            return validator.evaluate(validator.execute(context)), None
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            return None, f"{type(exc).__name__}: {exc}"

    # PURITY needs a trace the validator has not seen; DETERMINISM needs two
    # runs over equal inputs. Both take their own deep copy so neither can
    # be the thing that contaminates the other.
    first_input = copy.deepcopy(trace)
    before = _canonical(first_input)
    first, first_error = _run(first_input)
    after = _canonical(first_input)

    if first_error is not None:
        detail = f"evaluate() raised {first_error}"
        checks.append(CheckOutcome("SHAPE", CHECK_FAIL, detail))
        checks.append(CheckOutcome("STATUS", CHECK_FAIL, detail))
        checks.append(CheckOutcome("DETERMINISM", CHECK_FAIL, detail))
        checks.append(CheckOutcome(
            "PURITY", CHECK_PASS if before == after else CHECK_FAIL,
            "" if before == after else "the trace passed in was modified",
        ))
        report.checks = checks
        return report

    to_dict = getattr(first, "to_dict", None)
    payload: Any = None
    if to_dict is None:
        checks.append(CheckOutcome("SHAPE", CHECK_FAIL, "evaluate() did not return a ValidatorResult"))
    else:
        # `to_dict` is the validator author's code too, and so is whatever it
        # returns — a to_dict() that raises, or one returning something the
        # `in` test cannot be applied to, is a SHAPE failure to report, not a
        # traceback out of a runner that promises never to raise.
        missing: list[str] = []
        serializable = False
        try:
            payload = to_dict()
            missing = [
                key for key in ("validator_id", "version", "status", "score", "severity",
                                "observations", "evidence_ids")
                if key not in payload
            ]
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            payload = None
            missing = [f"to_dict() raised {type(exc).__name__}: {exc}"]
        else:
            try:
                json.dumps(payload)
                serializable = True
            except (TypeError, ValueError) as exc:
                missing.append(f"not JSON-serializable: {exc}")
        if not missing and serializable:
            checks.append(CheckOutcome("SHAPE", CHECK_PASS))
        else:
            checks.append(CheckOutcome("SHAPE", CHECK_FAIL, f"result is missing required fields: {missing}"))

    observed_status = payload.get("status") if isinstance(payload, dict) else None
    if observed_status in Status.ALL:
        checks.append(CheckOutcome("STATUS", CHECK_PASS))
    else:
        checks.append(CheckOutcome(
            "STATUS", CHECK_FAIL,
            f"status {observed_status!r} is not one of {list(Status.ALL)}",
        ))

    second_input = copy.deepcopy(trace)
    second, second_error = _run(second_input)
    if second_error is not None:
        checks.append(CheckOutcome(
            "DETERMINISM", CHECK_FAIL, f"the second run raised where the first did not: {second_error}",
        ))
    else:
        # Serializing the two results runs the validator author's to_dict()
        # (and, via `default=repr`, its __repr__) once more. Either can
        # raise, and this runner reports rather than propagates.
        try:
            first_canon = _canonical(first.to_dict() if hasattr(first, "to_dict") else first)
            second_canon = _canonical(second.to_dict() if hasattr(second, "to_dict") else second)
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            checks.append(CheckOutcome(
                "DETERMINISM", CHECK_FAIL,
                f"the two results could not be compared: {type(exc).__name__}: {exc}",
            ))
        else:
            if first_canon == second_canon:
                checks.append(CheckOutcome("DETERMINISM", CHECK_PASS))
            else:
                checks.append(CheckOutcome(
                    "DETERMINISM", CHECK_FAIL,
                    "two evaluations of the same trace produced different results:\n"
                    f"  first:  {first_canon}\n  second: {second_canon}",
                ))

    if before == after:
        checks.append(CheckOutcome("PURITY", CHECK_PASS))
    else:
        checks.append(CheckOutcome(
            "PURITY", CHECK_FAIL,
            "the trace handed to the validator was modified by evaluating it — a validator "
            "must not write into its own input, or the next validator in the run reads a "
            "trace the agent never produced",
        ))

    report.checks = checks
    return report
