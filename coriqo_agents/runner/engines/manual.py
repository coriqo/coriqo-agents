"""The `manual` engine — a human did the testing and wrote down what they
found.

This adapter exists so the CLI, the manifest and the ingest path are the
same three steps whether a tool ran or a person did. It is not a way to make
hand-written findings look attested: the server records `engine='manual'`
against the execution and every result, and Coriqo's API treats
that engine id as unattested, so a report can
say which results nothing executed. The adapter does not soften that and
must not: a manual result's value is that it is labelled.

Division of labour between `run` and `normalize` is deliberate and differs
from the work-package sketch, which said `run` is a no-op and `normalize`
reads the file. Reading a file is a side effect whose result can change
between two calls, and `normalize` is required to be deterministic for a
given `RawRun` (the engine adapter contract). So `run` reads the
bytes and `normalize` parses what it was handed — same work, in the method
that is allowed to touch the world.

Pure stdlib.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contract import Severity, Status
from .contract import EngineAdapter, NormalizedResult, RawRun
from .plan import EnginePlan

#: Filename the adapter reads out of its workdir unless told otherwise.
DEFAULT_RESULTS_FILENAME = "manual_results.json"

ENGINE_ID = "manual"


class ManualAdapter(EngineAdapter):
    """Reads a human-written results file and normalizes it.

    `workdir` is where `manual_results.json` lives. `adapter_version` is this
    adapter's own version, since there is no external tool to interrogate for
    one — a hand-run test has no binary and pretending otherwise would put a
    fabricated version number in a signed manifest.
    """

    engine_id = ENGINE_ID
    engine_version = "1.0.0"

    def __init__(self, *, workdir: str | Path, results_filename: str = DEFAULT_RESULTS_FILENAME) -> None:
        super().__init__()
        self.workdir = Path(workdir)
        self.results_filename = results_filename

    # -- prepare ---------------------------------------------------------
    def prepare(self, campaign_bundle: dict[str, Any]) -> EnginePlan:
        """The plan is the list of validators the bundle declares. There is
        no config to write: the "engine" is a person."""
        return EnginePlan(
            engine_id=self.engine_id,
            validators=declared_validators(campaign_bundle),
            config={"results_file": self.results_filename},
            workdir=str(self.workdir),
        )

    # -- run -------------------------------------------------------------
    def run(self, plan: EnginePlan) -> RawRun:
        started = datetime.now(timezone.utc)
        path = self.workdir / self.results_filename
        artifact = b""
        output: Any = None
        exit_code = 0
        error: str | None = None
        try:
            artifact = path.read_bytes()
        except FileNotFoundError:
            exit_code, error = 2, f"no {self.results_filename} in {self.workdir}"
        except OSError as exc:
            exit_code, error = 3, f"{self.results_filename} could not be read: {type(exc).__name__}: {exc}"
        else:
            try:
                # Keep whatever bytes were read even when parsing fails: a
                # malformed file is still the artifact a reviewer needs in
                # order to see what went wrong.
                output = json.loads(artifact.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                exit_code = 3
                error = f"{self.results_filename} could not be parsed: {type(exc).__name__}: {exc}"
        return RawRun(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            output=output,
            artifact=artifact,
            invocation={
                "results_file": str(path),
                "exit_code": exit_code,
                "error": error,
                "intended_validators": [v.get("id") for v in plan.validators],
                "plan_hash": plan.plan_hash(),
            },
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    # -- normalize -------------------------------------------------------
    def normalize(self, raw: RawRun) -> list[NormalizedResult]:
        intended = list(raw.invocation.get("intended_validators") or [])
        if not isinstance(raw.output, dict):
            reason = raw.invocation.get("error") or "the results file held no JSON object"
            return [_error_result(vid, self.engine_id, reason) for vid in intended]

        entries = raw.output.get("results")
        if not isinstance(entries, list):
            return [
                _error_result(vid, self.engine_id, "the results file has no 'results' array")
                for vid in intended
            ]

        by_validator: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("validator_id"), str):
                by_validator[entry["validator_id"]] = entry

        results: list[NormalizedResult] = []
        for validator_id in intended or sorted(by_validator):
            entry = by_validator.get(validator_id)
            if entry is None:
                results.append(_error_result(
                    validator_id, self.engine_id,
                    "no entry for this validator in the results file",
                ))
                continue
            status = entry.get("status")
            if status not in Status.ALL:
                results.append(_error_result(
                    validator_id, self.engine_id,
                    f"status {status!r} is not one of {', '.join(Status.ALL)}",
                ))
                continue
            severity = entry.get("severity")
            observations = [o for o in (entry.get("observations") or []) if isinstance(o, str)]
            results.append(NormalizedResult(
                validator_id=validator_id,
                version=str(entry.get("version") or "manual"),
                status=status,
                score=_float_or_none(entry.get("score")),
                severity=severity if severity in Severity.ALL else None,
                observations=observations,
                engine=self.engine_id,
            ))
        return results

    # -- raw_artifact ----------------------------------------------------
    def raw_artifact(self, raw: RawRun) -> bytes:
        return raw.artifact


def declared_validators(campaign_bundle: dict[str, Any]) -> list[dict[str, str]]:
    """`[{"id","version"}]` for every validator the compiled bundle declares.

    Shared by both adapters. Reads the same path `RunnerEngine.__init__`
    reads (`template.protocol.validators`) so an adapter and the engine
    cannot disagree about which validators a bundle contains — a mismatch
    there is what `UnknownValidatorError` exists to catch, and catching it is
    cheaper than producing a manifest the server refuses.
    """
    declared = (((campaign_bundle or {}).get("template") or {}).get("protocol") or {}).get("validators")
    if not isinstance(declared, list):
        return []
    out: list[dict[str, str]] = []
    for entry in declared:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            out.append({"id": entry["id"], "version": str(entry.get("version") or "0.0.0")})
    return out


def _error_result(validator_id: str, engine: str, reason: str) -> NormalizedResult:
    return NormalizedResult(
        validator_id=validator_id,
        version="0.0.0",
        status=Status.ERROR,
        observations=[reason],
        engine=engine,
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
