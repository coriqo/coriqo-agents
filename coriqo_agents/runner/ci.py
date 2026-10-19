"""`coriqo-agents runner ci`: gate a CI pipeline on Coriqo's own
sealed assurance verdict.

This is the hook a bank's pipeline calls on every agent version. It runs the
campaign's protocol validators against a recorded trace, uploads the signed
results exactly the way `coriqo-agents runner run` does (same device
signature, same manifest shape), then reads back the verdict CORIQO computed
— `GET /assurance/{execution_id}/diff` and its control determinations — and
turns that into a process exit code and a JUnit XML report any CI tool
already knows how to show.

Referee boundary: this module never computes a
pass/fail score of its own. Two different kinds of "decided" cross this
file, and neither originates here:

  1. Each validator's own PASS/FAIL/WARNING/... — decided by the validator
     (`Validator.evaluate`, phase-3 code, deterministic, no LLM), the same
     as any other runner invocation. This file runs that code because
     something customer-side has to; it does not re-interpret the result.
  2. `regressed`/`improved`/`unchanged` and each control's determination —
     decided by the SERVER (`diff_service.py`, a pure table lookup over
     statuses the validators already produced) and fetched over the wire,
     never recomputed locally. `_gate()` below only reads
     `diff["counts"]` and `diff["control_determinations"]` that the server
     returned; it contains no comparison logic of its own.

No secret ever reaches stdout/stderr, the JSON payload, or the JUnit file:
the API key and device key are read from files/args and used only inside
request headers this module builds and sends, never echoed back. See
`sdk/coriqo-agents/tests/test_ci_no_secrets.py` for the automated check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import httpx

from ._common import DeviceKeyError, load_device_key
from .contract import ValidatorContext
from .validators import resolve_validators
from .engine import InvalidBundleError, RunnerEngine, UnknownValidatorError
from .otel_ingest import MultipleAgentsInBatchError, otlp_json_bytes_to_trace
from .signing import DeviceSigner
from .uploader import ExecutionUploader

FAIL_ON_CHOICES = ("regressed", "ineffective")

#: Exit codes. Kept small and documented here rather than scattered through
#: the function bodies below, since a CI job's next step branches on these.
#:
#:   0  the run completed and the sealed verdict was retrieved; no
#:      --fail-on condition was met.
#:   1  a real gate failure: the sealed verdict was retrieved and at least
#:      one --fail-on condition was met (a validator regressed, or a
#:      control went INEFFECTIVE).
#:   2  a usage/input error: bad flags, an unreadable bundle/trace/device
#:      key, a campaign in the wrong state, an unknown validator, etc. —
#:      nothing was necessarily run against the target, and this exit code
#:      says "fix the invocation", not "the agent regressed".
#:   3  the upload of results was rejected by the server.
#:   4  the sealed verdict could not be retrieved at all — auth failure,
#:      server error, network/timeout, or any other non-409 error from the
#:      diff endpoint. This is NOT a pass: the gate was never evaluated, so
#:      the pipeline must fail closed rather than print "GATE PASSED".
EXIT_OK = 0
EXIT_GATE_FAILED = 1        # a --fail-on condition was met against a retrieved verdict
EXIT_USAGE_ERROR = 2        # bad input/usage — renumbered from the old EXIT_GATE_FAILED reuse
EXIT_UPLOAD_REJECTED = 3    # renumbered: was 2, now distinct from EXIT_USAGE_ERROR
EXIT_GATE_UNAVAILABLE = 4   # the verdict was never retrieved — fail closed, not "GATE PASSED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coriqo-agents runner ci",
        description=(
            "Run this campaign's assurance validators against a recorded trace and "
            "gate the pipeline on Coriqo's own sealed verdict."
        ),
        epilog=(
            "Exit codes:\n"
            "  0  no --fail-on condition was met; the sealed verdict was retrieved.\n"
            "  1  gate failed: the sealed verdict showed a regression or an INEFFECTIVE\n"
            "     control matching --fail-on.\n"
            "  2  usage error: bad flags, an unreadable bundle/trace/device key, a\n"
            "     campaign in the wrong state, or an unknown validator. Nothing was\n"
            "     necessarily graded — fix the invocation and rerun.\n"
            "  3  the signed results upload was rejected by the server.\n"
            "  4  the sealed verdict could not be retrieved at all (auth failure,\n"
            "     server error, network/timeout, malformed execution id). This is\n"
            "     NOT a pass — the gate was never evaluated.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--campaign", required=True, help="Campaign id this run belongs to.")
    parser.add_argument(
        "--agent-version", required=True, dest="agent_version",
        help="Agent version ref under test. Must match the campaign's own agent_version_ref "
             "— governance approved the campaign for one version, and this flag is checked "
             "against it before anything runs, rather than silently testing whatever the "
             "campaign happens to point at.",
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Compiled protocol bundle JSON.")
    parser.add_argument(
        "--trace", required=True, type=Path,
        help="An OTLP/HTTP+JSON trace export file, or a directory of them (all merged, same "
             "agent identity required).",
    )
    parser.add_argument("--engine", default="native", choices=("native",),
                         help="Validators run natively against the trace — no third-party tool in this path.")
    parser.add_argument("--environment", default=None)
    parser.add_argument(
        "--execution", default=None,
        help="Attach to a run Coriqo already created (the id in a run-trigger message) "
             "instead of creating a new one. The run must still be waiting; a finished "
             "run is refused.",
    )
    parser.add_argument("--base-url", required=True, help="Coriqo base URL.")
    parser.add_argument("--tenant", required=True, help="Tenant slug.")
    parser.add_argument(
        "--api-key", required=True,
        help="Service account X-API-Key — creates the execution and reads the sealed diff. "
             "Read from an environment variable in CI, never hardcoded in the pipeline file.",
    )
    parser.add_argument(
        "--device-key", required=True, type=Path,
        help="PEM file holding this host's enrolled Ed25519 private key — signs the manifest/results upload.",
    )
    parser.add_argument(
        "--pipeline-id", default=None,
        help="This CI run's own id (e.g. $GITHUB_RUN_ID, $CI_PIPELINE_ID) — shown on the Runs "
             "list so two CI runs of the same agent version are never indistinguishable.",
    )
    parser.add_argument(
        "--fail-on", action="append", choices=FAIL_ON_CHOICES, dest="fail_on", default=None,
        help="Fail the pipeline when the sealed diff shows this. Repeatable "
             "(--fail-on regressed --fail-on ineffective). Default: both.",
    )
    parser.add_argument(
        "--against", default=None,
        help="Execution id to diff against. Default: the server's own rule — this run's "
             "retest target if any, else the campaign's previous completed/failed run.",
    )
    parser.add_argument(
        "--retain-trace", action="store_true", dest="retain_trace",
        help="After the results upload, also send the trace so Coriqo can keep it for re-performance "
             "(`coriqo-agents runner verify`). Only works when an org admin has turned on trace "
             "retention in Settings; otherwise one line says so and the run continues. Never changes "
             "the exit code.",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between status polls.")
    parser.add_argument("--poll-timeout", type=float, default=120.0, help="Seconds to wait for the execution to complete.")
    parser.add_argument("--junit", type=Path, default=None, help="Write a JUnit XML report here.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the result as JSON instead of a table.")
    return parser


class _CiApiClient:
    """Plain X-API-Key calls against the campaign/diff endpoints — separate
    from `ExecutionUploader` (device-signed) because these two endpoint
    families use different auth (see runner_router.py's own note that the
    two routers intentionally sit on different auth planes)."""

    def __init__(self, base_url: str, tenant: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key, "X-Tenant-Slug": tenant}

    def get(self, path: str) -> dict:
        try:
            resp = httpx.get(f"{self._base_url}{path}", headers=self._headers, timeout=30)
        except httpx.HTTPError as exc:
            raise CiApiError(None, f"could not reach Coriqo at {self._base_url}{path}: {exc}") from exc
        return self._parse(resp)

    def post(self, path: str, body: dict) -> dict:
        try:
            resp = httpx.post(f"{self._base_url}{path}", headers=self._headers, json=body, timeout=30)
        except httpx.HTTPError as exc:
            raise CiApiError(None, f"could not reach Coriqo at {self._base_url}{path}: {exc}") from exc
        return self._parse(resp)

    @staticmethod
    def _parse(resp: httpx.Response) -> dict:
        if resp.status_code >= 400:
            raise CiApiError(resp.status_code, f"Coriqo rejected the request ({resp.status_code}): {resp.text}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            # A 2xx with a body that isn't JSON — a proxy/WAF challenge page
            # or a load balancer's HTML error page, most often. Not a
            # `CiApiError` from the status code, but it must still flow
            # through the same exit-code contract rather than raising
            # `json.JSONDecodeError` straight out of `main()`.
            raise CiApiError(
                resp.status_code, f"server returned a non-JSON response ({resp.status_code}): {exc}"
            ) from exc


class CiApiError(Exception):
    """An HTTP error from the Coriqo API, carrying the actual status code so
    callers can branch on it — e.g. gate-check logic treats 409 ("no
    baseline yet") completely differently from every other status. A bare
    `RuntimeError` used to erase that distinction, which is exactly what let
    a 401/500/network error get silently treated as the documented 409
    no-baseline case."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class CiUsageError(Exception):
    """A bad input or a precondition this CLI can check before touching the
    network — a real target's problem, not a crash."""


def _load_trace(path: Path) -> dict[str, Any]:
    """One merged trace dict from `--trace`: a single OTLP/JSON export file,
    or a directory of them. All files must resolve to the same agent
    identity — merging two different agents' events into one trace would
    misattribute one agent's evidence to another's execution, the same
    concern `otlp_json_to_trace` already raises `MultipleAgentsInBatchError`
    for within a single file."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise CiUsageError(f"no .json trace files found under {path}")
    merged: dict[str, Any] | None = None
    for file in files:
        try:
            data = file.read_bytes()
        except OSError as exc:
            raise CiUsageError(f"could not read trace file {file}: {exc}") from exc
        try:
            trace = otlp_json_bytes_to_trace(data)
        except MultipleAgentsInBatchError as exc:
            raise CiUsageError(f"{file}: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise CiUsageError(f"{file} is not a valid OTLP/JSON export: {exc}") from exc
        if merged is None:
            merged = trace
            continue
        if trace["agent_id"] and merged["agent_id"] and trace["agent_id"] != merged["agent_id"]:
            raise CiUsageError(
                f"{file} carries agent_id {trace['agent_id']!r}, but an earlier file under "
                f"{path} carried {merged['agent_id']!r} — a CI trace directory must be one "
                "agent's own export, not a mix"
            )
        merged["events"] = list(merged.get("events") or []) + list(trace.get("events") or [])
    return merged or {"trace_id": "", "agent_id": "", "agent_version": "", "events": []}


def _load_device_key(path: Path):
    """Thin wrapper over the shared loader (`_common.load_device_key`) that
    converts its `DeviceKeyError` into this module's own `CiUsageError`, the
    same way `__main__.py` converts it into `SystemExit`."""
    try:
        return load_device_key(path)
    except DeviceKeyError as exc:
        raise CiUsageError(str(exc)) from exc


def _retain_trace(uploader: ExecutionUploader, execution_id: str, trace: dict) -> None:
    """`--retain-trace`: send the trace if the workspace keeps traces. Every
    outcome is one plain line; none of them can change the gate's exit code,
    because a trace that was not kept does not make the agent's result worse."""
    try:
        policy = uploader.get_trace_policy(execution_id)
    except RuntimeError as exc:
        print(f"trace not kept: could not read the workspace's trace retention setting ({exc})", file=sys.stderr)
        return
    if not policy.get("enabled"):
        print(
            "trace not kept: trace retention is off for this workspace. An org admin can turn it on in "
            "Settings > Integrations > Runners and CI. The run continues.",
            file=sys.stderr,
        )
        return
    try:
        sent = uploader.upload_trace(execution_id, trace)
    except RuntimeError as exc:
        print(f"trace not kept: {exc}", file=sys.stderr)
        return
    print(f"trace kept until {sent.get('expires_at', 'the retention date')} (sha256 {str(sent.get('sha256', ''))[:12]}...)")


def _wait_for_completion(api: _CiApiClient, execution_id: str, *, interval: float, timeout: float) -> dict:
    """Poll `GET /assurance/{id}` (the execution's bare read, not the diff)
    until the SERVER marks the execution completed or failed — the upload
    endpoint decides that status synchronously today, but polling rather
    than trusting the upload response is what keeps this correct if
    ingestion ever becomes asynchronous, and it costs one extra round trip
    when it hasn't."""
    deadline = time.monotonic() + timeout
    execution = api.get(f"/api/v1/assurance/{execution_id}")
    while execution.get("status") not in ("completed", "failed"):
        if time.monotonic() >= deadline:
            raise CiUsageError(
                f"execution {execution_id} did not reach a final status within {timeout:.0f}s "
                f"(still {execution.get('status')!r}) — see the Execution Detail page"
            )
        time.sleep(interval)
        execution = api.get(f"/api/v1/assurance/{execution_id}")
    return execution


class GateUnavailableError(Exception):
    """The diff body Coriqo returned is missing or malformed the one field
    the gate must read (`counts`) — a verdict was never actually retrieved,
    even though the HTTP call itself came back 2xx. Callers must route this
    into the same fail-closed `EXIT_GATE_UNAVAILABLE` path as a network/auth
    failure, not let it surface as an uncaught `KeyError`."""


def _gate(diff: dict, fail_on: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Whether the sealed diff Coriqo returned crosses this caller's
    threshold, and why. Reads `diff.get("counts")`/`diff.get("control_determinations")`
    only — both already-decided server output, nothing computed here.

    Raises `GateUnavailableError` if `counts` is missing or not a dict: that
    means the response body doesn't actually carry a verdict, so the gate
    was never evaluated and must not be reported as pass or fail."""
    counts = diff.get("counts")
    if not isinstance(counts, dict):
        raise GateUnavailableError(
            f"diff response had no usable 'counts' field (got {counts!r})"
        )
    reasons: list[str] = []
    if "regressed" in fail_on:
        regressed = counts.get("regressed", 0)
        if regressed:
            reasons.append(f"{regressed} validator(s) regressed against the baseline")
    if "ineffective" in fail_on:
        determinations = diff.get("control_determinations") or []
        ineffective = []
        unresolved = 0
        for d in determinations:
            if d.get("after") != "INEFFECTIVE":
                continue
            control_id = d.get("control_id")
            if control_id is None:
                # Fail closed: a determination naming no control can't be
                # matched to anything an approver signed off on, so it's
                # treated the same as an INEFFECTIVE one rather than
                # skipped — we can't verify what this control is, so we
                # don't trust it.
                unresolved += 1
            else:
                ineffective.append(control_id)
        if ineffective:
            reasons.append(f"control(s) now INEFFECTIVE: {', '.join(sorted(ineffective))}")
        if unresolved:
            reasons.append(
                f"{unresolved} determination(s) had no control_id and were treated as failing"
            )
    return (len(reasons) > 0, reasons)


def _junit_xml(*, execution_id: str, campaign_id: str, diff: dict | None, failed: bool, reasons: list[str]) -> str:
    """One `<testsuite>` per run, one `<testcase>` per validator the diff
    compares — so a CI tool's native JUnit view lists validators the same
    way Execution Detail does, not a single opaque pass/fail line."""
    rows = (diff.get("validators") or []) if diff else []
    if not isinstance(rows, list):
        # A malformed body (validators present but not a list) — degrade to
        # "no rows" rather than crash the JUnit write. The gate's pass/fail
        # exit code is decided before this function is ever called, so a
        # bad `validators` shape here must not be able to take that verdict
        # down with it.
        rows = []
    suite = ET.Element("testsuite", {
        "name": "coriqo-assurance-ci",
        "tests": str(len(rows)),
        "failures": str(sum(1 for r in rows if r.get("change") == "regressed")),
    })
    props = ET.SubElement(suite, "properties")
    for key, value in (("execution_id", execution_id), ("campaign_id", campaign_id)):
        ET.SubElement(props, "property", {"name": key, "value": value})
    for row in rows:
        # `.get()` throughout: this dict comes off the wire, and a
        # malformed row here shouldn't crash JUnit generation and take the
        # whole exit-code decision down with it — better a report with
        # "unknown" than no report at all.
        validator_id = row.get("validator_id", "unknown")
        case = ET.SubElement(suite, "testcase", {
            "classname": "assurance.validator", "name": validator_id,
        })
        if row.get("change") == "regressed":
            before, after = row.get("before"), row.get("after")
            ET.SubElement(case, "failure", {
                "message": f"{validator_id}: {before} -> {after}",
            }).text = escape(f"regressed from {before} to {after}")
    if not rows:
        # No baseline yet, or the diff could not be computed — still a real
        # JUnit file (a CI tool showing "0 tests" beats a missing report),
        # carrying the reason as a system-out rather than a fabricated case.
        ET.SubElement(suite, "system-out").text = escape(
            "; ".join(reasons) if reasons else "no baseline to compare against yet"
        )
    return '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(suite, encoding="unicode")


def _write_junit(path: Path, xml: str) -> None:
    """Write the JUnit report, turning an unwritable `--junit` path (bad
    directory, no permission, disk full) into a stderr note rather than a
    raw traceback. This runs after the gate decision is already made, so it
    never changes the exit code — it only decides whether a write failure
    here also crashes the process."""
    try:
        path.write_text(xml, encoding="utf-8")
    except OSError as exc:
        print(f"could not write the JUnit report to {path}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fail_on = tuple(args.fail_on) if args.fail_on else FAIL_ON_CHOICES

    try:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"the protocol bundle could not be read from {args.bundle}: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    if not isinstance(bundle, dict):
        print(f"the protocol bundle in {args.bundle} must be a JSON object", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        trace = _load_trace(args.trace)
        private_key = _load_device_key(args.device_key)
    except CiUsageError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR

    api = _CiApiClient(args.base_url, args.tenant, args.api_key)

    try:
        campaign = api.get(f"/api/v1/assurance/campaigns/{args.campaign}")
    except CiApiError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR
    allowed_statuses = ("approved", "running", "review") if args.execution else ("approved", "running")
    if campaign.get("status") not in allowed_statuses:
        print(
            f"campaign {args.campaign} is {campaign.get('status')!r} — a CI run needs an "
            "approved (or already-running) campaign, same rule Campaign Detail's 'Run this "
            "in CI' snippet enforces",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    if campaign.get("agent_version_ref") != args.agent_version:
        print(
            f"--agent-version {args.agent_version!r} does not match campaign {args.campaign}'s "
            f"approved agent_version_ref {campaign.get('agent_version_ref')!r} — governance "
            "approved this campaign for one version; testing a different one needs a new "
            "campaign, not a flag override",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    if args.execution:
        try:
            existing = api.get(f"/api/v1/assurance/{args.execution}")
        except CiApiError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE_ERROR
        if existing.get("status") not in ("pending", "running"):
            print(
                f"execution {args.execution} is {existing.get('status')!r} — only a run that is "
                "still waiting can be attached to. Start a new run without --execution.",
                file=sys.stderr,
            )
            return EXIT_USAGE_ERROR
        created = {"execution": {"id": args.execution}}
    else:
        try:
            created = api.post(
                f"/api/v1/assurance/campaigns/{args.campaign}/executions",
                {
                    "engine": args.engine, "environment": args.environment,
                    "source": "ci", "pipeline_id": args.pipeline_id,
                },
            )
        except CiApiError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE_ERROR
    execution_id = (created.get("execution") or {}).get("id")
    if not execution_id:
        print(
            f"Coriqo's response to creating the execution had no execution.id: {created!r}",
            file=sys.stderr,
        )
        return EXIT_GATE_UNAVAILABLE

    try:
        engine = RunnerEngine(bundle)
        validators = resolve_validators(bundle)
    except InvalidBundleError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR

    context = ValidatorContext(
        execution_id=execution_id, agent_model_id=campaign.get("agent_model_id", ""),
        agent_version_ref=args.agent_version, environment=args.environment,
        inputs={"trace": trace},
    )
    try:
        results, manifest, evidence, failure_scenarios = engine.run(validators, context)
    except UnknownValidatorError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR

    signer = DeviceSigner(private_key=private_key)
    uploader = ExecutionUploader(base_url=args.base_url, signer=signer, tenant_slug=args.tenant)
    try:
        uploader.upload_manifest(execution_id, manifest)
        uploader.upload_results(execution_id, [r.to_dict() for r in results], evidence, failure_scenarios)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_UPLOAD_REJECTED

    if args.retain_trace:
        _retain_trace(uploader, execution_id, trace)

    try:
        _wait_for_completion(api, execution_id, interval=args.poll_interval, timeout=args.poll_timeout)
    except CiUsageError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR
    except CiApiError as exc:
        # A poll can 401 (rotated key), 404 (execution id vanished), or 5xx
        # mid-run — none of that is "fix your flags", and none of it means
        # the gate passed. Same fail-closed exit as the diff-fetch failure
        # below.
        print(
            f"the sealed verdict could not be retrieved for execution {execution_id}: {exc} "
            "— the gate was never evaluated, so this run fails closed rather than passing",
            file=sys.stderr,
        )
        return EXIT_GATE_UNAVAILABLE

    detail_url = f"{args.base_url.rstrip('/')}/executions/{execution_id}?campaign={args.campaign}"
    diff_path = f"/api/v1/assurance/{execution_id}/diff" + (f"?against={args.against}" if args.against else "")
    diff: dict | None = None
    failed = False
    reasons: list[str] = []
    try:
        diff = api.get(diff_path)
    except CiApiError as exc:
        if exc.status_code == 409:
            # The documented shape of a campaign's FIRST CI run ever —
            # nothing to regress against is not a regression, so this does
            # not fail the pipeline; every later run has a baseline.
            reasons = [str(exc)]
        else:
            # Any other failure (401/403 auth, 5xx, a network/timeout error,
            # a malformed execution id, ...) means the sealed verdict was
            # NEVER retrieved. That is not the same thing as "no baseline
            # yet" and must not be reported as a pass — fail closed with a
            # dedicated exit code instead of silently printing GATE PASSED.
            print(
                f"the sealed verdict could not be retrieved for execution {execution_id}: {exc} "
                "— the gate was never evaluated, so this run fails closed rather than passing",
                file=sys.stderr,
            )
            if args.junit:
                _write_junit(
                    args.junit,
                    _junit_xml(
                        execution_id=execution_id, campaign_id=args.campaign, diff=None,
                        failed=True, reasons=[f"verdict not retrieved: {exc}"],
                    ),
                )
            return EXIT_GATE_UNAVAILABLE
    else:
        try:
            failed, reasons = _gate(diff, fail_on)
        except GateUnavailableError as exc:
            # `diff` came back 2xx but without a usable `counts` field — the
            # gate was never actually evaluated. Same fail-closed exit as
            # every other "verdict not retrieved" branch above.
            print(
                f"the sealed verdict could not be retrieved for execution {execution_id}: {exc} "
                "— the gate was never evaluated, so this run fails closed rather than passing",
                file=sys.stderr,
            )
            if args.junit:
                _write_junit(
                    args.junit,
                    _junit_xml(
                        execution_id=execution_id, campaign_id=args.campaign, diff=None,
                        failed=True, reasons=[f"verdict not retrieved: {exc}"],
                    ),
                )
            return EXIT_GATE_UNAVAILABLE

    # The gate decision (`failed`, `reasons`) is final at this point — the
    # JUnit report is a best-effort side artifact and must never be able to
    # change the exit code this function returns. Any failure building or
    # writing it degrades to a stderr note, not a crash.
    if args.junit:
        try:
            junit_xml = _junit_xml(
                execution_id=execution_id, campaign_id=args.campaign, diff=diff, failed=failed, reasons=reasons,
            )
        except Exception as exc:  # noqa: BLE001 - last-resort guard, see comment above
            print(f"could not build the JUnit report: {exc}", file=sys.stderr)
        else:
            _write_junit(args.junit, junit_xml)

    payload = {
        "execution_id": execution_id, "campaign_id": args.campaign,
        "detail_url": detail_url, "gate_failed": failed, "reasons": reasons,
        "diff": diff,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Execution: {execution_id}")
        print(f"Detail:    {detail_url}")
        if diff is None:
            print(f"Diff:      not available ({reasons[0] if reasons else 'unknown reason'})")
        else:
            print(f"Diff:      against {diff.get('against_execution_id', 'unknown')}")
            for row in diff.get("validators") or []:
                print(
                    f"  {row.get('validator_id', 'unknown'):<32} {row.get('before') or '—':<12} "
                    f"-> {row.get('after') or '—':<12} [{row.get('change', 'unknown')}]"
                )
        if failed:
            print(f"GATE FAILED: {'; '.join(reasons)}")
        else:
            print("GATE PASSED")

    return EXIT_GATE_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
