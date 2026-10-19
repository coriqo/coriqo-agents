"""`coriqo-agents runner watch`: continuous production monitoring,
customer-side.

    coriqo-agents runner watch \\
        --execution <standing-execution-id> --bundle bundle.json \\
        --trace ./otlp --window 1h \\
        --base-url <url> --tenant <slug> --device-key key.pem [--once]

## What this does, and what it is not allowed to do

It tails a directory of OTLP/JSON exports the customer's own collector
writes, cuts them into fixed windows, runs the bundle's validators over each
window's traces, and uploads ONE window result per window: statuses, turn
counters, and — for failing validators only — payload hashes plus the
FailureScenario fields.

It is a monitor. It sits beside the agent, after the fact, reading files.
It has no inline path, no proxy, no hook, and no way to refuse a turn. That
is deliberate: inline enforcement is out of scope by design, because holding an enforcement point would break the independence the whole
product rests on. If a change to this file starts to look like it could
block a request, it is in the wrong file.

## What leaves the customer

Never a payload. Per window per validator: a status, `turns_evaluated`,
`turns_failed`, and `failure_digests` — sha256 hashes plus the scenario
prose the validator itself produced — on FAILING validators only. A clean
window uploads counters and nothing else.

## Restart safety

Window ids are content-addressed (`window_id`, a digest over the execution,
the window bounds and the bundle hash), and the server recomputes them
rather than trusting the client's. A watcher that crashes and re-reads the
same files re-derives the same ids, so the server reports them as duplicates
instead of double-counting. There is no cursor file to lose or corrupt.

Zero new dependencies: stdlib plus the `httpx` the uploader already uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from ._common import DeviceKeyError, load_device_key
from .contract import Status, ValidatorContext
from .engine import RunnerEngine
from .otel_ingest import MultipleAgentsInBatchError, otlp_json_bytes_to_traces
from .signing import DeviceSigner
from .uploader import ExecutionUploader
from .validators import resolve_validators

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UPLOAD_REJECTED = 3

# Accepted `--window` suffixes. Deliberately coarse: a monitoring window
# below a minute is a per-turn stream in disguise, and this feature's whole
# affordability argument is that the server's row rate is windows, not turns.
_WINDOW_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
MIN_WINDOW_SECONDS = 60
MAX_WINDOW_SECONDS = 86400

# Must stay <= the server's `monitoring_service.MAX_WINDOWS_PER_UPLOAD`. An
# outage leaves a backlog larger than this, and it has to be uploaded in
# chunks rather than as one request the server refuses whole.
MAX_WINDOWS_PER_UPLOAD = 200

# Failure digests kept per validator per window. The server enforces the same
# cap (it, not this file, is the trust boundary) — this keeps a well-behaved
# runner from building a payload it knows will be refused.
MAX_DIGESTS_PER_RESULT = 20


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


class WatchUsageError(Exception):
    """A bad input or a precondition checkable before touching the network."""


def parse_window(value: str) -> int:
    """'5m', '1h', '900s' -> seconds."""
    text = (value or "").strip().lower()
    if not text:
        raise WatchUsageError("--window is required, e.g. 1h")
    unit = text[-1]
    if unit not in _WINDOW_UNITS:
        raise WatchUsageError(f"--window {value!r}: expected a suffix of s, m, h or d")
    try:
        amount = int(text[:-1])
    except ValueError:
        raise WatchUsageError(f"--window {value!r}: {text[:-1]!r} is not a whole number")
    if amount <= 0:
        raise WatchUsageError("--window must be positive")
    seconds = amount * _WINDOW_UNITS[unit]
    if seconds < MIN_WINDOW_SECONDS:
        raise WatchUsageError(
            f"--window {value!r} is under {MIN_WINDOW_SECONDS}s; a shorter window is a per-turn "
            "stream, which this monitor deliberately does not send"
        )
    if seconds > MAX_WINDOW_SECONDS:
        raise WatchUsageError(f"--window {value!r} is over a day, which is not a monitoring window")
    return seconds


def compute_window_id(
    *, execution_id: str, window_start: datetime, window_end: datetime, bundle_hash: str | None,
) -> str:
    """Content-addressed window id.

    MUST stay byte-identical to the window id Coriqo's API computes. The server recomputes it and refuses a mismatch, so
    a drift between the two shows up as every window landing unverified
    rather than as silent corruption — but keeping them in step is still the
    contract, and `sdk/coriqo-agents/tests/test_monitoring_watch.py` pins the
    exact digest for a fixed input so a change on either side is caught."""
    material = "|".join([
        execution_id,
        _isoformat(window_start),
        _isoformat(window_end),
        bundle_hash or "",
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _isoformat(value: datetime) -> str:
    return _aware(value).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _event_time(event: dict[str, Any]) -> datetime | None:
    raw = event.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return _aware(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError:
        return None


def window_start_for(when: datetime, *, window_seconds: int) -> datetime:
    """The fixed grid this window falls on.

    Windows are aligned to the UTC epoch rather than to when the watcher
    happened to start. Two watchers that restart at different times cut the
    same traffic into the SAME windows, which is what makes the
    content-addressed id an actual duplicate guard rather than a
    per-process one."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed = int((_aware(when) - epoch).total_seconds())
    return epoch + timedelta(seconds=elapsed - (elapsed % window_seconds))


# --------------------------------------------------------------------- #
# Reading traces
# --------------------------------------------------------------------- #

def _mtime(file: Path) -> datetime | None:
    """The file's last-modified time, or None if it cannot be read — None
    means "keep it", because a file we cannot stat is not a file we should
    silently skip."""
    try:
        return datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _parse_iso(value: str) -> datetime:
    return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))


def load_traces(path: Path, *, since: datetime | None = None) -> list[dict[str, Any]]:
    """Every trace under `--trace`, one file or a directory of them.

    Unlike `ci.py::_load_trace`, this does NOT merge everything into a
    single trace and does not refuse a directory carrying more than one
    agent: a production trace drop legitimately contains many independent
    interactions, and each one is a turn to be evaluated on its own. Traces
    from other agents are filtered out by the caller against the execution's
    own agent identity rather than raising, because a shared collector
    directory is the normal deployment, not a mistake."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise WatchUsageError(f"no .json trace files found under {path}")
    if since is not None:
        # Skip files the collector last touched before the backfill horizon.
        # Without this a watcher running for months re-reads and re-parses the
        # ENTIRE retained trace directory on every poll — unbounded CPU and
        # memory growth on the customer's own host, which is the last place
        # this product should be the reason something gets OOM-killed. mtime
        # is the right signal for append-only collector output: a file still
        # being written to has a current mtime and is kept.
        recent = [f for f in files if _mtime(f) is None or _mtime(f) >= since]
        # Never let the horizon empty the list and turn a quiet period into
        # the "no trace files" error, which in daemon mode is just noise.
        # The fallback is the file with the newest MTIME, not the lexically
        # last one: collectors name batches by uuid or `batchN.json`, neither
        # of which sorts in time order, so `files[-1:]` could hand back a
        # stale file and have the watcher re-process it every poll forever.
        files = recent or [max(files, key=lambda f: _mtime(f) or datetime.min.replace(tzinfo=timezone.utc))]
    traces: list[dict[str, Any]] = []
    for file in files:
        try:
            data = file.read_bytes()
        except OSError as exc:
            raise WatchUsageError(f"could not read trace file {file}: {exc}") from exc
        try:
            traces.extend(otlp_json_bytes_to_traces(data))
        except MultipleAgentsInBatchError:
            # Not an error here — see this function's docstring.
            continue
        except (ValueError, TypeError) as exc:
            raise WatchUsageError(f"{file} is not a valid OTLP/JSON export: {exc}") from exc
    return traces


def bucket_traces(
    traces: Iterable[dict[str, Any]], *, window_seconds: int,
) -> dict[datetime, list[dict[str, Any]]]:
    """Group traces into aligned windows by their FIRST event's timestamp.

    First event, not last: a long-running agent interaction belongs to the
    window it began in, so a trace can never straddle two windows and be
    counted in both. A trace whose events carry no parseable timestamp is
    dropped rather than filed under 'now' — putting undated evidence in an
    arbitrary window would put a number in front of an examiner that nothing
    supports."""
    buckets: dict[datetime, list[dict[str, Any]]] = {}
    for trace in traces:
        times = [t for t in (_event_time(e) for e in (trace.get("events") or [])) if t]
        if not times:
            continue
        start = window_start_for(min(times), window_seconds=window_seconds)
        buckets.setdefault(start, []).append(trace)
    return buckets


# --------------------------------------------------------------------- #
# Evaluating a window
# --------------------------------------------------------------------- #

# A validator result that counts the turn as failed. WARNING is included at
# high/critical severity only, matching `findings_service.should_raise_
# finding` — so "a failed turn" here and "a finding worth raising" there
# mean the same thing.
_SEVERE = frozenset({"high", "critical"})


def _is_failed_turn(result) -> bool:
    if result.status == Status.FAIL:
        return True
    return result.status == Status.WARNING and (result.severity in _SEVERE)


@dataclass
class WindowEvaluation:
    """One window's uploadable payload — counters and digests, no traffic."""
    window_start: datetime
    window_end: datetime
    turn_count: int
    trace_count: int
    results: list[dict[str, Any]]

    def to_payload(self, *, execution_id: str, bundle_hash: str | None) -> dict[str, Any]:
        return {
            "window_id": compute_window_id(
                execution_id=execution_id, window_start=self.window_start,
                window_end=self.window_end, bundle_hash=bundle_hash,
            ),
            "window_start": _isoformat(self.window_start),
            "window_end": _isoformat(self.window_end),
            "bundle_hash": bundle_hash,
            "turn_count": self.turn_count,
            "trace_count": self.trace_count,
            "results": self.results,
        }


def evaluate_window(
    *,
    bundle: dict[str, Any],
    traces: list[dict[str, Any]],
    window_start: datetime,
    window_seconds: int,
    context_template: ValidatorContext,
    engine: RunnerEngine | None = None,
    validators: list | None = None,
) -> WindowEvaluation:
    """Run every declared validator over every trace in this window and
    reduce to one row per validator.

    One trace is one turn. The reduction is deterministic and local — no
    model, no server call, no judgement beyond counting:

      * `turns_evaluated` counts the traces this validator actually applied
        to (anything but NOT_APPLICABLE);
      * `turns_failed` counts FAILs plus high/critical WARNINGs;
      * `status` is only ever ERROR or INCONCLUSIVE from here — that is the
        runner saying it could not evaluate, which only the runner can know.
        The PASS/WARNING/FAIL verdict is the SERVER's to derive from the
        counts (D5/D9), and this file must not assert it.
    """
    # Built once per process by the caller when there is one; falling back to
    # per-call construction keeps this function usable on its own (the tests
    # call it directly). Rebuilding per window re-parsed the bundle and
    # re-instantiated every validator for every window in a backlog.
    if engine is None or validators is None:
        engine = RunnerEngine(bundle)
        validators = resolve_validators(bundle)
    window_end = window_start + timedelta(seconds=window_seconds)

    evaluated: dict[str, int] = {}
    failed: dict[str, int] = {}
    versions: dict[str, str] = {}
    severities: dict[str, str | None] = {}
    unevaluable: dict[str, str] = {}
    unevaluable_turns: dict[str, int] = {}
    digests: dict[str, list[dict[str, Any]]] = {}

    for trace in traces:
        context = ValidatorContext(
            execution_id=context_template.execution_id,
            agent_model_id=context_template.agent_model_id,
            agent_version_ref=context_template.agent_version_ref,
            environment=context_template.environment,
            configuration=dict(context_template.configuration or {}),
            inputs={"trace": trace},
        )
        results, _manifest, _evidence, scenarios = engine.run(validators, context)
        scenario_by_index = {s.get("result_index"): s for s in (scenarios or [])}
        for index, result in enumerate(results):
            vid = result.validator_id
            versions[vid] = result.version
            if result.status in (Status.ERROR, Status.INCONCLUSIVE):
                # The runner could not evaluate this turn. COUNTED, not just
                # noted: an earlier version only remembered that it happened,
                # so a validator that errored on 999 of 1000 turns and passed
                # the one remaining turn uploaded `turns_evaluated=1,
                # turns_failed=0` with no status, and the server derived PASS.
                # A broken validator read as a clean one — exactly the failure
                # this branch exists to prevent.
                unevaluable[vid] = result.status
                unevaluable_turns[vid] = unevaluable_turns.get(vid, 0) + 1
                continue
            if result.status == Status.NOT_APPLICABLE:
                continue
            evaluated[vid] = evaluated.get(vid, 0) + 1
            if _is_failed_turn(result):
                failed[vid] = failed.get(vid, 0) + 1
                if result.severity:
                    severities[vid] = result.severity
                # Capped on INSERT, not on output. Building the full list and
                # slicing it later meant memory scaled with the size of the
                # incident being reported: a validator failing on most turns
                # of a high-volume window would accumulate one dict per
                # failed turn, so the watcher was most likely to be
                # OOM-killed exactly when its report mattered most. The count
                # (`turns_failed`) is what carries the scale; the digests are
                # a sample.
                sample = digests.setdefault(vid, [])
                if len(sample) < MAX_DIGESTS_PER_RESULT:
                    sample.append(_failure_digest(
                        trace=trace, result=result, scenario=scenario_by_index.get(index),
                    ))

    rows: list[dict[str, Any]] = []
    for vid in sorted(set(evaluated) | set(unevaluable)):
        row: dict[str, Any] = {
            "validator_id": vid,
            "validator_version": versions.get(vid, "0.0.0"),
            "turns_evaluated": evaluated.get(vid, 0),
            "turns_failed": failed.get(vid, 0),
        }
        # The window is INCONCLUSIVE whenever the turns the runner could not
        # evaluate outnumber the ones it could — including the case where it
        # evaluated none at all. Below that, the counts stand on their own and
        # the server derives the verdict.
        if unevaluable_turns.get(vid, 0) >= max(1, evaluated.get(vid, 0)):
            row["status"] = unevaluable[vid]
        if vid in severities:
            row["severity"] = severities[vid]
        if vid in digests:
            # Already capped at insert (above) — a window where a validator
            # failed 40,000 turns carries the count and a sample of 20, never
            # 40,000 digests, which would be both an unbounded JSONB column
            # on a row the server keeps for 90 days and unbounded memory here.
            row["failure_digests"] = digests[vid]
        rows.append(row)

    turn_count = sum(
        1 for trace in traces
        for event in (trace.get("events") or [])
        if event.get("event_type") == "USER_REQUEST"
    ) or len(traces)

    return WindowEvaluation(
        window_start=window_start, window_end=window_end,
        turn_count=turn_count, trace_count=len(traces), results=rows,
    )


def _failure_digest(*, trace: dict[str, Any], result, scenario: dict[str, Any] | None) -> dict[str, Any]:
    """A hashed pointer to a failing turn, plus the FailureScenario fields.

    The trace id is hashed, not sent: a raw trace id is a join key back into
    the bank's own observability stack, and a monitor that ships join keys
    for production traffic is shipping traffic by another name. Observations
    are the validator's own fixed prose about what rule was broken, never a
    payload excerpt."""
    digest: dict[str, Any] = {
        "trace_sha256": hashlib.sha256(str(trace.get("trace_id") or "").encode("utf-8")).hexdigest(),
        "severity": result.severity,
        "observations": [str(o) for o in (result.observations or [])][:3],
    }
    if scenario:
        for key in ("category", "threat", "trigger", "expected_behavior",
                    "forbidden_behavior", "observed_behavior"):
            if scenario.get(key):
                digest[key] = scenario[key]
    return digest


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coriqo-agents runner watch",
        description=(
            "Evaluate production traces window by window and upload the "
            "aggregate to a standing Coriqo assurance execution."
        ),
        epilog=(
            "Exit codes: 0 windows uploaded (or nothing new to send); "
            "2 usage/input error; 3 an upload was rejected by the server.\n"
            "This command monitors. It cannot block, gate or slow the agent "
            "it watches."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--execution", required=True, help="The standing (streaming) execution id.")
    parser.add_argument("--bundle", required=True, type=Path, help="The signed protocol bundle.")
    parser.add_argument("--trace", required=True, type=Path,
                        help="An OTLP/JSON export file, or a directory of them.")
    parser.add_argument("--window", default="1h", help="Window size: 5m, 1h, 1d. Default 1h.")
    parser.add_argument("--base-url", help="Coriqo base URL.")
    parser.add_argument("--tenant", help="Tenant slug.")
    parser.add_argument("--device-key", type=Path, help="Ed25519 private key PEM for this device.")
    # Several shipped validators need configuration to assess anything at
    # all — `tool_authorization` reports INCONCLUSIVE without an
    # `authorized_tools` roster rather than manufacturing a PASS. Without
    # this flag a watch run would disclose "could not evaluate" forever,
    # which is honest but useless.
    parser.add_argument("--config", type=Path,
                        help="JSON file merged into every validator's context.configuration "
                             "(e.g. {\"authorized_tools\": [...]}).")
    parser.add_argument("--agent-model-id", default="")
    parser.add_argument("--agent-version-ref", default="")
    parser.add_argument("--environment", default="PRODUCTION")
    parser.add_argument("--max-backfill-days", type=int, default=30,
                        help="Ignore trace files and windows older than this. Bounds a "
                             "long-running watcher's memory and per-poll work. Default 30.")
    parser.add_argument("--poll-seconds", type=int, default=60,
                        help="How often to re-read --trace when looping. Default 60.")
    parser.add_argument("--once", action="store_true",
                        help="Evaluate what is on disk now, upload, and exit. "
                             "Use this from cron instead of running a daemon.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate and print the window payloads without uploading. "
                             "Nothing leaves the machine.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        window_seconds = parse_window(args.window)
    except WatchUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.max_backfill_days < 1:
        print("error: --max-backfill-days must be at least 1", file=sys.stderr)
        return EXIT_USAGE

    if not args.dry_run and not (args.base_url and args.tenant and args.device_key):
        print("error: --base-url, --tenant and --device-key are required unless --dry-run",
              file=sys.stderr)
        return EXIT_USAGE

    try:
        bundle = json.loads(args.bundle.read_text())
    except (OSError, ValueError) as exc:
        print(f"error: could not read --bundle {args.bundle}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not isinstance(bundle, dict):
        print(f"error: --bundle {args.bundle} is not a JSON object", file=sys.stderr)
        return EXIT_USAGE
    bundle_hash = bundle.get("bundle_hash")

    uploader = None
    if not args.dry_run:
        try:
            private_key = load_device_key(args.device_key)
        except DeviceKeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        uploader = ExecutionUploader(
            base_url=args.base_url, signer=DeviceSigner(private_key=private_key),
            tenant_slug=args.tenant,
        )

    configuration: dict[str, Any] = {}
    if args.config:
        try:
            configuration = json.loads(args.config.read_text())
        except (OSError, ValueError) as exc:
            print(f"error: could not read --config {args.config}: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if not isinstance(configuration, dict):
            print(f"error: --config {args.config} is not a JSON object", file=sys.stderr)
            return EXIT_USAGE

    context = ValidatorContext(
        execution_id=args.execution, agent_model_id=args.agent_model_id,
        agent_version_ref=args.agent_version_ref, environment=args.environment,
        configuration=configuration,
    )
    # Built once, not per window. Also surfaces a bad bundle or an unknown
    # validator id as a usage error now, rather than on the first window
    # after the watcher has already been running for an hour.
    try:
        engine = RunnerEngine(bundle)
        validators = resolve_validators(bundle)
    except (ValueError, KeyError) as exc:
        print(f"error: --bundle {args.bundle} is not usable: {exc}", file=sys.stderr)
        return EXIT_USAGE
    # Windows already sent in THIS process, mapped to their start so the set
    # can be pruned. Only an optimisation — the server's content-addressed
    # guard is what actually prevents double counting, and it survives a
    # restart, which this does not.
    #
    # A plain unbounded set was a slow leak: a watcher is meant to run for
    # months, and at a 5-minute window that is ~105k ids a year, held forever
    # alongside a trace directory re-read every poll. Pruned below on the
    # same horizon the backfill window uses.
    sent: dict[str, datetime] = {}

    while True:
        horizon = datetime.now(timezone.utc) - timedelta(days=args.max_backfill_days)
        try:
            traces = load_traces(args.trace, since=horizon)
        except WatchUsageError as exc:
            # In daemon mode this must NOT kill the process. The collector
            # directory is momentarily empty during log rotation, a file is
            # half-written, a mount blips — all transient, all recoverable on
            # the next poll. Exiting here stops monitoring silently, and
            # silence from a monitor reads as "nothing wrong", which is worse
            # than no monitor at all. `--once` still returns the exit code,
            # because there a caller (cron, a pipeline) is watching for it.
            print(f"error: {exc}", file=sys.stderr)
            if args.once:
                return EXIT_USAGE
            print(f"retrying in {max(1, args.poll_seconds)}s", file=sys.stderr)
            time.sleep(max(1, args.poll_seconds))
            continue

        payloads: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        current = window_start_for(now, window_seconds=window_seconds)
        for start, window_traces in sorted(bucket_traces(traces, window_seconds=window_seconds).items()):
            # Never upload the window that is still filling: it would be
            # sent again next poll with more turns in it, and the server
            # would take the first version as final because the id is the
            # same. Only closed windows are complete evidence.
            if start >= current:
                continue
            # Older than the backfill horizon: the server's own retention
            # would prune it anyway, and evaluating it every poll forever is
            # the unbounded work `--max-backfill-days` exists to stop.
            if start < horizon:
                continue
            # Skip BEFORE evaluating. The window id depends only on the
            # execution, the bounds and the bundle hash — never on the
            # evaluation — so checking after was pure waste: in daemon mode
            # the watcher re-read the whole trace directory each poll and
            # re-ran every validator over every historical window, once a
            # minute by default, then threw all of it away. With a month of
            # retained traces that is the entire month re-validated every
            # poll.
            window_id = compute_window_id(
                execution_id=args.execution, window_start=start,
                window_end=start + timedelta(seconds=window_seconds),
                bundle_hash=bundle_hash,
            )
            if window_id in sent:
                continue
            evaluation = evaluate_window(
                bundle=bundle, traces=window_traces, window_start=start,
                window_seconds=window_seconds, context_template=context,
                engine=engine, validators=validators,
            )
            payloads.append(evaluation.to_payload(
                execution_id=args.execution, bundle_hash=bundle_hash,
            ))

        if payloads:
            if args.dry_run:
                print(json.dumps({"windows": payloads}, indent=2, sort_keys=True))
            else:
                # Chunked to the server's cap. A backlog larger than that —
                # exactly the outage-recovery case this feature plans for —
                # used to be sent as one over-cap request, get a 400 for the
                # WHOLE batch, and leave the watcher unable to ever drain it.
                delivered: list[dict[str, Any]] = []
                try:
                    for batch in _chunks(payloads, MAX_WINDOWS_PER_UPLOAD):
                        response = uploader.upload_monitoring_windows(args.execution, batch)
                        print(json.dumps(response, indent=2, sort_keys=True))
                        delivered.extend(batch)
                except RuntimeError as exc:
                    # Same reasoning as the trace-read failure above: a
                    # rotated key, a 502 from a proxy, a network blip must not
                    # end a monitor that is supposed to run for months. Only
                    # the windows that actually landed go into `sent`, so the
                    # rest are retried next poll — and the server's
                    # content-addressed guard makes that retry free.
                    print(f"error: {exc}", file=sys.stderr)
                    if args.once:
                        return EXIT_UPLOAD_REJECTED
                    print(f"retrying in {max(1, args.poll_seconds)}s", file=sys.stderr)
                    payloads = delivered
            for payload in payloads:
                sent[payload["window_id"]] = _parse_iso(payload["window_start"])
        elif args.once:
            print("No closed windows to report.")

        if args.once:
            return EXIT_OK

        # Bound the dedup set to the same horizon the backfill uses. Anything
        # older can no longer be produced by `load_traces`, so remembering it
        # buys nothing and costs memory for the life of the process.
        for window_id in [k for k, v in sent.items() if v < horizon]:
            del sent[window_id]

        time.sleep(max(1, args.poll_seconds))
