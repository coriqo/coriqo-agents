"""`coriqo-agents runner watch`, the customer-side monitor.

Runs standalone, same convention as its siblings in this directory:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_monitoring_watch.py

What is pinned here, and why each one matters:

  * the window-id digest, byte for byte, against the server's own formula —
    the two are separate implementations of one contract, and a silent drift
    would land every window unverified;
  * epoch-aligned windows, so two watchers that started at different times
    cut the same traffic into the same windows (which is what makes the
    content-addressed id a real duplicate guard rather than a per-process
    one);
  * the runner never asserts a PASS/FAIL verdict — that is the server's
    (D5/D9). The runner sends counters, and only ever a status when it could
    not evaluate at all;
  * nothing leaves the machine but counters and hashes: no payload, no raw
    trace id.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from coriqo_agents.runner import monitoring as watch

EXECUTION_ID = "ex_watch_1"
BUNDLE_HASH = "b" * 64


def _bundle() -> dict:
    return {
        "id": "proto-1",
        "version": "1.0",
        "bundle_hash": BUNDLE_HASH,
        "template": {"protocol": {"validators": [{"id": "tool_authorization"}]}},
    }


@contextlib.contextmanager
def _watch_workspace():
    """A bundle, a validator config and an empty trace directory in a temp
    tree — the scaffolding every CLI-level test below needs. Yields
    `(trace_dir, argv_prefix)` so a test writes its trace files and runs the
    CLI without restating six lines of setup."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle_path = tmp_path / "bundle.json"
        bundle_path.write_text(json.dumps(_bundle()))
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"authorized_tools": ["search"]}))
        trace_dir = tmp_path / "otlp"
        trace_dir.mkdir()
        yield trace_dir, [
            "--execution", EXECUTION_ID, "--bundle", str(bundle_path),
            "--trace", str(trace_dir), "--window", "1h",
            "--config", str(config_path), "--dry-run",
        ]


def _capture(fn):
    """Run `fn` with stdout/stderr captured; returns (result, out, err).
    A KeyboardInterrupt is how the daemon tests end their loop, so it is
    swallowed rather than treated as a failure."""
    out, err = io.StringIO(), io.StringIO()
    result = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            result = fn()
        except KeyboardInterrupt:
            pass
    return result, out.getvalue(), err.getvalue()


def _otlp(trace_id: str, when: datetime, *, tool: str = "search", authorized: bool = True) -> dict:
    """One minimal OTLP/JSON export: a single tool call, authorized or not."""
    nanos = str(int(when.timestamp() * 1_000_000_000))
    # No explicit `coriqo.event_type`: the `gen_ai.tool.name` branch of
    # otel_ingest's mapping is the one that carries the tool NAME onto the
    # event, and the tool name is what `tool_authorization` checks. Setting
    # the event type by hand takes the earlier branch and leaves `tool` null.
    attrs = [
        {"key": "gen_ai.tool.name", "value": {"stringValue": tool}},
        {"key": "coriqo.authorization",
         "value": {"stringValue": "APPROVED" if authorized else "DENIED"}},
    ]
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                {"key": "coriqo.agent_id", "value": {"stringValue": "agent-1"}},
                {"key": "coriqo.agent_version", "value": {"stringValue": "1.0.0"}},
            ]},
            "scopeSpans": [{
                "spans": [{
                    "traceId": trace_id,
                    "spanId": "0" * 16,
                    "name": "tool",
                    "startTimeUnixNano": nanos,
                    "endTimeUnixNano": nanos,
                    "attributes": attrs,
                }],
            }],
        }],
    }


# The server's values for the two limits that cross the SDK/API boundary.
# Duplicated here as literals ON PURPOSE, exactly as the window-id digest is
# below: the SDK has no dependency on the API package, so a test that
# imported the server's constants could not detect the two drifting apart —
# it would just agree with whatever the server now says. If this test fails,
# one side moved and the other has to follow.
#
# Why they matter if they drift: a runner batching above the server's window
# cap gets a 400 for the whole request and can never drain an outage backlog;
# a runner building more digests than the server accepts gets the window
# refused outright.
SERVER_MAX_WINDOWS_PER_UPLOAD = 200
SERVER_MAX_DIGESTS_PER_RESULT = 20


def test_limits_that_cross_the_server_boundary_still_agree():
    assert watch.MAX_WINDOWS_PER_UPLOAD <= SERVER_MAX_WINDOWS_PER_UPLOAD, (
        "the runner batches more windows per request than the server accepts"
    )
    assert watch.MAX_DIGESTS_PER_RESULT <= SERVER_MAX_DIGESTS_PER_RESULT, (
        "the runner builds more failure digests than the server accepts"
    )
    print("  ✓ the SDK's upload limits are still within the server's")


def test_digests_are_capped_while_being_collected_not_afterwards():
    """Round-2 review: digests accumulated one dict per failed turn and were
    sliced only at output, so watcher memory scaled with the size of the
    incident it was reporting — most likely to be OOM-killed exactly when the
    report mattered. The cap has to bind during collection."""
    start = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    # Far more failing turns than the cap, all in one window.
    traces = []
    for i in range(SERVER_MAX_DIGESTS_PER_RESULT * 5):
        traces.extend(watch.otlp_json_bytes_to_traces(json.dumps(
            _otlp(f"{i:032x}", start + timedelta(seconds=i), tool="wire_transfer")).encode()))

    evaluation = watch.evaluate_window(
        bundle=_bundle(), traces=traces, window_start=start, window_seconds=3600,
        context_template=watch.ValidatorContext(
            execution_id=EXECUTION_ID, agent_model_id="m", agent_version_ref="v",
            environment="PRODUCTION", configuration={"authorized_tools": ["search"]}),
    )
    row = evaluation.results[0]
    # The COUNT carries the scale; the digests are a bounded sample.
    assert row["turns_failed"] == SERVER_MAX_DIGESTS_PER_RESULT * 5
    assert len(row["failure_digests"]) == watch.MAX_DIGESTS_PER_RESULT
    print("  ✓ digests stay bounded during a large incident; the count carries the scale")


def test_the_quiet_period_fallback_picks_the_newest_file_by_mtime():
    """Round-2 review: the fallback was `files[-1:]`, the lexically last
    name. Collectors name batches by uuid or `batchN.json`, neither of which
    sorts in time order, so the watcher could re-process a stale file every
    poll."""
    import os

    with tempfile.TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        # 'aaa' sorts last but is OLD; 'zzz' sorts first-from-the-end here
        # only by luck — the point is that name order != time order.
        stale = trace_dir / "zzz-stale.json"
        fresh = trace_dir / "aaa-fresh.json"
        stale_body = json.dumps(_otlp("3" * 32, datetime.now(timezone.utc)))
        fresh_body = json.dumps(_otlp("4" * 32, datetime.now(timezone.utc)))
        stale.write_text(stale_body)
        fresh.write_text(fresh_body)
        old = (datetime.now(timezone.utc) - timedelta(days=400)).timestamp()
        os.utime(stale, (old, old))

        # otel_ingest decodes the wire trace id, so compare against what the
        # fresh file itself parses to rather than the literal we wrote.
        expected = watch.otlp_json_bytes_to_traces(fresh_body.encode())[0]["trace_id"]

        # A horizon in the future excludes both, forcing the fallback.
        future = datetime.now(timezone.utc) + timedelta(days=1)
        traces = watch.load_traces(trace_dir, since=future)
        assert len(traces) == 1
        assert traces[0]["trace_id"] == expected, traces[0]["trace_id"]
        print("  ✓ the fallback keeps the newest file by mtime, not by filename")


def test_window_id_matches_the_servers_formula():
    """Both sides hash `execution|start|end|bundle_hash` with the same ISO-Z
    rendering. Asserted against a literal digest, not against an import of
    the server's function — the SDK has no dependency on the API package,
    and a test that called the server's own code could not detect the two
    drifting apart."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    expected = hashlib.sha256(
        "|".join([EXECUTION_ID, "2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z", BUNDLE_HASH])
        .encode("utf-8")
    ).hexdigest()
    got = watch.compute_window_id(
        execution_id=EXECUTION_ID, window_start=start, window_end=end, bundle_hash=BUNDLE_HASH)
    assert got == expected, f"{got} != {expected}"
    print("  ✓ window id matches the server's documented digest")


def test_windows_are_aligned_to_the_epoch_not_to_process_start():
    """Two watchers starting 20 minutes apart must file the same event in
    the same window, or the content-addressed duplicate guard degrades into
    a per-process one."""
    a = watch.window_start_for(datetime(2026, 9, 1, 13, 5, tzinfo=timezone.utc), window_seconds=3600)
    b = watch.window_start_for(datetime(2026, 9, 1, 13, 25, tzinfo=timezone.utc), window_seconds=3600)
    assert a == b == datetime(2026, 9, 1, 13, 0, tzinfo=timezone.utc)
    print("  ✓ windows sit on a fixed epoch-aligned grid")


def test_parse_window_refuses_a_per_turn_stream():
    for good, seconds in (("5m", 300), ("1h", 3600), ("1d", 86400), ("900s", 900)):
        assert watch.parse_window(good) == seconds
    for bad in ("1s", "0h", "2d", "hour", ""):
        try:
            watch.parse_window(bad)
        except watch.WatchUsageError:
            continue
        raise AssertionError(f"--window {bad!r} should have been refused")
    print("  ✓ sub-minute and over-a-day windows are refused")


def test_traces_bucket_by_their_first_event():
    base = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    traces = [
        {"events": [{"timestamp": base.isoformat().replace("+00:00", "Z")}]},
        {"events": [{"timestamp": (base + timedelta(minutes=90)).isoformat().replace("+00:00", "Z")}]},
        {"events": [{"timestamp": "not-a-timestamp"}]},   # undated: dropped, not guessed
        {"events": []},
    ]
    buckets = watch.bucket_traces(traces, window_seconds=3600)
    assert sorted(buckets) == [base, base + timedelta(hours=1)], sorted(buckets)
    assert sum(len(v) for v in buckets.values()) == 2
    print("  ✓ traces bucket by first event; undated traces are dropped, not filed under 'now'")


def test_the_runner_sends_counters_and_never_a_pass_fail_verdict():
    start = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    # One turn inside the roster, one outside it — `tool_authorization`
    # checks roster membership, so the off-roster tool is the failing turn.
    clean = watch.otlp_json_bytes_to_traces(
        json.dumps(_otlp("a" * 32, start, tool="search")).encode())
    dirty = watch.otlp_json_bytes_to_traces(
        json.dumps(_otlp("b" * 32, start + timedelta(minutes=1), tool="wire_transfer")).encode())

    evaluation = watch.evaluate_window(
        bundle=_bundle(), traces=clean + dirty, window_start=start, window_seconds=3600,
        context_template=watch.ValidatorContext(
            execution_id=EXECUTION_ID, agent_model_id="m", agent_version_ref="v",
            environment="PRODUCTION", configuration={"authorized_tools": ["search"]}),
    )
    assert evaluation.trace_count == 2
    rows = {r["validator_id"]: r for r in evaluation.results}
    row = rows["tool_authorization"]
    assert row["turns_evaluated"] == 2
    assert row["turns_failed"] == 1
    # D5/D9: the runner does NOT assert the verdict. It may only carry a
    # status when it could not evaluate at all.
    assert row.get("status") in (None, "ERROR", "INCONCLUSIVE"), row
    print("  ✓ the window row carries counters, and no PASS/FAIL the runner decided")


def test_nothing_but_counters_and_hashes_leaves_the_machine():
    start = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    traces = watch.otlp_json_bytes_to_traces(
        json.dumps(_otlp("c" * 32, start, tool="wire_transfer", authorized=False)).encode())
    evaluation = watch.evaluate_window(
        bundle=_bundle(), traces=traces, window_start=start, window_seconds=3600,
        context_template=watch.ValidatorContext(
            execution_id=EXECUTION_ID, agent_model_id="m", agent_version_ref="v",
            environment="PRODUCTION", configuration={"authorized_tools": ["search"]}),
    )
    payload = json.dumps(evaluation.to_payload(execution_id=EXECUTION_ID, bundle_hash=BUNDLE_HASH))
    # The raw trace id is a join key back into the bank's own observability
    # stack. It must be hashed, never shipped.
    assert "c" * 32 not in payload, payload
    digests = evaluation.results[0].get("failure_digests") or []
    assert digests, "a failing validator should carry a digest"
    assert all("trace_sha256" in d for d in digests)
    assert all("payload" not in d for d in digests)
    print("  ✓ digests carry a hashed trace pointer and scenario prose, never a payload")


def test_dry_run_uploads_nothing_and_skips_the_window_still_filling():
    with _watch_workspace() as (trace_dir, argv):
        now = datetime.now(timezone.utc)
        closed = now - timedelta(hours=3)
        (trace_dir / "closed.json").write_text(json.dumps(_otlp("d" * 32, closed)))
        # An event inside the window that has not finished yet: uploading it
        # would freeze a partial window under an id that can never be
        # revised.
        (trace_dir / "open.json").write_text(json.dumps(_otlp("e" * 32, now)))

        code, out, _err = _capture(lambda: watch.main(argv + ["--once"]))
        assert code == watch.EXIT_OK, code
        printed = json.loads(out)
        assert len(printed["windows"]) == 1, printed
        window = printed["windows"][0]
        assert window["window_start"] == watch._isoformat(
            watch.window_start_for(closed, window_seconds=3600))
        print("  ✓ --dry-run prints only closed windows and uploads nothing")


def test_a_transient_trace_read_failure_does_not_kill_the_daemon():
    """Round-2 review: a `WatchUsageError` from re-reading `--trace` (the
    collector directory momentarily empty during log rotation, a half-written
    file, a mount blip) returned out of the while-loop and ended the process.
    Monitoring then stopped silently — and silence from a monitor reads as
    "nothing wrong", which is worse than no monitor.

    Driven through `main` with a directory that is empty on the first poll
    and populated on the second."""
    with _watch_workspace() as (trace_dir, argv):  # trace dir starts EMPTY
        polls = {"n": 0}
        real_sleep = watch.time.sleep

        def fake_sleep(_seconds):
            polls["n"] += 1
            if polls["n"] == 1:
                # The collector finishes rotating and writes a file.
                closed = datetime.now(timezone.utc) - timedelta(hours=3)
                (trace_dir / "a.json").write_text(json.dumps(_otlp("f" * 32, closed)))
            else:
                raise KeyboardInterrupt  # end the daemon for the test

        watch.time.sleep = fake_sleep
        try:
            _, out, err = _capture(lambda: watch.main(argv + ["--poll-seconds", "1"]))
        finally:
            watch.time.sleep = real_sleep

        assert "no .json trace files found" in err, err
        assert "retrying in" in err
        # It survived the empty poll and went on to report the real window.
        assert polls["n"] >= 2, "the daemon exited on the transient failure"
        assert '"window_id"' in out, out[:400]
        print("  ✓ a transient trace-read failure retries instead of ending the watch")


def test_once_still_reports_a_trace_read_failure_as_an_exit_code():
    """The daemon's resilience must not swallow the signal a cron/pipeline
    caller depends on."""
    with _watch_workspace() as (_trace_dir, argv):  # trace dir starts EMPTY
        code, _out, err = _capture(lambda: watch.main(argv + ["--once"]))
        assert code == watch.EXIT_USAGE, code
        assert "retrying" not in err
        print("  ✓ --once still returns EXIT_USAGE rather than retrying forever")


def test_stale_trace_files_are_skipped_past_the_backfill_horizon():
    """Round-2 review: `load_traces` re-read and re-parsed the WHOLE retained
    trace directory on every poll, so a watcher meant to run for months grew
    its per-poll work without bound on the customer's own host."""
    with tempfile.TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        old = trace_dir / "old.json"
        new = trace_dir / "new.json"
        old.write_text(json.dumps(_otlp("1" * 32, datetime.now(timezone.utc))))
        new.write_text(json.dumps(_otlp("2" * 32, datetime.now(timezone.utc))))
        import os
        ancient = (datetime.now(timezone.utc) - timedelta(days=400)).timestamp()
        os.utime(old, (ancient, ancient))

        horizon = datetime.now(timezone.utc) - timedelta(days=30)
        assert len(watch.load_traces(trace_dir)) == 2                 # unbounded
        assert len(watch.load_traces(trace_dir, since=horizon)) == 1  # bounded

        # A horizon that would exclude EVERYTHING keeps the newest file
        # rather than raising "no trace files" at a quiet watcher.
        future = datetime.now(timezone.utc) + timedelta(days=1)
        assert len(watch.load_traces(trace_dir, since=future)) == 1
        print("  ✓ trace files older than the backfill horizon are skipped, never all of them")


def _run_all() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — a standalone runner
            failures += 1
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
    print("FAILED" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
