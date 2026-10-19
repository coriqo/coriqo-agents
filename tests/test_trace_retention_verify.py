"""Trace provenance, `runner verify` and `--retain-trace`.

Runs standalone, same convention as its siblings in this directory:

    PYTHONPATH="$PWD:$PWD/sdk/coriqo-agents" python sdk/coriqo-agents/tests/test_trace_retention_verify.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
from pathlib import Path

from coriqo_agents.runner import ci as ci_module
from coriqo_agents.runner import verify as verify_module
from coriqo_agents.runner.contract import ValidatorContext
from coriqo_agents.runner.engine import RunnerEngine, _trace_provenance
from coriqo_agents.runner.manifest import hash_value
from coriqo_agents.runner.validators import resolve_validators

from _ci_test_helpers import HAVE_CRYPTOGRAPHY, bundle, run_ci, run_tests

TOOL_SECRET = "tool-name-that-must-not-leave-the-host"


def _event(i: int, ts: str | None, tool: str = "loan_system") -> dict:
    event = {
        "event_id": f"evt_{i}", "trace_id": "t1", "agent_id": "a1", "event_type": "TOOL_CALL",
        "tool": tool, "action": "update", "actor": "agent", "authorization": "APPROVED",
        "payload": {"note": "x"},
    }
    if ts is not None:
        event["timestamp"] = ts
    return event


def _trace() -> dict:
    return {
        "trace_id": "t1", "agent_id": "a1", "agent_version": "1", "source": "otlp_json",
        "events": [
            _event(1, "2026-10-01T10:00:00+00:00"),
            _event(2, "2026-10-01T10:00:05+00:00"),
            _event(3, "2026-10-01T10:02:05Z", tool=TOOL_SECRET),
        ],
    }


def _honest_run(trace: dict | None = None, with_evidence: bool = False):
    """(bundle, manifest, results, trace) from a real engine run, the way ci does."""
    b = bundle()
    trace = trace or _trace()
    engine = RunnerEngine(b)
    context = ValidatorContext(
        execution_id="ex_1", agent_model_id="m1", agent_version_ref="v1", environment=None,
        inputs={"trace": trace},
    )
    results, manifest, evidence, _scenarios = engine.run(resolve_validators(b), context)
    if with_evidence:
        return b, manifest, [r.to_dict() for r in results], trace, evidence
    return b, manifest, [r.to_dict() for r in results], trace


def _write(d: Path, name: str, value) -> Path:
    path = d / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _run_verify(argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = verify_module.main(argv)
    return code, out.getvalue() + err.getvalue()


def _local_args(d: Path, b, manifest, results, trace, evidence=None) -> list[str]:
    return [
        "--bundle", str(_write(d, "bundle.json", b)), "--manifest", str(_write(d, "manifest.json", manifest)),
        "--results", str(_write(d, "results.json", {"results": results, "evidence": evidence or {}})),
        "--trace", str(_write(d, "trace.json", trace)),
    ]


# -- provenance ---------------------------------------------------------------

def test_provenance_adds_timing_signals_and_no_content():
    provenance = _trace_provenance(ValidatorContext("e", "m", None, None, inputs={"trace": _trace()}))
    assert provenance["source"] == "otlp_json" and provenance["event_count"] == 3
    assert provenance["first_event_at"] == "2026-10-01T10:00:00+00:00"
    assert provenance["last_event_at"] == "2026-10-01T10:02:05+00:00"
    assert provenance["max_gap_seconds"] == 120.0
    assert TOOL_SECRET not in json.dumps(provenance) and "loan_system" not in json.dumps(provenance)
    print("  ok provenance carries first/last event time and the longest silence, no content")


def test_provenance_without_timestamps_keeps_the_old_shape():
    trace = _trace()
    for e in trace["events"]:
        e.pop("timestamp", None)
    provenance = _trace_provenance(ValidatorContext("e", "m", None, None, inputs={"trace": trace}))
    assert provenance == {"source": "otlp_json", "event_count": 3}, provenance
    print("  ok a trace with no event times yields exactly the old two keys")


def test_one_event_gap_is_zero():
    trace = _trace()
    trace["events"] = trace["events"][:1]
    provenance = _trace_provenance(ValidatorContext("e", "m", None, None, inputs={"trace": trace}))
    assert provenance["max_gap_seconds"] == 0.0
    print("  ok a single event has a zero gap")


# -- verify -------------------------------------------------------------------

def test_verify_reproduces_an_honest_trace():
    b, manifest, results, trace, evidence = _honest_run(with_evidence=True)
    with tempfile.TemporaryDirectory() as tmp:
        # The execution id comes from the evidence block of the kept upload body.
        code, output = _run_verify(_local_args(Path(tmp), b, manifest, results, trace, evidence))
        assert code == verify_module.EXIT_REPRODUCED, output
        # Or from --execution, when only the results are kept.
        code2, output2 = _run_verify(_local_args(Path(tmp), b, manifest, results, trace) + ["--execution", "ex_1"])
        assert code2 == verify_module.EXIT_REPRODUCED, output2
        # With neither, it says what is missing instead of failing blind.
        code3, output3 = _run_verify(_local_args(Path(tmp), b, manifest, results, trace))
        assert code3 == verify_module.EXIT_NOT_REPRODUCED and "--execution" in output3, output3
    assert "REPRODUCED" in output and "DID NOT" not in output and "tool_authorization" in output
    print("  ok an honest trace reproduces, exit 0")


def test_verify_fails_when_the_trace_is_altered_by_one_byte():
    b, manifest, results, trace = _honest_run()
    altered = copy.deepcopy(trace)
    altered["events"][0]["payload"]["note"] = "y"
    with tempfile.TemporaryDirectory() as tmp:
        code, output = _run_verify(_local_args(Path(tmp), b, manifest, results, altered))
    assert code == verify_module.EXIT_NOT_REPRODUCED, output
    assert "DID NOT REPRODUCE" in output and "does not hash" in output
    print("  ok a one-byte trace change does not reproduce, exit 1")


def test_verify_refuses_a_trace_file_that_repeats_a_key_or_holds_nan():
    b, manifest, results, trace = _honest_run()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        args = _local_args(d, b, manifest, results, trace)
        trace_path = d / "trace.json"
        text = json.dumps(trace)
        trace_path.write_text(text.replace('"events":', '"events": [], "events":', 1), encoding="utf-8")
        code, output = _run_verify(args)
        assert code == verify_module.EXIT_USAGE_ERROR and "repeats the key" in output, output
        trace_path.write_text(text[:-1] + ', "score": NaN}', encoding="utf-8")
        code, output = _run_verify(args)
        assert code == verify_module.EXIT_USAGE_ERROR and "NaN" in output, output
    print("  ok a repeated key or NaN in an input file is refused before anything is compared")


def test_verify_fails_when_a_filed_result_is_altered():
    b, manifest, results, trace = _honest_run()
    altered = copy.deepcopy(results)
    altered[0]["observations"] = [altered[0]["observations"][0] + "."]
    with tempfile.TemporaryDirectory() as tmp:
        code, output = _run_verify(_local_args(Path(tmp), b, manifest, altered, trace))
    assert code == verify_module.EXIT_NOT_REPRODUCED, output
    assert "DIFFERS" in output
    print("  ok a one-byte change in a filed result does not reproduce, exit 1")


def test_verify_fails_when_a_signed_result_hash_is_altered():
    b, manifest, results, trace = _honest_run()
    forged = copy.deepcopy(manifest)
    forged["result_hashes"][0] = "0" * 64
    with tempfile.TemporaryDirectory() as tmp:
        code, _ = _run_verify(_local_args(Path(tmp), b, forged, results, trace))
    assert code == verify_module.EXIT_NOT_REPRODUCED
    print("  ok a changed signed hash does not reproduce")


def test_verify_usage_errors_exit_two():
    code, output = _run_verify([])
    assert code == verify_module.EXIT_USAGE_ERROR, output
    with tempfile.TemporaryDirectory() as tmp:
        b = _write(Path(tmp), "bundle.json", bundle())
        code, output = _run_verify(["--bundle", str(b), "--manifest", str(Path(tmp) / "m.json")])
    assert code == verify_module.EXIT_USAGE_ERROR and "--results" in output, output
    print("  ok missing inputs exit 2 and name the missing flags")


def test_verify_fetch_failure_exits_three_and_never_prints_the_key():
    secret = "sk-verify-secret-91ac"
    real_get = verify_module.httpx.get

    def refuse(url, headers=None, timeout=None):
        raise verify_module.httpx.ConnectError("connection refused")

    verify_module.httpx.get = refuse
    try:
        with tempfile.TemporaryDirectory() as tmp:
            b = _write(Path(tmp), "bundle.json", bundle())
            code, output = _run_verify([
                "--bundle", str(b), "--execution", "ex_1", "--base-url", "https://coriqo.example",
                "--tenant", "acme_bank", "--api-key", secret,
            ])
    finally:
        verify_module.httpx.get = real_get
    assert code == verify_module.EXIT_FETCH_FAILED, output
    assert secret not in output
    print("  ok an unreachable Coriqo exits 3 and the API key is not printed")


def test_verify_fetches_manifest_results_and_trace_from_coriqo():
    b, manifest, results, trace = _honest_run()
    detail = {"runner_manifest": manifest, "validator_results": [
        {**r, "validator_version": r["version"]} for r in results]}
    for r in detail["validator_results"]:
        r.pop("version")
    seen: list[str] = []

    class _Resp:
        def __init__(self, status_code, body):
            self.status_code, self._body = status_code, body
            self.content = json.dumps(body).encode()
            self.text = json.dumps(body)

        def json(self):
            return self._body

    def fake_get(url, headers=None, timeout=None):
        seen.append(url)
        assert headers["X-API-Key"] == "sk-x"
        return _Resp(200, trace if url.endswith("/trace") else detail)

    real_get = verify_module.httpx.get
    verify_module.httpx.get = fake_get
    try:
        with tempfile.TemporaryDirectory() as tmp:
            bp = _write(Path(tmp), "bundle.json", b)
            code, output = _run_verify([
                "--bundle", str(bp), "--execution", "ex_1", "--base-url", "https://coriqo.example",
                "--tenant", "acme_bank", "--api-key", "sk-x",
            ])
    finally:
        verify_module.httpx.get = real_get
    assert code == verify_module.EXIT_REPRODUCED, output
    assert "fetched from Coriqo's record" in output and "local files only" not in output, output
    assert any(u.endswith("/assurance/ex_1/trace") for u in seen), seen
    print("  ok --execution fetches the signed manifest, results and kept trace and reproduces")


def test_local_mode_says_it_only_shows_the_files_agree_and_prints_the_manifest_hash():
    b, manifest, results, trace, evidence = _honest_run(with_evidence=True)
    with tempfile.TemporaryDirectory() as tmp:
        code, output = _run_verify(_local_args(Path(tmp), b, manifest, results, trace, evidence))
    assert code == verify_module.EXIT_REPRODUCED, output
    assert "REPRODUCED (local files only" in output and "does not show that Coriqo signed" in output, output
    assert "NOT checked against Coriqo" in output and hash_value(manifest)[:16] in output, output
    outcome = verify_module.verify(bundle=b, manifest=manifest, results=results, trace=trace,
                                   execution_id="ex_1", manifest_source="local")
    assert outcome["manifest_checked_against_coriqo"] is False and outcome["manifest_sha256"] == hash_value(manifest)
    assert "local files" in verify_module.__doc__ and "local files" in " ".join(
        verify_module.build_parser().epilog.split())
    print("  ok a local-mode REPRODUCED line says the manifest was not checked against Coriqo")


def test_wrong_bundle_is_its_own_problem_not_a_validator_difference():
    b, manifest, results, trace = _honest_run()
    other = copy.deepcopy(b)
    other["bundle_hash"] = "0" * 64
    outcome = verify_module.verify(bundle=other, manifest=manifest, results=results, trace=trace, execution_id="ex_1")
    assert outcome["reproduced"] is False and outcome["bundle_matches_manifest"] is False
    assert any(p.startswith("this is not the bundle that run used") for p in outcome["problems"]), outcome["problems"]
    # A bundle that matches the manifest raises no such problem.
    ok = verify_module.verify(bundle=b, manifest=manifest, results=results, trace=trace, execution_id="ex_1")
    assert ok["bundle_matches_manifest"] is True and ok["reproduced"] is True
    with tempfile.TemporaryDirectory() as tmp:
        args = _local_args(Path(tmp), other, manifest, results, trace) + ["--execution", "ex_1"]
        code, output = _run_verify(args)
    assert code == verify_module.EXIT_NOT_REPRODUCED and "this is not the bundle that run used" in output, output
    print("  ok a bundle that is not the one the manifest names is reported as its own problem")


# -- --retain-trace -----------------------------------------------------------

def test_retain_trace_with_retention_off_prints_one_line_and_does_not_fail_the_gate():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped - cryptography not installed)")
        return
    plain_code, _, _ = run_ci(ci_module)
    code, output, calls = run_ci(ci_module, extra_args=["--retain-trace"])
    assert code == plain_code, (code, plain_code, output)
    assert ("DEVICE_GET", "/api/v1/executions/ex_ci_1/trace-policy") in calls, calls
    assert not any(kind == "UPLOAD" and path.endswith("/trace") for kind, path in calls), calls
    assert "trace not kept: trace retention is off" in output and "Settings" in output
    print("  ok retention off: one plain line, no trace upload, same exit code")


def test_retain_trace_uploads_when_retention_is_on():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped - cryptography not installed)")
        return
    code, output, calls = run_ci(
        ci_module, extra_args=["--retain-trace"], trace_policy={"enabled": True, "days": 400, "max_mb": 25},
    )
    assert any(kind == "UPLOAD" and path.endswith("/executions/ex_ci_1/trace") for kind, path in calls), calls
    assert "trace not kept" not in output
    print("  ok retention on: the trace is uploaded after the results")


def test_without_the_flag_the_trace_is_never_touched():
    if not HAVE_CRYPTOGRAPHY:
        print("  (skipped - cryptography not installed)")
        return
    _, _, calls = run_ci(ci_module)
    assert not any(kind == "DEVICE_GET" for kind, _ in calls)
    assert not any(kind == "UPLOAD" and path.endswith("/trace") for kind, path in calls)
    print("  ok no flag, no policy call and no trace upload")


def test_hash_matches_the_server_contract():
    # Coriqo's API recomputes this hash server-side.
    trace = _trace()
    _, manifest, _, _ = _honest_run(trace)
    assert hash_value(trace) in manifest["input_dataset_hashes"]
    print("  ok the manifest commits to hash_value(trace)")


if __name__ == "__main__":
    raise SystemExit(run_tests(dict(globals())))
