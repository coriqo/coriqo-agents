"""`python -m coriqo_agents.runner` — run an engine adapter against a target
and upload the signed manifest and results.

This is the hook a bank's CI pipeline calls. It is deliberately a command,
not a library entry point: the runner executes inside the customer's
environment, on their schedule, and Coriqo has no way to trigger it — which
is the property that lets a bank point it at production-adjacent systems at
all.

    python -m coriqo_agents.runner \\
        --execution ex_123 --engine promptfoo \\
        --bundle bundle.json --target target.json --scope scope.json \\
        --workdir ./run --base-url https://coriqo.example --tenant acme_bank \\
        --device-key ./device.key

`--dry-run` prepares the engine config and prints the plan without executing
or uploading anything, so an operator can read exactly what would be fired at
their agent before it is. That is not a convenience: nothing should reach a
customer's system because a config file was assumed to be right.

Exit codes: 0 the run completed and uploaded; 1 a usage or input error; 2 the
upload was rejected. A run whose ENGINE failed still exits 0 after uploading —
the disclosed ERROR results are the point, and a pipeline that treated them as
a crash would delete the evidence that the tool did not run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._common import DeviceKeyError, load_device_key
from .contract import ValidatorContext
from .engine import InvalidBundleError, RunnerEngine, UnknownValidatorError
from .engines import ADAPTERS, GarakAdapter, InspectAdapter, ManualAdapter, PromptfooAdapter, PyritAdapter
from .signing import DeviceSigner
from .uploader import ExecutionUploader

#: The CI gate is a run mode, not a third-party engine adapter — it
#: does not implement EngineAdapter's four-method contract (it runs native
#: validators directly, the same as `run()` without `--engine`), so it
#: dispatches here rather than joining `engines.ADAPTERS`.
_CI_SUBCOMMAND = "ci"

#: The continuous monitor, dispatched the same way and for the same
#: reason: `watch` runs native validators over production traces on a window
#: grid, so it is a run mode rather than an engine adapter.
_WATCH_SUBCOMMAND = "watch"

#: Re-performance of a signed result on a kept trace. Offline except for the
#: optional fetch from Coriqo; dispatched like the other run modes.
_VERIFY_SUBCOMMAND = "verify"

#: The benchmark harness and validator conformance suite. Dispatched the
#: same way and for the same reason as `ci`/`watch`: neither implements
#: EngineAdapter's contract, and neither uploads anything — `bench` scores a
#: corpus offline and `conformance` runs six checks over one validator class.
_BENCH_SUBCOMMAND = "bench"
_CONFORMANCE_SUBCOMMAND = "conformance"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coriqo_agents.runner",
        description="Run an assurance engine against a target and upload the signed results.",
    )
    parser.add_argument("--execution", required=True, help="Assurance execution id from Coriqo.")
    parser.add_argument(
        "--engine", required=True, choices=sorted(ADAPTERS),
        help="Which engine adapter to run. 'manual' reads a human-written manual_results.json.",
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Compiled protocol bundle JSON.")
    parser.add_argument("--target", type=Path, help="Target JSON: endpoint_ref, auth_env, headers.")
    parser.add_argument("--scope", type=Path, help="Scope JSON: out_of_scope entries.")
    parser.add_argument(
        "--workdir", type=Path, default=Path("./coriqo-run"),
        help="Where the engine config, generated tests and results are written.",
    )
    parser.add_argument("--agent-model-id", default="", help="Agent under test, for the result context.")
    parser.add_argument("--agent-version-ref", default=None, help="Version of the agent under test.")
    parser.add_argument("--environment", default=None, help="Where the target runs: STAGING, SANDBOX, PRODUCTION.")
    parser.add_argument("--base-url", help="Coriqo base URL. Required unless --dry-run.")
    parser.add_argument("--tenant", help="Tenant slug. Required unless --dry-run.")
    parser.add_argument(
        "--device-key", type=Path,
        help="PEM file holding this host's enrolled Ed25519 private key. Required unless --dry-run.",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds before the engine is killed.")
    parser.add_argument("--num-tests", type=int, default=5, help="Test cases per Promptfoo plugin.")
    parser.add_argument(
        "--max-turns", type=int, default=5,
        help="Turns per PyRIT conversation before the attack gives up on an objective.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write the engine config and print the plan. Runs nothing, uploads nothing.",
    )
    return parser


def _load_json(path: Path | None, what: str) -> dict:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{what} could not be read from {path}: {exc}")
    if not isinstance(value, dict):
        raise SystemExit(f"{what} in {path} must be a JSON object")
    return value


def _build_adapter(args: argparse.Namespace, target: dict, scope: dict):
    if args.engine == PromptfooAdapter.engine_id:
        return PromptfooAdapter(
            execution_id=args.execution, target=target, scope=scope, workdir=args.workdir,
            timeout_seconds=args.timeout, num_tests=args.num_tests,
        )
    if args.engine == PyritAdapter.engine_id:
        return PyritAdapter(
            execution_id=args.execution, target=target, scope=scope, workdir=args.workdir,
            max_turns=args.max_turns, wall_clock_seconds=args.timeout,
        )
    if args.engine == ManualAdapter.engine_id:
        return ManualAdapter(workdir=args.workdir)
    if args.engine == GarakAdapter.engine_id:
        return GarakAdapter(
            execution_id=args.execution, target=target, scope=scope, workdir=args.workdir,
            timeout_seconds=args.timeout,
        )
    if args.engine == InspectAdapter.engine_id:
        return InspectAdapter(
            execution_id=args.execution, target=target, scope=scope, workdir=args.workdir,
            timeout_seconds=args.timeout,
        )
    raise SystemExit(f"no adapter for engine {args.engine!r}")


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv[:1] == [_CI_SUBCOMMAND]:
        from .ci import main as ci_main
        return ci_main(raw_argv[1:])
    if raw_argv[:1] == [_WATCH_SUBCOMMAND]:
        from .monitoring import main as watch_main
        return watch_main(raw_argv[1:])
    if raw_argv[:1] == [_VERIFY_SUBCOMMAND]:
        from .verify import main as verify_main
        return verify_main(raw_argv[1:])
    if raw_argv[:1] == [_BENCH_SUBCOMMAND]:
        from .bench import main as bench_main
        return bench_main(raw_argv[1:])
    if raw_argv[:1] == [_CONFORMANCE_SUBCOMMAND]:
        from ..openformat.conformance_cli import main as conformance_main
        return conformance_main(raw_argv[1:])
    # `coriqo-agents runner run ...` (the cli.py console-script spelling)
    # names the subcommand explicitly; `python -m coriqo_agents.runner ...`
    # (this module's own long-standing spelling) omits it. Both parse the
    # same flags either way.
    if raw_argv[:1] == ["run"]:
        raw_argv = raw_argv[1:]

    args = build_parser().parse_args(raw_argv)
    bundle = _load_json(args.bundle, "the protocol bundle")
    target = _load_json(args.target, "the target")
    scope = _load_json(args.scope, "the scope")

    if not args.dry_run:
        missing = [
            name for name, value in
            (("--base-url", args.base_url), ("--tenant", args.tenant), ("--device-key", args.device_key))
            if not value
        ]
        if missing:
            raise SystemExit(f"{', '.join(missing)} are required unless --dry-run is passed")

    adapter = _build_adapter(args, target, scope)

    if args.dry_run:
        plan = adapter.prepare(bundle)
        print(json.dumps(plan.disclosure(), indent=2, sort_keys=True))
        if plan.config_path:
            print(f"\nengine config written to {plan.config_path}", file=sys.stderr)
        return 0

    try:
        engine = RunnerEngine(bundle)
    except InvalidBundleError as exc:
        raise SystemExit(str(exc))
    context = ValidatorContext(
        execution_id=args.execution,
        agent_model_id=args.agent_model_id,
        agent_version_ref=args.agent_version_ref,
        environment=args.environment,
    )
    try:
        results, manifest, evidence, failure_scenarios = engine.run_engine(adapter, context)
    except UnknownValidatorError as exc:
        # engine.py's own docstring: this can fire AFTER adapter.run(plan)
        # has already sent real traffic at the customer's target (the
        # post-normalize check is a second guard, not the only one — the
        # first runs before adapter.run()). A run against the target did
        # happen even though nothing gets uploaded here; disclose that
        # rather than crashing with a raw traceback and no record at all.
        print(
            f"a run against the target may have already happened, but its results "
            f"cannot be uploaded: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        private_key = load_device_key(args.device_key)
    except DeviceKeyError as exc:
        raise SystemExit(str(exc))
    signer = DeviceSigner(private_key=private_key)
    uploader = ExecutionUploader(base_url=args.base_url, signer=signer, tenant_slug=args.tenant)
    try:
        uploader.upload_manifest(args.execution, manifest)
        response = uploader.upload_results(
            args.execution, [r.to_dict() for r in results], evidence, failure_scenarios,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
