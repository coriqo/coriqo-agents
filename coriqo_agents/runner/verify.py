"""`coriqo-agents runner verify` - re-perform a signed assurance result.

Takes a signed manifest, the results it covers, the trace the validators
judged and the protocol bundle, re-runs the bundle's validators on that trace
on this machine, and checks that what comes out is what was signed:

  1. the trace hashes to one of the manifest's `input_dataset_hashes`
     (so it is the trace that was judged, not a lookalike);
  2. each result as filed hashes to the manifest's `result_hashes` entry
     (so the filed result is the signed one);
  3. each re-run result hashes to the same entry (so running the same
     deterministic validator on the same trace gives the same result).

  4. the protocol bundle you supplied is the bundle the manifest names
     (its `bundle_hash`), reported as its own problem when it is not.

Verdict: REPRODUCED only if every check on every validator holds.

What a local-file run does and does not prove
---------------------------------------------
Fetched from Coriqo, the manifest and results are the copies Coriqo holds for
that run. With local files (--manifest, --results, --trace), the result proves
only that the files you supplied agree with each other: a manifest is signed by
no one but the request that carried it, so anyone can build a consistent set
by running the validators on any trace. It does NOT prove that Coriqo signed or
holds that manifest. Compare the manifest hash that verify prints with the
manifest on the run's record in Coriqo, or fetch with --execution.

No LLM, and no network beyond Coriqo: the validators are deterministic code
that ships in this package. The API key is used only in a request header and
never printed.

Two ways to supply the inputs:

    # fetch the signed manifest, results and kept trace from Coriqo
    coriqo-agents runner verify --execution ex_123 --bundle bundle.json \\
        --base-url https://coriqo.example --tenant acme_bank --api-key "$CORIQO_API_KEY"

    # a client's own copies
    coriqo-agents runner verify --bundle bundle.json --manifest manifest.json \\
        --results results.json --trace trace.json [--execution ex_123]

Exit codes: 0 reproduced, 1 did not reproduce, 2 usage error, 3 could not
fetch from Coriqo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from .contract import ValidatorContext
from .engine import InvalidBundleError, RunnerEngine
from .manifest import canonical_result_hash, hash_value
from .validators import resolve_validators

EXIT_REPRODUCED = 0
EXIT_NOT_REPRODUCED = 1
EXIT_USAGE_ERROR = 2
EXIT_FETCH_FAILED = 3


class VerifyUsageError(Exception):
    pass


class VerifyFetchError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coriqo-agents runner verify",
        description="Re-run a signed assurance result on its kept trace and check that it reproduces.",
        epilog=(
            "Exit codes:\n"
            "  0  REPRODUCED: the trace, every filed result and every re-run result match the manifest.\n"
            "     With local files this shows only that the files agree with each other. It does not\n"
            "     show that Coriqo signed that manifest: compare the printed manifest hash with the\n"
            "     run's record in Coriqo, or fetch with --execution.\n"
            "  1  DID NOT REPRODUCE: at least one check failed (see the table).\n"
            "  2  usage error: missing or unreadable input.\n"
            "  3  could not fetch the manifest, results or trace from Coriqo.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bundle", required=True, type=Path, help="The compiled protocol bundle JSON the run used. Checked against the "
                        "manifest's bundle_hash.")
    parser.add_argument("--execution", default=None,
                        help="Execution id. With --base-url it names the run to fetch; with local files it "
                             "is optional and only seeds the evidence ids (taken from the results when absent).")
    parser.add_argument("--base-url", default=None, help="Coriqo base URL (fetch mode).")
    parser.add_argument("--tenant", default=None, help="Tenant slug (fetch mode).")
    parser.add_argument("--api-key", default=None,
                        help="Service account X-API-Key with audit:export (fetch mode). Read it from an "
                             "environment variable, never type it into a shared shell history.")
    parser.add_argument("--manifest", type=Path, default=None, help="Signed manifest JSON (local mode).")
    parser.add_argument("--results", type=Path, default=None,
                        help="Results JSON: the upload body {\"results\": [...]}, or a bare list (local mode).")
    parser.add_argument("--trace", type=Path, default=None, help="The trace JSON (local mode).")
    parser.add_argument("--configuration", type=Path, default=None,
                        help="The configuration snapshot JSON the run used, when it had one (for example the "
                             "authorized_tools roster). Checked against the manifest's configuration_snapshot_hash.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the outcome as JSON.")
    return parser


# -- inputs ------------------------------------------------------------------

def _no_repeated_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(
                f"it repeats the key {key!r} in one object, so different JSON tools would read different "
                "values from the same file"
            )
        out[key] = value
    return out


def _no_nonstandard_number(token):
    raise ValueError(f"it contains {token}, which is not valid JSON and is read differently by different tools")


def _strict_loads(text):
    """json.loads that refuses what readers disagree on: a repeated key in an
    object, and NaN / Infinity. A file that hashes the same but reads two ways
    is not evidence of what was judged."""
    return json.loads(text, object_pairs_hook=_no_repeated_keys, parse_constant=_no_nonstandard_number)


def _read_json(path: Path, what: str) -> Any:
    try:
        return _strict_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerifyUsageError(f"{what} could not be read from {path}: {exc}") from exc


def _normalise_results(raw: Any) -> list[dict]:
    """Accept the upload body, a bare list, or Coriqo's execution detail
    (`validator_results`, which names the version `validator_version`)."""
    if isinstance(raw, dict):
        raw = raw.get("results", raw.get("validator_results"))
    if not isinstance(raw, list) or not all(isinstance(r, dict) for r in raw):
        raise VerifyUsageError("the results must be a JSON list of result objects (or {\"results\": [...]})")
    out = []
    for r in raw:
        r = dict(r)
        if "version" not in r and "validator_version" in r:
            r["version"] = r["validator_version"]
        out.append(r)
    return out


def _fetch(base_url: str, tenant: str, api_key: str, path: str) -> httpx.Response:
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}{path}", headers={"X-API-Key": api_key, "X-Tenant-Slug": tenant}, timeout=60,
        )
    except httpx.HTTPError as exc:
        raise VerifyFetchError(f"could not reach Coriqo at {base_url}{path}: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise VerifyFetchError(f"Coriqo answered {path} with {resp.status_code}: {detail}")
    return resp


def _load_remote(args: argparse.Namespace) -> tuple[dict, list[dict], dict]:
    detail = _fetch(args.base_url, args.tenant, args.api_key, f"/api/v1/assurance/{args.execution}").json()
    manifest = detail.get("runner_manifest")
    if not isinstance(manifest, dict):
        raise VerifyFetchError(f"execution {args.execution} has no signed manifest on file")
    results = _normalise_results(detail.get("validator_results"))
    trace_resp = _fetch(args.base_url, args.tenant, args.api_key, f"/api/v1/assurance/{args.execution}/trace")
    try:
        trace = _strict_loads(trace_resp.content.decode("utf-8"))
    except ValueError as exc:
        raise VerifyFetchError(f"the kept trace for {args.execution} is not valid JSON: {exc}") from exc
    return manifest, results, trace


def _load_local(args: argparse.Namespace) -> tuple[dict, list[dict], dict]:
    manifest = _read_json(args.manifest, "the manifest")
    if not isinstance(manifest, dict):
        raise VerifyUsageError(f"the manifest in {args.manifest} must be a JSON object")
    results = _normalise_results(_read_json(args.results, "the results"))
    trace = _read_json(args.trace, "the trace")
    if not isinstance(trace, dict):
        raise VerifyUsageError(f"the trace in {args.trace} must be a JSON object")
    return manifest, results, trace


def _execution_id_from(results_body_hint: list[dict], explicit: str | None, evidence: dict | None) -> str:
    if explicit:
        return explicit
    for envelope in (evidence or {}).values():
        if isinstance(envelope, dict) and envelope.get("execution_id"):
            return str(envelope["execution_id"])
    return ""


# -- the check ---------------------------------------------------------------

def verify(
    *, bundle: dict, manifest: dict, results: list[dict], trace: dict, execution_id: str,
    configuration: dict | None = None, manifest_source: str = "coriqo",
) -> dict:
    """Run the checks. Returns {"reproduced": bool, "manifest_source",
    "manifest_sha256", "manifest_checked_against_coriqo", "bundle_matches_manifest",
    "trace": {...}, "validators": [{validator_id, status, filed_matches,
    rerun_matches, note}], "problems": [...]}. Pure: no I/O.

    `manifest_source` is "coriqo" when the manifest was fetched from Coriqo's
    record and "local" when it came from files the caller supplied. Local means
    the verdict only shows the files agree with each other."""
    problems: list[str] = []
    bundle_hash = bundle.get("bundle_hash")
    signed_bundle_hash = manifest.get("bundle_hash")
    bundle_ok = bundle_hash is not None and bundle_hash == signed_bundle_hash
    if not bundle_ok:
        problems.append(
            "this is not the bundle that run used: the bundle you supplied has hash "
            f"{bundle_hash or '(none)'} and the manifest names {signed_bundle_hash or '(none)'}. Get the bundle "
            "for that protocol version; any 're-run differs' below may only be this"
        )
    trace_digest = hash_value(trace)
    committed = manifest.get("input_dataset_hashes") or []
    trace_ok = trace_digest in committed
    if not trace_ok:
        problems.append(
            "the trace does not hash to any input_dataset_hashes entry in the signed manifest, so it is not "
            "the trace that was judged"
        )

    configuration = configuration if configuration is not None else {}
    signed_config_hash = manifest.get("configuration_snapshot_hash")
    if signed_config_hash is not None and hash_value(configuration) != signed_config_hash:
        problems.append(
            "the run used a configuration snapshot that was not supplied (or differs); pass it with "
            "--configuration so the validators see the same settings"
        )

    if not execution_id and any(r.get("evidence_ids") for r in results):
        problems.append(
            "the results cite evidence ids, which are derived from the execution id; pass --execution "
            "(or the results file with its evidence block) so they can be re-derived"
        )

    declared = manifest.get("validators") or []
    signed_hashes = manifest.get("result_hashes") or []
    if len(declared) != len(signed_hashes):
        problems.append(
            f"the manifest lists {len(declared)} validator(s) but {len(signed_hashes)} result hash(es)"
        )
    expected = {v.get("id"): signed_hashes[i] for i, v in enumerate(declared) if i < len(signed_hashes)}
    order = [v.get("id") for v in declared]

    filed = {r.get("validator_id"): r for r in results}
    stray = sorted(set(filed) - set(order))
    if stray:
        problems.append(f"results were filed for validator(s) the manifest does not list: {stray}")

    rerun: dict[str, dict] = {}
    try:
        engine = RunnerEngine(bundle)
        validators = [v for v in resolve_validators(bundle) if v.validator_id in expected]
        missing = [vid for vid in order if vid not in {v.validator_id for v in validators}]
        if missing:
            problems.append(f"the bundle does not carry validator(s) {missing} that the manifest lists")
        validators.sort(key=lambda v: order.index(v.validator_id))
        context = ValidatorContext(
            execution_id=execution_id, agent_model_id="",
            agent_version_ref=manifest.get("system_version_ref"), environment=None,
            configuration=configuration, inputs={"trace": trace},
        )
        rerun_results, _manifest, _evidence, _scen = engine.run(validators, context)
        rerun = {r.validator_id: r.to_dict() for r in rerun_results}
    except (InvalidBundleError, KeyError) as exc:
        problems.append(f"the validators could not be re-run: {exc}")

    rows = []
    for vid in order:
        want = expected.get(vid)
        filed_result = filed.get(vid)
        filed_ok = filed_result is not None and want is not None and canonical_result_hash(filed_result) == want
        rerun_result = rerun.get(vid)
        rerun_ok = rerun_result is not None and want is not None and canonical_result_hash(rerun_result) == want
        note = ""
        if filed_result is None:
            note = "no filed result"
        elif not filed_ok:
            note = "filed result differs from the signed hash"
        elif rerun_result is None:
            note = "not re-run"
        elif not rerun_ok:
            note = "re-run differs from the signed hash"
        rows.append({
            "validator_id": vid,
            "status": (filed_result or {}).get("status"),
            "rerun_status": (rerun_result or {}).get("status"),
            "filed_matches": filed_ok,
            "rerun_matches": rerun_ok,
            "note": note,
        })

    reproduced = trace_ok and not problems and bool(rows) and all(r["filed_matches"] and r["rerun_matches"] for r in rows)
    return {
        "reproduced": reproduced,
        "manifest_source": manifest_source,
        "manifest_sha256": hash_value(manifest),
        "manifest_checked_against_coriqo": manifest_source == "coriqo",
        "bundle_matches_manifest": bundle_ok,
        "trace": {"sha256": trace_digest, "matches_manifest": trace_ok},
        "validators": rows,
        "problems": problems,
    }


def _print_table(outcome: dict) -> None:
    if outcome["manifest_checked_against_coriqo"]:
        print("Manifest       fetched from Coriqo's record for this run")
    else:
        print(f"Manifest       from files you supplied, hash {outcome['manifest_sha256'][:16]}...  "
              "NOT checked against Coriqo")
    print(f"Bundle         {'is the bundle the manifest names' if outcome['bundle_matches_manifest'] else 'is NOT the bundle the manifest names'}")
    print(f"Trace hash     {outcome['trace']['sha256'][:16]}...  "
          f"{'matches the manifest' if outcome['trace']['matches_manifest'] else 'DOES NOT match the manifest'}")
    print(f"{'Validator':<34} {'Filed':<8} {'Signed hash':<13} {'Re-run':<8} {'Re-run hash'}")
    for r in outcome["validators"]:
        print(
            f"{str(r['validator_id']):<34} {str(r['status'] or '-'):<8} "
            f"{'match' if r['filed_matches'] else 'DIFFERS':<13} "
            f"{str(r['rerun_status'] or '-'):<8} {'match' if r['rerun_matches'] else 'DIFFERS'}"
            + (f"   ({r['note']})" if r["note"] else "")
        )
    for problem in outcome["problems"]:
        print(f"Problem: {problem}")
    if not outcome["reproduced"]:
        print("DID NOT REPRODUCE")
    elif outcome["manifest_checked_against_coriqo"]:
        print("REPRODUCED")
    else:
        print("REPRODUCED (local files only: they agree with each other. This does not show that Coriqo "
              "signed or holds this manifest; compare the manifest hash with the run's record in Coriqo.)")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE_ERROR if exc.code not in (0, None) else 0

    remote = args.base_url or args.tenant or args.api_key
    local = args.manifest or args.results or args.trace
    try:
        if remote and local:
            raise VerifyUsageError("use either --base-url/--tenant/--api-key/--execution or "
                                   "--manifest/--results/--trace, not both")
        if remote:
            missing = [n for n, v in (("--base-url", args.base_url), ("--tenant", args.tenant),
                                      ("--api-key", args.api_key), ("--execution", args.execution)) if not v]
            if missing:
                raise VerifyUsageError(f"fetching from Coriqo needs {', '.join(missing)}")
        elif local:
            missing = [n for n, v in (("--manifest", args.manifest), ("--results", args.results),
                                      ("--trace", args.trace)) if not v]
            if missing:
                raise VerifyUsageError(f"local files need {', '.join(missing)}")
        else:
            raise VerifyUsageError("give --execution with --base-url/--tenant/--api-key, or --manifest, "
                                   "--results and --trace")
        configuration = _read_json(args.configuration, "the configuration") if args.configuration else None
        if configuration is not None and not isinstance(configuration, dict):
            raise VerifyUsageError(f"the configuration in {args.configuration} must be a JSON object")
        bundle = _read_json(args.bundle, "the protocol bundle")
        if not isinstance(bundle, dict):
            raise VerifyUsageError(f"the protocol bundle in {args.bundle} must be a JSON object")
        if remote:
            manifest, results, trace = _load_remote(args)
            execution_id = args.execution
        else:
            manifest, results, trace = _load_local(args)
            body = _read_json(args.results, "the results")
            evidence = body.get("evidence") if isinstance(body, dict) else None
            execution_id = _execution_id_from(results, args.execution, evidence)
    except VerifyUsageError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR
    except VerifyFetchError as exc:
        print(f"could not fetch what verify needs: {exc}", file=sys.stderr)
        return EXIT_FETCH_FAILED

    outcome = verify(bundle=bundle, manifest=manifest, results=results, trace=trace,
                     execution_id=execution_id, configuration=configuration,
                     manifest_source="coriqo" if remote else "local")
    if args.as_json:
        print(json.dumps(outcome, indent=2, sort_keys=True))
    else:
        _print_table(outcome)
    return EXIT_REPRODUCED if outcome["reproduced"] else EXIT_NOT_REPRODUCED


if __name__ == "__main__":
    raise SystemExit(main())
