"""`coriqo-agents runner conformance` — run the validator conformance suite
against a validator you wrote, named by its manifest.

    coriqo-agents runner conformance --manifest my-validator.json --trace sample.json

Exit codes: 0 every check passed; 1 at least one check failed; 2 the
manifest, the trace, or the entry point could not be read.

The manifest is checked against the published `coriqo-validator-manifest`
schema first. It is a claim by its author and this command does not verify
any of it beyond shape — except `validator_id` and `version`, which the
suite compares against what the class actually declares, because a manifest
that names a different id than the code writes into its results would put
somebody else's name on your findings.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .conformance import CHECK_FAIL, CHECK_PASS, CheckOutcome, check_conformance
from .formats import (
    VALIDATOR_MANIFEST_FORMAT,
    VALIDATOR_MANIFEST_VERSION,
    UnsupportedFormatError,
    validate_against_schema,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE_ERROR = 2


def load_entry_point(entry_point: str):
    module_path, _, class_name = entry_point.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coriqo-agents runner conformance",
        description="Check that a validator is identifiable, versioned, deterministic and pure.",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Validator manifest JSON.")
    parser.add_argument(
        "--trace", required=True, type=Path,
        help="A sample trace JSON file to evaluate. The determinism and purity checks are only "
             "as strong as the trace you give them — use one your validator actually reacts to.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read an input: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    # The manifest is schema-checked below; the trace was not checked at all,
    # so a file holding `null`, a list or a bare string reached
    # check_conformance and died on `input_trace.get(...)` with a raw
    # AttributeError traceback — the one malformed-input path in this CLI that
    # still crashed instead of reporting. Same refuse-and-say-why shape as
    # every other bad input here.
    if not isinstance(trace, dict):
        kind = "null" if trace is None else type(trace).__name__
        print(f"the trace at {args.trace} is a JSON {kind}, not an object — "
              "a trace is a JSON object with an `events` list.", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        validate_against_schema(manifest, VALIDATOR_MANIFEST_FORMAT, VALIDATOR_MANIFEST_VERSION)
    except UnsupportedFormatError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - jsonschema.ValidationError, kept importable-free
        print(f"the manifest does not conform to {VALIDATOR_MANIFEST_FORMAT} "
              f"version {VALIDATOR_MANIFEST_VERSION}: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        validator_cls = load_entry_point(manifest["entry_point"])
    except (ImportError, AttributeError, ValueError) as exc:
        print(f"could not import entry_point {manifest['entry_point']!r}: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    report = check_conformance(validator_cls, trace=trace)
    # Read BEFORE the manifest checks are appended. `ConformanceReport.passed`
    # also asserts that every one of the six core checks has an outcome, so
    # extending `checks` with ids it does not know about makes it False no
    # matter how the suite went — which is how the first version of this CLI
    # printed six passes and then "FAILED".
    suite_passed = report.passed

    # The manifest is a published claim about the code; these two fields are
    # the ones a mismatch would misattribute a result under, so they are
    # checked rather than trusted.
    for field_name in ("validator_id", "version"):
        declared = manifest.get(field_name)
        actual = getattr(validator_cls, field_name, None)
        report.checks.append(CheckOutcome(
            f"MANIFEST_{field_name.upper()}",
            CHECK_PASS if declared == actual else CHECK_FAIL,
            "" if declared == actual
            else f"the manifest says {declared!r}, the class declares {actual!r}",
        ))

    manifest_checks_passed = all(
        c.status == CHECK_PASS for c in report.checks if c.check_id.startswith("MANIFEST_")
    )
    passed = suite_passed and manifest_checks_passed

    if args.as_json:
        payload = report.to_dict()
        # `to_dict` recomputes `passed` off the extended check list for the
        # same reason as above; overwrite it with the verdict this CLI
        # actually reached.
        payload["passed"] = passed
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{manifest['validator_id']}: {'PASSED' if passed else 'FAILED'}")
        for check in report.checks:
            line = f"  {check.status:<6} {check.check_id}"
            if check.detail:
                line += f" — {check.detail}"
            print(line)
    return EXIT_OK if passed else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
