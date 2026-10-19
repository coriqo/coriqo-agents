"""`coriqo-agents runner bench`: the benchmark harness.

    coriqo-agents runner bench --corpus corpus.json --out result.json

Runs each corpus case's trace through exactly the validators that case
labels, and writes a `coriqo-benchmark-result` document. One command, no
server, no credentials, no network: a vendor runs it against their own
agent's recorded traces in their own environment and nothing leaves it.

Why the result is reproducible, which is the only property that makes a
published score worth anything:

  * no timestamp, no hostname, no run id in the output (bench_format.build_result)
  * validators are resolved from VALIDATOR_REGISTRY by id and run in the
    corpus's own order, so two machines run the same code in the same order
  * every validator's `evaluate` is deterministic — the conformance suite's
    DETERMINISM check is what establishes that, per validator
  * the corpus's own bytes are hashed into the result

`--validators` restricts the run to a subset. It cannot ADD a validator a
case does not label: an unlabelled (case, validator) pair has no expected
outcome, so running it would produce a check with nothing to compare
against, and a check with nothing to compare against has no place in a
score.

A validator that raises, or returns ERROR/INCONCLUSIVE, produces an
`unevaluated` check. It is never a match and never enters the denominator —
the harness cannot turn "this did not run" into "this passed".

A malformed case — `"trace": null`, an `expected` that is not an object —
is treated the same way, and for the same reason: it scores nothing, the
skip is written to stderr, and the rest of the run still reports. One bad
case in a corpus does not discard the scores of the good ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..openformat.bench_format import build_result, classify
from ..openformat.formats import (
    BENCHMARK_CORPUS_FORMAT,
    BENCHMARK_CORPUS_VERSION,
    UnsupportedFormatError,
    require_format,
)
from .contract import ValidatorContext
from .validators import VALIDATOR_REGISTRY

EXIT_OK = 0
EXIT_USAGE_ERROR = 2

#: The status recorded when a validator raised rather than returning a
#: result. Not one of the comparable statuses, so `classify` calls it
#: unevaluated — which is the honest reading: the harness has no verdict.
_CRASHED = "HARNESS_ERROR"


def _sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("coriqo-agents")
    except Exception:  # noqa: BLE001 - a source checkout is not an installed dist
        return "0.0.0+unknown"


def run_corpus(
    corpus: dict[str, Any],
    corpus_bytes: bytes,
    *,
    only: list[str] | None = None,
    skipped: list[str] | None = None,
) -> dict[str, Any]:
    """Score `corpus` and return the benchmark result document.

    `corpus` must already have been checked with `require_format` — this
    function does it again anyway, because it is public and a caller that
    skipped the check would otherwise get partial garbage rather than a
    refusal.

    A malformed *case* does not abort the run. It scores nothing and the
    other cases are still reported; pass a list as `skipped` to receive one
    line per thing that was passed over, so the degradation is visible
    rather than a quietly shorter `cases` array.
    """
    require_format(corpus, BENCHMARK_CORPUS_FORMAT, BENCHMARK_CORPUS_VERSION)

    cases: list[dict[str, Any]] = []
    validator_versions: dict[str, str] = {}
    if skipped is None:
        skipped = []

    for index, case in enumerate(corpus["cases"]):
        # A case whose own shape is unreadable scores nothing and is skipped,
        # rather than aborting the run: the scores already computed for the
        # other cases are real results and throwing them away to report one
        # bad case is the worse trade. Nothing here is silent — every skip is
        # written to stderr by `main`.
        try:
            case_id = case["case_id"]
            domain = case["domain"]
            labels = case["expected"]
        except (TypeError, KeyError, IndexError) as exc:
            skipped.append(f"cases[{index}]: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(case_id, str) or not isinstance(domain, str):
            skipped.append(f"cases[{index}]: case_id and domain must both be strings")
            continue
        if not isinstance(labels, dict) or not all(isinstance(k, str) for k in labels):
            skipped.append(
                f"cases[{case_id}]: `expected` must be an object keyed by validator id"
            )
            continue

        checks: list[dict[str, Any]] = []
        # Sorted so two machines emit checks in the same order regardless of
        # how the corpus file's JSON object happened to be written.
        for validator_id in sorted(labels):
            if only is not None and validator_id not in only:
                continue
            expected = labels[validator_id]
            if not isinstance(expected, str):
                # No readable label for this pair. Same reasoning as an
                # unlabelled (case, validator) pair: there is nothing to
                # compare against, and a check with nothing to compare
                # against has no place in a score.
                skipped.append(
                    f"cases[{case_id}].expected[{validator_id}] is not a status string"
                )
                continue
            validator_cls = VALIDATOR_REGISTRY.get(validator_id)
            if validator_cls is None:
                # A corpus labelling a validator this SDK does not ship is
                # not an error to abort on — a corpus is allowed to be
                # newer than the harness. It is an unevaluated check, and
                # it says so in `observed`.
                checks.append({
                    "validator_id": validator_id,
                    "validator_version": "unknown",
                    "expected": expected,
                    "observed": "NOT_IMPLEMENTED",
                    "outcome": classify(expected, "NOT_IMPLEMENTED"),
                })
                continue
            # Everything that reads the case's own fields is inside the try
            # as well as the validator call: a case with `"trace": null` or a
            # non-object `configuration` is a malformed case, and the module
            # promises that produces an unevaluated check — not an
            # AttributeError out of the middle of a multi-case run.
            version_ran = "unknown"
            try:
                validator = validator_cls({})
                version_ran = getattr(validator, "version", "unknown")
                trace = case["trace"]
                context = ValidatorContext(
                    execution_id=f"bench:{case_id}",
                    agent_model_id=trace.get("agent_id", ""),
                    agent_version_ref=trace.get("agent_version"),
                    environment="BENCHMARK",
                    configuration=dict(case.get("configuration") or {}),
                    inputs={"trace": trace},
                )
                result = validator.evaluate(validator.execute(context))
                observed = result.status
                version_ran = result.version
            except Exception:  # noqa: BLE001 - a crashing validator scores nothing, not zero
                observed = _CRASHED
                if not isinstance(version_ran, str):
                    version_ran = "unknown"
            validator_versions[validator_id] = version_ran
            checks.append({
                "validator_id": validator_id,
                "validator_version": version_ran,
                "expected": expected,
                "observed": observed,
                "outcome": classify(expected, observed),
            })
        if not checks:
            # Every validator this case labels was filtered out by
            # --validators. Dropping the case entirely is right: the result
            # schema requires at least one check per case, and a case with
            # no checks would otherwise be published as a case that was run.
            continue
        cases.append({"case_id": case_id, "domain": domain, "checks": checks})

    return build_result(
        corpus=corpus,
        corpus_bytes=corpus_bytes,
        cases=cases,
        sdk_version=_sdk_version(),
        validator_versions=validator_versions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coriqo-agents runner bench",
        description=(
            "Run a Coriqo agent assurance benchmark corpus through the validators and write a "
            "reproducible result document. Nothing is uploaded and nothing leaves this machine."
        ),
    )
    parser.add_argument("--corpus", required=True, type=Path, help="Benchmark corpus JSON file.")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the result document. Defaults to stdout.",
    )
    parser.add_argument(
        "--validators", default=None,
        help="Comma-separated validator ids to restrict the run to. Cannot add a validator a "
             "case does not label.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        corpus_bytes = args.corpus.read_bytes()
    except OSError as exc:
        print(f"could not read the corpus at {args.corpus}: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    try:
        corpus = json.loads(corpus_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"{args.corpus} is not valid JSON: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    only = [v.strip() for v in args.validators.split(",") if v.strip()] if args.validators else None
    skipped: list[str] = []
    try:
        result = run_corpus(corpus, corpus_bytes, only=only, skipped=skipped)
    except UnsupportedFormatError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE_ERROR
    except KeyError as exc:
        print(f"the corpus is missing a required field: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except (TypeError, AttributeError, ValueError) as exc:
        # The corpus's top-level shape, not a single case's — `run_corpus`
        # degrades a bad case rather than raising, so anything reaching here
        # means the document itself is not a corpus.
        print(f"the corpus is not readable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    for note in skipped:
        print(f"skipped (scores nothing): {note}", file=sys.stderr)

    document = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(document)
    else:
        args.out.write_text(document, encoding="utf-8")
        summary = result["score"]
        ratio = "n/a" if summary["ratio"] is None else f"{summary['ratio']:.3f}"
        print(
            f"{result['totals']['cases']} case(s), {summary['matched']}/{summary['scored']} "
            f"checks matched (ratio {ratio}), {summary['unevaluated']} unevaluated "
            f"-> {args.out}",
            file=sys.stderr,
        )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
