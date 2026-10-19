"""`verify-assurance-package` — the offline reference verifier.

    verify-assurance-package package.json
    verify-assurance-package package.json --content-hash <hash from the seal receipt>

Exit codes, because a pipeline branches on them:

  0  verified   — every check ran and passed
  1  failed     — a check ran and failed
  2  unreadable — not JSON, or a schema_version this tool does not implement
  3  incomplete — nothing failed, but at least one check could not run.
                  NOT a pass: the most common reason is that no content hash
                  was supplied, which means nothing tied the document to
                  anything Coriqo sealed.

There is no `--pubkey`. See package_verify.py's module docstring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .formats import UnsupportedFormatError
from .package_verify import (
    VERDICT_FAILED,
    VERDICT_INCOMPLETE,
    VERDICT_VERIFIED,
    verify_package_file,
)

EXIT_VERIFIED = 0
EXIT_FAILED = 1
EXIT_UNREADABLE = 2
EXIT_INCOMPLETE = 3

_EXIT_FOR_VERDICT = {
    VERDICT_VERIFIED: EXIT_VERIFIED,
    VERDICT_FAILED: EXIT_FAILED,
    VERDICT_INCOMPLETE: EXIT_INCOMPLETE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify-assurance-package",
        description=(
            "Check a Coriqo assurance package offline: its declared version, its structure "
            "against the published schema, its internal references, and (when you supply one) "
            "its content hash. Does not check a signature and makes no authenticity claim."
        ),
    )
    parser.add_argument("package", type=Path, help="The assurance package JSON file.")
    parser.add_argument(
        "--content-hash", default=None,
        help="The content_hash from the seal receipt Coriqo returned with this package. "
             "Without it the CONTENT_HASH check cannot run and the verdict is `incomplete`.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the full report as JSON instead of the human-readable summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_package_file(args.package, expected_content_hash=args.content_hash)
    except OSError as exc:
        print(f"could not read {args.package}: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    except json.JSONDecodeError as exc:
        print(f"{args.package} is not valid JSON: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    except UnsupportedFormatError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_UNREADABLE

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"verdict: {report['verdict']}")
        for check in report["checks"]:
            line = f"  {check['status']:<12} {check['check_id']}"
            if check["detail"]:
                line += f" — {check['detail']}"
            print(line)
        print(f"\n{report['note']}")
    # A verdict this CLI has no exit code for is a programming error in
    # package_verify, not something to report as success.
    return _EXIT_FOR_VERDICT.get(report["verdict"], EXIT_FAILED)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
