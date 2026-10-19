"""Offline verification of an assurance package, with an explicit
boundary around what "verified" is allowed to mean.

**This does not check a signature, and it never says a package is
authentic.** A Coriqo assurance package is sealed by appending it to the
tenant's Ed25519-signed, Merkle-chained governance ledger;
the signature is over a
ledger event, not over the package document, and the document a bank
receives carries the seal receipt (`content_hash`, `governance_event_id`)
beside it rather than inside it. There is no sealed-envelope format today
that an offline tool could check with a public key, so this module does not
offer a `--pubkey` flag that would imply there is. Publishing one would be
the worst kind of promise: a verifier whose output reads like an
authenticity claim while checking nothing of the sort.

What it can do offline, and does:

  VERSION       the package declares a schema_version this tool implements
  SCHEMA        the document validates against the published JSON Schema
  REFERENCES    every id the document cites internally exists in it
  CONTENT_HASH  the document re-hashes to a `content_hash` you supply
                separately (from the seal receipt, or from a Coriqo API
                response) — this proves the document has not been edited
                since sealing, GIVEN that you trust where the hash came
                from. That "given" is the whole difference between this and
                a signature check.

A verdict is `verified` only when every check ran and passed. A check that
could not run makes the verdict `incomplete`, never `verified` — the same
rule the benchmark scoring applies, for the same reason.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .formats import SCHEMA_DIR, UnsupportedFormatError

#: schema_version -> the published schema file for it, or None when the
#: version is one this tool recognises but has no checked-in schema for.
#: 1.0 is real and still readable by consumers; only 1.1 was ever published
#: as a schema file, so a 1.0 package gets a disclosed unevaluated SCHEMA
#: check rather than a silent pass.
KNOWN_SCHEMA_VERSIONS: dict[str, str | None] = {
    "1.0": None,
    "1.1": "assurance-package-1.1.schema.json",
}

CHECK_PASS = "PASS"
CHECK_FAIL = "FAIL"
CHECK_UNEVALUATED = "UNEVALUATED"

VERDICT_VERIFIED = "verified"
VERDICT_FAILED = "failed"
VERDICT_INCOMPLETE = "incomplete"


def canonical_content_hash(package: dict[str, Any]) -> str:
    """The hash Coriqo's seal computed over this package.

    Mirrors the canonical hash Coriqo's API computes when sealing:
    sha256 of JSON with sorted keys, no separators whitespace, non-ASCII
    left as-is — over the package document exactly as it arrived.

    It does NOT strip nulls. Coriqo seals
    `package.model_dump(exclude_none=True)`, and `exclude_none` applies to
    the model's own fields, not to nulls sitting inside the plain dicts
    those fields hold — a validator result's `"severity": null` is in the
    sealed payload and therefore in the hash. A first version of this
    function recursed and dropped them, and it disagreed with every sealed
    record; `test_fp_15_offline_verifier_parity.py` is what caught that.

    This is a second implementation of the server's canonicalisation, in a
    package that deliberately cannot import it. The two are held together
    only by that test. If it is ever deleted, this docstring is a lie and
    every offline hash check starts silently reporting untouched packages
    as edited.
    """
    return hashlib.sha256(
        json.dumps(package, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()


def _reference_problems(package: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    validators = package.get("validators") or []
    validator_ids = {v.get("validator_id") for v in validators if isinstance(v, dict)}
    manifest = set(package.get("evidence_manifest") or [])

    for index, finding in enumerate(package.get("findings") or []):
        if not isinstance(finding, dict):
            problems.append(f"findings[{index}] is not an object")
            continue
        vid = finding.get("validator_id")
        # A finding with no validator_id at all is not a reference problem:
        # the package schema does not require one, and a finding raised by
        # a reviewer rather than a validator legitimately has none.
        if vid is not None and vid not in validator_ids:
            problems.append(
                f"findings[{index}] cites validator_id {vid!r}, which is not in `validators`"
            )

    for index, validator in enumerate(validators):
        if not isinstance(validator, dict):
            problems.append(f"validators[{index}] is not an object")
            continue
        for evidence_id in validator.get("evidence_ids") or []:
            if evidence_id not in manifest:
                problems.append(
                    f"validators[{index}] ({validator.get('validator_id')!r}) cites evidence "
                    f"{evidence_id!r}, which is not in `evidence_manifest`"
                )
    return problems


def verify_package(
    package: Any,
    *,
    expected_content_hash: str | None = None,
) -> dict[str, Any]:
    """Run the offline checks. Raises `UnsupportedFormatError` when the
    document is not a package this tool can read at all — refusing is the
    correct behaviour for an unknown version, not reading it partially."""
    if not isinstance(package, dict):
        raise UnsupportedFormatError(
            f"an assurance package is a JSON object; got {type(package).__name__}"
        )
    declared = package.get("schema_version")
    # `declared` is whatever was in the JSON, so it can be a list or an
    # object — both unhashable, and `x in {...}` on an unhashable x is a
    # TypeError, not a miss. Checking the type first keeps every malformed
    # version on the one documented path: UnsupportedFormatError, which the
    # CLI reports as a verification failure instead of a traceback.
    if not isinstance(declared, str) or declared not in KNOWN_SCHEMA_VERSIONS:
        raise UnsupportedFormatError(
            f"package declares schema_version {declared!r}; this verifier implements "
            f"{sorted(KNOWN_SCHEMA_VERSIONS)}. Refusing to read it — a package from a newer "
            "version may have changed what an existing field means."
        )

    checks: list[dict[str, str]] = [
        {"check_id": "VERSION", "status": CHECK_PASS, "detail": f"schema_version {declared}"}
    ]

    schema_file = KNOWN_SCHEMA_VERSIONS[declared]
    if schema_file is None:
        checks.append({
            "check_id": "SCHEMA", "status": CHECK_UNEVALUATED,
            "detail": f"no JSON Schema was ever published for schema_version {declared}; "
                      "the document's structure was not checked",
        })
    else:
        try:
            import jsonschema
        except ImportError:
            checks.append({
                "check_id": "SCHEMA", "status": CHECK_UNEVALUATED,
                "detail": "jsonschema is not installed (pip install coriqo-agents[spec]); "
                          "the document's structure was not checked",
            })
        else:
            schema = json.loads((SCHEMA_DIR / schema_file).read_text(encoding="utf-8"))
            try:
                jsonschema.validate(package, schema)
            except jsonschema.ValidationError as exc:
                checks.append({
                    "check_id": "SCHEMA", "status": CHECK_FAIL,
                    "detail": f"{'/'.join(str(p) for p in exc.absolute_path) or '<root>'}: {exc.message}",
                })
            else:
                checks.append({"check_id": "SCHEMA", "status": CHECK_PASS, "detail": schema_file})

    problems = _reference_problems(package)
    checks.append(
        {"check_id": "REFERENCES", "status": CHECK_PASS, "detail": ""}
        if not problems else
        {"check_id": "REFERENCES", "status": CHECK_FAIL, "detail": "; ".join(problems)}
    )

    if expected_content_hash is None:
        checks.append({
            "check_id": "CONTENT_HASH", "status": CHECK_UNEVALUATED,
            "detail": "no content hash was supplied, so nothing establishes that this document "
                      "is the one Coriqo sealed",
        })
    else:
        actual = canonical_content_hash(package)
        if actual == expected_content_hash.strip().lower():
            checks.append({"check_id": "CONTENT_HASH", "status": CHECK_PASS, "detail": actual})
        else:
            checks.append({
                "check_id": "CONTENT_HASH", "status": CHECK_FAIL,
                "detail": f"document hashes to {actual}, expected {expected_content_hash}",
            })

    # Enumerate the good status and require every check to hold it. A check
    # in any other state — FAIL, UNEVALUATED, or something a later version
    # of this file adds — keeps the verdict away from `verified`.
    if any(c["status"] == CHECK_FAIL for c in checks):
        verdict = VERDICT_FAILED
    elif all(c["status"] == CHECK_PASS for c in checks):
        verdict = VERDICT_VERIFIED
    else:
        verdict = VERDICT_INCOMPLETE

    return {
        "verdict": verdict,
        "schema_version": declared,
        "checks": checks,
        "signature_checked": False,
        "note": (
            "This tool does not check a signature and makes no claim that the package is "
            "authentic. See coriqo_agents.openformat.package_verify for what it does check."
        ),
    }


def verify_package_file(path: Path, *, expected_content_hash: str | None = None) -> dict[str, Any]:
    return verify_package(
        json.loads(path.read_text(encoding="utf-8")),
        expected_content_hash=expected_content_hash,
    )
