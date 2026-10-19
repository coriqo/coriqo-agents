"""The identity and version of every interchange format this SDK
reads or writes, plus the loader for the checked-in JSON Schemas.

**These schemas are the source of truth, not a rendering of the code.** The
constants below are read back out of the schema files at import time rather
than typed a second time here, so a vocabulary in code that disagrees with
the published schema is not a thing that can exist — there is only one copy.
`docs/specs/` holds published copies of the same bytes; a byte-identity test
(`tests/test_fp15_published_schemas.py`) fails when they drift apart.

Every document in these formats carries `format` and `format_version` as
REQUIRED fields. A consumer that meets a `format_version` it does not know
must refuse the document. That is the whole reason the field is required
rather than defaulted: a defaulted version lets an old reader silently
misread a new document, and a benchmark score misread is a score reported as
something it is not.

`format_version` is a bare integer string ("1"), not a semver. It changes
only when an existing field is removed or its meaning changes. Adding an
optional field does not change it — a reader of version 1 must ignore
properties it does not recognise inside `trace` and `events`, which is why
those two objects (and only those two) allow additional properties.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

#: Format ids. The filename of each schema is `<format id>-<version>.schema.json`.
BENCHMARK_CORPUS_FORMAT = "coriqo-benchmark-corpus"
BENCHMARK_RESULT_FORMAT = "coriqo-benchmark-result"
VALIDATOR_MANIFEST_FORMAT = "coriqo-validator-manifest"

#: The only version of each format this SDK can read or write.
BENCHMARK_CORPUS_VERSION = "1"
BENCHMARK_RESULT_VERSION = "1"
VALIDATOR_MANIFEST_VERSION = "1"


class UnsupportedFormatError(ValueError):
    """A document names a format or format_version this SDK does not
    implement. Deliberately its own type: refusing a document is a normal,
    expected outcome a caller should be able to catch and report as "I
    cannot read this", not lump in with a corrupt file."""


def schema_path(format_id: str, version: str) -> Path:
    return SCHEMA_DIR / f"{format_id}-{version}.schema.json"


def load_schema(format_id: str, version: str) -> dict[str, Any]:
    path = schema_path(format_id, version)
    if not path.exists():
        raise UnsupportedFormatError(
            f"no checked-in schema for format {format_id!r} version {version!r} (looked in {path})"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def require_format(document: Any, format_id: str, version: str) -> None:
    """Refuse anything that is not this exact format at this exact version.

    Checked BEFORE any field is read, including by the JSON Schema
    validation that follows it, so the error a caller gets names the real
    problem ("this is a corpus, you handed me a result") rather than a
    cascade of missing-property complaints from the wrong schema.
    """
    if not isinstance(document, dict):
        raise UnsupportedFormatError(
            f"expected a JSON object carrying format {format_id!r}, got {type(document).__name__}"
        )
    found_format = document.get("format")
    if found_format != format_id:
        raise UnsupportedFormatError(
            f"document declares format {found_format!r}, this reader implements {format_id!r}"
        )
    found_version = document.get("format_version")
    if found_version != version:
        raise UnsupportedFormatError(
            f"document declares {format_id} format_version {found_version!r}; this reader implements "
            f"only {version!r}. Refusing to read it rather than guessing which fields still mean "
            "what they used to."
        )


def _corpus_expected_statuses() -> tuple[str, ...]:
    schema = load_schema(BENCHMARK_CORPUS_FORMAT, BENCHMARK_CORPUS_VERSION)
    enum = (
        schema["properties"]["cases"]["items"]["properties"]["expected"]
        ["additionalProperties"]["enum"]
    )
    return tuple(enum)


#: The statuses a corpus may label a case with, AND therefore the only
#: statuses an observed result can be compared against. Read off the
#: published schema, never retyped.
#:
#: This is rule 8's shape: the GOOD set is enumerated, so ERROR,
#: INCONCLUSIVE, and any status a third-party validator invents next year
#: all fall through to `unevaluated` without anyone having to remember to
#: add them to a bad-list. A benchmark that scored an unrecognised status
#: as a match would be a way to launder an incomplete run into a clean one.
COMPARABLE_STATUSES: tuple[str, ...] = _corpus_expected_statuses()

#: What a scored check can come out as.
OUTCOME_MATCHED = "matched"
OUTCOME_MISMATCHED = "mismatched"
OUTCOME_UNEVALUATED = "unevaluated"


def validate_against_schema(document: Any, format_id: str, version: str) -> None:
    """JSON Schema validation, on top of `require_format`.

    Needs the `spec` extra (`pip install coriqo-agents[spec]`); `jsonschema`
    is not a base dependency of this SDK. A caller without it gets a clear
    ImportError naming the extra rather than a silently skipped check —
    skipping validation because the library is missing would turn "this
    document conforms" into "nobody looked".
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - exercised by the extra's absence
        raise ImportError(
            "schema validation needs the `spec` extra: pip install coriqo-agents[spec]"
        ) from exc
    require_format(document, format_id, version)
    jsonschema.validate(document, load_schema(format_id, version))
