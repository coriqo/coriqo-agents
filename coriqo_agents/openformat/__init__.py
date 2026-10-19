"""The open interchange formats, the validation SDK's conformance
suite, and the offline package verifier.

Everything in this subpackage is a promise to somebody outside Coriqo. A
field published here is a field that has to keep meaning what it means; the
way to change one is a new `format_version`, not an edit. See
`docs/specs/CHANGELOG.md` for what has changed and `formats.py` for how
versioning works.

Nothing here calls a model. Both producing a benchmark record and checking
one are string comparisons over deterministic validator output, which is what makes a published score something a stranger can
reproduce rather than something they have to trust.
"""
from __future__ import annotations

from .formats import (
    BENCHMARK_CORPUS_FORMAT,
    BENCHMARK_CORPUS_VERSION,
    BENCHMARK_RESULT_FORMAT,
    BENCHMARK_RESULT_VERSION,
    COMPARABLE_STATUSES,
    VALIDATOR_MANIFEST_FORMAT,
    VALIDATOR_MANIFEST_VERSION,
    UnsupportedFormatError,
    load_schema,
    require_format,
    validate_against_schema,
)

__all__ = [
    "BENCHMARK_CORPUS_FORMAT",
    "BENCHMARK_CORPUS_VERSION",
    "BENCHMARK_RESULT_FORMAT",
    "BENCHMARK_RESULT_VERSION",
    "COMPARABLE_STATUSES",
    "VALIDATOR_MANIFEST_FORMAT",
    "VALIDATOR_MANIFEST_VERSION",
    "UnsupportedFormatError",
    "load_schema",
    "require_format",
    "validate_against_schema",
]
