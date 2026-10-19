"""Canonical hashing for tool-call args/results.

Serializes the same way Coriqo's own server does for input/output hashing
(sorted keys, compact separators), but is NOT byte-identical to it: this prepends "sha256:"
(matching the format shown in the integration guide's examples) while
_canonical_hash returns a bare hex digest. The server never recomputes or
cross-checks a tool_calls args_hash/result_hash against anything — these are
opaque, caller-provided audit values — so the difference has no functional
effect today. If you ever need to compare a hash from here against one
computed independently with the server's bare-hex convention, strip the
"sha256:" prefix first.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_value(obj: Any) -> str | None:
    """sha256 of *obj* as canonical JSON, prefixed "sha256:" to match the
    format shown in the integration guide. None in, None out — a tool call
    with no args or no result just omits that hash rather than hashing null."""
    if obj is None:
        return None
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def mandate_sha256(text: str | None) -> str:
    """Bare-hex sha256 of a mandate's text, for `coriqo-agents mandate check`.

    Line endings become "\\n" and trailing whitespace is dropped, so a file
    saved on Windows or with a final newline hashes the same as the text held
    in Coriqo. Inner whitespace is content. Coriqo's API
    does the same on the server; both tests pin one set of vectors."""
    normalised = (text or "").replace("\r\n", "\n").replace("\r", "\n").rstrip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
