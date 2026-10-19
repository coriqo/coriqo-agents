"""Deterministic PII/sensitive-data detection for the data_exfiltration
validator (validators/data_exfiltration.py).

This is a deliberate, documented DUPLICATE of the detection patterns in
Coriqo's server-side DLP redaction, not an import of it: this package ships standalone to
customer environments and cannot depend on server code — see manifest.py's
`canonical_bytes` for the exact same constraint already accepted in this
package (a byte-for-byte reimplementation, not a shared import, with a
comment pinning the two to stay in lockstep). If the server's
detectors change, this module should be re-diffed against it the same way.

Only the detection side is reused here (`find_sensitive_spans`) — the
validator does not need the server's masking/redaction behavior (nothing here
is ever written back to a document), only "does this payload contain data
classified as sensitive," which is what data_exfiltration.py's FAIL
condition is keyed on.

Every detector is a regex plus (where one exists) the same arithmetic check
the server uses (Luhn for card PANs, ABA checksum, IBAN mod-97):
deterministic, no LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["SensitiveSpan", "SECRET_PATTERNS", "find_sensitive_spans", "find_secret_spans"]

_CONTEXT_WINDOW = 48

_SSN_CONTEXT = re.compile(r"\b(?:ssn|tin|social\s+security)\b", re.IGNORECASE)
_ACCT_CONTEXT = re.compile(r"\b(?:account\s*(?:number|no\.?|#)?|acct|iban)\b", re.IGNORECASE)
_ABA_CONTEXT = re.compile(r"\b(?:routing|aba|wire|wires?)\b", re.IGNORECASE)

_PAN_RE = re.compile(r"(?<![\d*])(?:\d[ -]?){12,18}\d(?![\d*-])")
_SSN_DASHED_RE = re.compile(r"(?<![\d*-])\d{3}-\d{2}-\d{4}(?![\d-])")
_NINE_DIGIT_RE = re.compile(r"(?<![\d*.-])\d{9}(?![\d.-])")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

_SECRET_RES = [
    # PEM first: a private key block contains base64 that would otherwise be
    # shredded into partial JWT-lookalike hits — same ordering rationale as
    # the server's DLP.
    re.compile(
        r"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----[A-Za-z0-9+/=\s]{0,8000}?"
        r"-----END [A-Z ]{0,30}PRIVATE KEY-----"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[po]_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b"),
    # "password: hunter2" / pwd=..., same as the server's DLP.
    re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S{1,256}", re.IGNORECASE),
]


#: Public alias for `_SECRET_RES` (the `credential_isolation` validator
#: needs to scan specifically for credential-shaped content — PEM blocks,
#: password labels, provider key prefixes, JWT shape — without pulling in
#: the PAN/SSN/IBAN detectors below). Same objects, not a copy: a pattern
#: added to `_SECRET_RES` is visible here immediately, and vice versa.
SECRET_PATTERNS: tuple = tuple(_SECRET_RES)


@dataclass
class SensitiveSpan:
    kind: str  # "card_pan" | "ssn" | "bank_account" | "secret"
    start: int
    end: int


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(iban: str) -> bool:
    rearranged = iban[4:] + iban[:4]
    try:
        num = "".join(str(int(c, 36)) for c in rearranged)
        return int(num) % 97 == 1
    except ValueError:
        return False


def _aba_ok(digits: str) -> bool:
    # ABA checksum: weights 3,7,1 repeating over the 9 digits — same as
    # the server's DLP.
    weights = (3, 7, 1, 3, 7, 1, 3, 7, 1)
    return sum(w * int(d) for w, d in zip(weights, digits)) % 10 == 0


def _has_context(text: str, start: int, end: int, ctx: re.Pattern) -> bool:
    window = text[max(0, start - _CONTEXT_WINDOW): end + _CONTEXT_WINDOW]
    return ctx.search(window) is not None


def find_sensitive_spans(text: str) -> list[SensitiveSpan]:
    """Every sensitive-data match in `text`, kind-labeled. Empty list means
    "nothing detected" — never raises on arbitrary input."""
    if not text:
        return []

    hits: list[SensitiveSpan] = []

    for pat in _SECRET_RES:
        for m in pat.finditer(text):
            hits.append(SensitiveSpan("secret", m.start(), m.end()))

    for m in _PAN_RE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits.append(SensitiveSpan("card_pan", m.start(), m.end()))

    for m in _SSN_DASHED_RE.finditer(text):
        hits.append(SensitiveSpan("ssn", m.start(), m.end()))

    for m in _NINE_DIGIT_RE.finditer(text):
        if _has_context(text, m.start(), m.end(), _SSN_CONTEXT):
            hits.append(SensitiveSpan("ssn", m.start(), m.end()))

    # ABA routing: 9 digits + checksum + context — checked on the same
    # 9-digit candidates as the SSN pass above, same as the server's DLP.
    for m in _NINE_DIGIT_RE.finditer(text):
        if _aba_ok(m.group()) and _has_context(text, m.start(), m.end(), _ABA_CONTEXT):
            hits.append(SensitiveSpan("bank_account", m.start(), m.end()))

    for m in _IBAN_RE.finditer(text):
        if _iban_ok(m.group()):
            hits.append(SensitiveSpan("bank_account", m.start(), m.end()))

    for m in re.finditer(r"(?<![\d*.-])\d{8,}(?![\d.-])", text):
        if _has_context(text, m.start(), m.end(), _ACCT_CONTEXT):
            hits.append(SensitiveSpan("bank_account", m.start(), m.end()))

    return hits


def find_secret_spans(text: str) -> list[SensitiveSpan]:
    """Just the `kind == "secret"` subset of `find_sensitive_spans` —
    credential material (PEM keys, provider key prefixes, JWT shape,
    password labels), not PAN/SSN/bank-account content. `credential_isolation.py`
    uses this directly rather than the full detector so a card number in a
    tool argument (a data_exfiltration concern) doesn't also trip a
    credential-isolation FAIL."""
    if not text:
        return []
    hits: list[SensitiveSpan] = []
    for pat in SECRET_PATTERNS:
        for m in pat.finditer(text):
            hits.append(SensitiveSpan("secret", m.start(), m.end()))
    return hits
