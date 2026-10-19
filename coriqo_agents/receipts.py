"""Offline verification of an evidence receipt (an inclusion-proof bundle).

Each check reports pass, fail or not_checked with a reason. A failed check
never hides the others, so a caller can see exactly which part broke.
"""
from __future__ import annotations

import builtins
import io
import sys
import threading

from . import _seal_verifier as sv

# The vendored verifier print()s diagnostics. Rather than swap the global
# sys.stdout (which would swallow other threads' output), shadow `print` inside
# the vendored module only, writing to a per-thread buffer while a call runs.
_tls = threading.local()


def _module_print(*args, **kwargs):
    buf = getattr(_tls, "buf", None)
    if buf is None:
        return builtins.print(*args, **kwargs)
    kwargs.pop("file", None)
    return builtins.print(*args, file=buf, **kwargs)


sv.print = _module_print


def _quiet(fn, *a, **k):
    prev = getattr(_tls, "buf", None)
    _tls.buf = io.StringIO()
    try:
        return fn(*a, **k)
    finally:
        captured = _tls.buf.getvalue()
        _tls.buf = prev
        if captured:
            sys.stderr.write(captured)


PASS, FAIL, NOT_CHECKED = "pass", "fail", "not_checked"
CHECKS = ("leaf_inclusion", "root_in_sth", "signature", "key_pinned", "tsa_imprint", "rekor")


def _r(status: str, reason: str = "") -> dict:
    return {"status": status, "reason": reason}


def verify_receipt(bundle: dict, public_key_pem: str | None = None,
                   *, allow_unpinned: bool = False) -> dict:
    """Verify `bundle` offline. Returns {"ok": bool, "checks": {name: {status, reason}}}.

    `ok` is True only when no check failed and leaf_inclusion, root_in_sth,
    signature and key_pinned all passed. Pass `public_key_pem` to pin a key you
    obtained outside Coriqo. Without it the bundle's own key is used, key_pinned
    is not_checked and `ok` is False, because a key taken from the bundle proves
    consistency only. `allow_unpinned=True` accepts that explicitly.
    """
    checks: dict[str, dict] = {}
    sth = bundle.get("sth_body") or {}

    problems = []
    for f in ("merkle_root", "tree_size"):
        if sth.get(f) is None:
            problems.append(f"sth_body is missing {f}")
        elif bundle.get(f) != sth.get(f):
            problems.append(f"{f} in the bundle disagrees with sth_body")
    checks["root_in_sth"] = (
        _r(FAIL, "; ".join(problems)) if problems
        else _r(PASS, "merkle_root and tree_size match the signed body"))

    if checks["root_in_sth"]["status"] != PASS:
        checks["leaf_inclusion"] = _r(
            NOT_CHECKED, "skipped: root_in_sth failed, so there is no signed root to check against")
    else:
        try:
            ok = _quiet(
                sv.verify_inclusion, bundle["leaf_input"], int(bundle["leaf_index"]),
                int(sth["tree_size"]), list(bundle["proof_path"]), sth["merkle_root"],
                int(sth.get("tree_version", 1)))
            checks["leaf_inclusion"] = _r(
                PASS if ok else FAIL,
                "leaf hashes up to the signed merkle_root" if ok
                else "proof_path does not reproduce the signed merkle_root")
        except (KeyError, ValueError, TypeError) as exc:
            checks["leaf_inclusion"] = _r(FAIL, f"malformed bundle: {exc!r}")

    pinned = public_key_pem is not None
    key = public_key_pem or bundle.get("public_key_pem")
    sig = bundle.get("checkpoint_signature")
    if not key or not sig:
        checks["signature"] = _r(FAIL, "no public key or signature to check")
    else:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            checks["signature"] = _r(
                NOT_CHECKED, "install coriqo-agents[receipts] (cryptography)")
        else:
            try:
                ok = _quiet(
                    sv.verify_signature, sth, sig, key, sth.get("algorithm", sv.ALGORITHM_ED25519))
            except ValueError:
                ok = False
            src = "pinned key" if pinned else "key from the bundle, not independently pinned"
            checks["signature"] = _r(
                PASS if ok else FAIL,
                f"signature valid ({src})" if ok else "signature does not verify")

    checks["key_pinned"] = (
        _r(PASS, "verified against a key you supplied") if pinned
        else _r(NOT_CHECKED, "no pinned public_key_pem; the bundle's own key was used"))

    tsa = bundle.get("tsa")
    token = tsa.get("tsa_token") if isinstance(tsa, dict) else None
    if not token:
        checks["tsa_imprint"] = _r(NOT_CHECKED, "bundle carries no TSA token")
    else:
        try:
            info = _quiet(sv._verify_tsa_offline, token, sth, None)
        except Exception as exc:  # noqa: BLE001
            info = {"available": False, "detail": f"token unreadable: {exc}"}
        if not info.get("available"):
            checks["tsa_imprint"] = _r(NOT_CHECKED, info.get("detail", "TSA check unavailable"))
        elif not info.get("signature_verified"):
            checks["tsa_imprint"] = _r(
                FAIL, "TSA token CMS signature is invalid: " + (info.get("detail") or "does not verify"))
        else:
            ca = ("signing CA pinned" if info.get("ca_verified") else
                  "signing CA NOT pinned" if info.get("ca_verified") is False else
                  "signing CA not checked")
            ctx = f"gen_time {info.get('gen_time') or 'unknown'}; {ca}"
            checks["tsa_imprint"] = (
                _r(PASS, f"token imprint equals SHA-256 of sth_body; {ctx}")
                if info.get("imprint_matches")
                else _r(FAIL, f"token imprint does not match sth_body; {ctx}"))

    checks["rekor"] = _r(
        NOT_CHECKED,
        "Rekor is not checked offline"
        + (" (bundle has an anchor but no inclusion proof)" if bundle.get("anchor") else ""))

    required = ["leaf_inclusion", "root_in_sth", "signature"]
    if not allow_unpinned:
        required.append("key_pinned")
    ok = (not any(c["status"] == FAIL for c in checks.values())
          and all(checks[n]["status"] == PASS for n in required))
    return {"ok": ok, "checks": {n: checks[n] for n in CHECKS}}
