"""Vendored verification primitives from seal-verifier/verify_proof.py.

Copied verbatim (Apache-2.0, see seal-verifier/LICENSE) rather than imported:
seal-verifier ships as a single script with no pyproject or wheel, so this
package cannot depend on it, and importing by path would break for anyone who
pip-installs the SDK. Do not edit the copied bodies. Change them upstream and
re-copy. tests/test_receipts.py::test_vendored_copy_matches_seal_verifier
fails if the two drift.

Copied: RFC 6962 inclusion checks, canonical JSON, signature check, and the
TSA token check. Optional imports (cryptography, rfc3161ng) stay lazy inside
the functions, as upstream.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timezone

EXIT_OK = 0
EXIT_TAMPER = 1        # signature invalid, or a hash link in the chain broke
EXIT_UNSEALED = 2       # no certificate.json — nothing to cryptographically verify
EXIT_MALFORMED = 3      # bundle/manifest/certificate isn't the shape this tool expects


# ---------------------------------------------------------------------------
# RFC 6962 Merkle primitives (copy of attestation/merkle.py — self-contained
# so examiners don't need to install the Coriqo package)
# ---------------------------------------------------------------------------

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


SUPPORTED_TREE_VERSIONS = (1, 2)


def _verify_inclusion_v1(
    leaf_input_hex: str,
    leaf_index: int,
    tree_size: int,
    proof_path_hex: list[str],
    expected_root_hex: str,
) -> bool:
    """
    Version-1 tree: odd levels pair the last node with itself (SPEC.md §5.2).

    Frozen. Every checkpoint sealed before tree_version 2 was signed against
    this shape and must keep verifying against it.
    """
    # The proof path must be exactly as deep as the declared tree. Without
    # this, tree_size is decorative: the walk below only compares a computed
    # root, so a path of any length that happens to reproduce the signed root
    # would pass while claiming a tree of a different size. The v1 tree
    # duplicates the last node on odd levels, which means a tree of n leaves
    # and one of n+1 leaves whose last leaf repeats share a root -- so the
    # declared size has to be checked, not assumed.
    #
    # This rule applies to v1 ONLY. Under RFC 6962 the path length varies by
    # index, so applying it to a v2 proof rejects honest records.
    expected_depth = 0 if tree_size == 1 else (tree_size - 1).bit_length()
    if len(proof_path_hex) != expected_depth:
        print(f"  [ERROR] proof_path has {len(proof_path_hex)} sibling(s); a v1 tree of "
              f"{tree_size} leaves requires exactly {expected_depth}")
        return False

    computed = _leaf_hash(bytes.fromhex(leaf_input_hex))
    idx = leaf_index

    for sibling_hex in proof_path_hex:
        sibling = bytes.fromhex(sibling_hex)
        if idx % 2 == 0:
            computed = _node_hash(computed, sibling)
        else:
            computed = _node_hash(sibling, computed)
        idx //= 2

    return computed == bytes.fromhex(expected_root_hex)


def _verify_inclusion_v2(
    leaf_input_hex: str,
    leaf_index: int,
    tree_size: int,
    proof_path_hex: list[str],
    expected_root_hex: str,
) -> bool:
    """
    Version-2 tree: RFC 6962 §2.1.1 inclusion-proof verification.

    tree_size is threaded through the walk (fn/sn) instead of being checked
    separately. sn must reach exactly 0 as the path is consumed: a path that is
    too short leaves sn != 0, and one that is too long trips sn == 0 mid-walk.
    So the declared size is verified by the algorithm itself, and no separate
    proof-depth rule is needed here -- nor would a fixed-depth rule be correct,
    because RFC 6962 audit-path length depends on the leaf's index as well as
    the tree size (a 5,161-leaf tree has valid lengths 4, 7, 8, 12 and 13).
    """
    fn = leaf_index
    sn = tree_size - 1
    r = _leaf_hash(bytes.fromhex(leaf_input_hex))

    for sibling_hex in proof_path_hex:
        if sn == 0:
            print(f"  [ERROR] proof_path is longer than a tree of {tree_size} "
                  f"leaves can produce")
            return False
        sibling = bytes.fromhex(sibling_hex)
        if (fn & 1) or fn == sn:
            r = _node_hash(sibling, r)
            if not (fn & 1):
                while fn and not (fn & 1):
                    fn >>= 1
                    sn >>= 1
        else:
            r = _node_hash(r, sibling)
        fn >>= 1
        sn >>= 1

    if sn != 0:
        print(f"  [ERROR] proof_path is shorter than a tree of {tree_size} "
              f"leaves requires")
        return False
    return r == bytes.fromhex(expected_root_hex)


def _rfc6962_root(leaf_hashes: list[bytes]) -> bytes:
    """
    Rebuild an RFC 6962 §2.1 Merkle root from already-hashed leaves (each
    element here is itself the leaf's pre-image, e.g. a head-log
    entry_hash — this function applies the leaf-hash prefix itself, exactly
    as the Coriqo server does).

    Only used to independently verify a bundled head_log/head.json's
    root_hash against head_log/entries.jsonl — every other check in this
    file only ever verifies a proof PATH, never rebuilds a tree from scratch,
    because a proof path is all the API otherwise hands the verifier.
    """
    if not leaf_hashes:
        return hashlib.sha256(b"").digest()
    current = [_leaf_hash(d) for d in leaf_hashes]
    while len(current) > 1:
        nxt = [_node_hash(current[i], current[i + 1]) for i in range(0, len(current) - 1, 2)]
        if len(current) % 2:
            nxt.append(current[-1])
        current = nxt
    return current[0]


def verify_inclusion(
    leaf_input_hex: str,
    leaf_index: int,
    tree_size: int,
    proof_path_hex: list[str],
    expected_root_hex: str,
    tree_version: int = 1,
) -> bool:
    """
    Verify a Merkle inclusion proof under the declared tree construction.

    leaf_input_hex    — the event_hash hex (the raw bytes hashed into the leaf)
    leaf_index        — position of this leaf in the tree
    tree_size         — total number of leaves
    proof_path_hex    — ordered list of sibling hashes (hex), leaf → root
    expected_root_hex — the Merkle root hex from the checkpoint
    tree_version      — 1 = duplicate-last-node, 2 = RFC 6962 (SPEC.md §5.2).
                        The caller MUST source this from the SIGNED body; a
                        tree_version taken from an unsigned field would let a
                        forger pick whichever construction their hand-built
                        path satisfies.
    """
    if tree_size == 0 or leaf_index < 0 or leaf_index >= tree_size:
        return False
    if tree_version not in SUPPORTED_TREE_VERSIONS:
        print(f"  [ERROR] unknown tree_version {tree_version!r}; this verifier "
              f"implements {SUPPORTED_TREE_VERSIONS}")
        return False

    if tree_version == 2:
        return _verify_inclusion_v2(
            leaf_input_hex, leaf_index, tree_size, proof_path_hex, expected_root_hex)
    return _verify_inclusion_v1(
        leaf_input_hex, leaf_index, tree_size, proof_path_hex, expected_root_hex)


# ---------------------------------------------------------------------------
# Ed25519 signature verification (requires cryptography package)
# ---------------------------------------------------------------------------

def _canonical_bytes(obj: dict) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


ALGORITHM_ED25519 = "ed25519"
ALGORITHM_ES256 = "es256"
SUPPORTED_ALGORITHMS = (ALGORITHM_ED25519, ALGORITHM_ES256)


def verify_signature(
    sth_body: dict,
    signature_hex: str,
    public_key_pem: str,
    algorithm: str = ALGORITHM_ED25519,
) -> bool:
    """
    Verify that signature_hex is a valid signature over
    canonical_bytes(sth_body) using the given PEM public key.

    `algorithm` defaults to ed25519 — SPEC.md 6.1, where ed25519 is spelled by
    the ABSENCE of the field. A caller checking a CHECKPOINT signature must
    pass the value from that checkpoint's own signed body and nowhere else
    (Rule 6.1.1); reading it from an unsigned part of a bundle would let a
    forger select the verification path.

    The default is deliberately NOT "derive it from the key". This function is
    shared by paths that have no signed algorithm declaration of their own --
    witness cosignatures, revocation entries, the global head log -- and those
    are ed25519 by definition today. Deriving from the key would silently
    accept some other algorithm on those paths the moment such a key appeared,
    which is the confusion this field exists to prevent. When witness keys gain
    algorithm agility they must pass their own declared value explicitly.

    Rule 6.1.2: the declared algorithm and the key must AGREE. A mismatch is a
    failure, never a fallback to whatever the key happens to support.
    Rule 6.1.3: an unrecognised algorithm fails closed.
    """
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        from cryptography.exceptions import InvalidSignature

        if algorithm not in SUPPORTED_ALGORITHMS:
            print(f"  [ERROR] Unrecognised signature algorithm {algorithm!r} — "
                  f"this verifier implements {list(SUPPORTED_ALGORITHMS)}. Failing "
                  "closed rather than skipping the signature check.")
            return False

        pub = load_pem_public_key(public_key_pem.encode())
        sig = bytes.fromhex(signature_hex)
        msg = _canonical_bytes(sth_body)

        if algorithm == ALGORITHM_ED25519:
            if not isinstance(pub, Ed25519PublicKey):
                print("  [ERROR] Signed body declares ed25519 (or omits the field, "
                      "which means ed25519) but the pinned public key is not an "
                      "Ed25519 key. Refusing to verify.")
                return False
            pub.verify(sig, msg)
        else:
            if not (isinstance(pub, ec.EllipticCurvePublicKey)
                    and isinstance(pub.curve, ec.SECP256R1)):
                print("  [ERROR] Signed body declares es256 but the pinned public "
                      "key is not an EC P-256 key. Refusing to verify.")
                return False
            pub.verify(sig, msg, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False
    except Exception as exc:
        print(f"  [ERROR] Signature check failed with exception: {exc}")
        return False


# ---------------------------------------------------------------------------
# Global head log, SPEC.md §14
# ---------------------------------------------------------------------------
def _verify_tsa_offline(token_b64: str, sth_body: dict, tsa_ca_path: str | None) -> dict:
    """
    Self-contained RFC 3161 TSA token check for the offline verifier — no
    import of Coriqo code, no network. Parses the DER TimeStampResp, confirms its
    messageImprint equals SHA256(canonical sth_body), and checks the CMS
    signature. Reimplemented here (rather than importing
    the Coriqo server) because this script is meant to be runnable with
    only the stdlib plus the packages already imported at the top of this
    file (cryptography) — see the module docstring's "no Coriqo import"
    promise for --bundle-dir/--certificate. Requires pyasn1 + pyasn1_modules
    + rfc3161ng only when a bundle actually carries a tsa_token; a bundle
    with none never imports them.
    """
    import base64

    try:
        import rfc3161ng
        from pyasn1.codec.der import decoder as der_decoder, encoder as der_encoder
        from pyasn1.type import univ as pyasn1_univ
        from cryptography import x509
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, padding
    except ImportError as exc:
        return {"available": False, "detail": f"rfc3161ng not installed ({exc}) — cannot parse TSA token"}

    try:
        token_der = base64.b64decode(token_b64)
        tsr, substrate = der_decoder.decode(token_der, asn1Spec=rfc3161ng.TimeStampResp())
        if substrate:
            return {"available": False, "detail": "extra data after TimeStampResp"}
        tst = tsr.getComponentByName("timeStampToken")
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "detail": f"could not parse TSA token: {exc}"}

    expected_hash = hashlib.sha256(_canonical_bytes(sth_body)).digest()

    signed_data = tst.getComponentByName("content")
    tstinfo_octets, _ = der_decoder.decode(
        bytes(signed_data["contentInfo"]["content"]), asn1Spec=pyasn1_univ.OctetString(),
    )
    tst_info, _ = der_decoder.decode(bytes(tstinfo_octets), asn1Spec=rfc3161ng.TSTInfo())
    imprint = tst_info.getComponentByName("messageImprint")
    imprint_matches = (
        imprint.getComponentByName("hashAlgorithm").getComponentByPosition(0) == rfc3161ng.id_sha256
        and bytes(imprint.getComponentByName("hashedMessage")) == expected_hash
    )
    gen_time_iso = None
    try:
        gen_time = rfc3161ng.get_timestamp(tst, naive=False)
        if gen_time.tzinfo is None:
            gen_time = gen_time.replace(tzinfo=timezone.utc)
        gen_time_iso = gen_time.astimezone(timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        pass

    # CMS signature. This is hand-rolled rather than rfc3161ng.check_timestamp because that
    # helper hard-codes RSA/PKCS1v15 and raises a bare TypeError against a
    # real EC-signed TSA response (observed against FreeTSA in practice).
    signature_verified = False
    sig_detail = "no signerInfo"
    ca_verified = None
    try:
        signer_info = signed_data["signerInfos"][0]
        cert_der = der_encoder.encode(signed_data["certificates"][0][0])
        cert = x509.load_der_x509_certificate(cert_der, default_backend())
        content = bytes(der_decoder.decode(
            bytes(signed_data["contentInfo"]["content"]), asn1Spec=pyasn1_univ.OctetString(),
        )[0])
        attrs = signer_info["authenticatedAttributes"]
        if len(attrs):
            s = pyasn1_univ.SetOf()
            for i, a in enumerate(attrs):
                s.setComponentByPosition(i, a)
            signed_bytes = der_encoder.encode(s)
        else:
            signed_bytes = content
        signature = bytes(signer_info["encryptedDigest"])
        digest_oid = signer_info["digestAlgorithm"]["algorithm"]
        signer_hash_name = rfc3161ng.oid_to_hash.get(digest_oid, "sha256")
        signer_hash = getattr(hashes, signer_hash_name.upper())()
        public_key = cert.public_key()
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, signed_bytes, ec.ECDSA(signer_hash))
        else:
            public_key.verify(signature, signed_bytes, padding.PKCS1v15(), signer_hash)
        signature_verified = True
        sig_detail = ""

        if tsa_ca_path:
            with open(tsa_ca_path, "rb") as fh:
                ca_pem_bytes = fh.read()
            ca_certs = []
            for block in ca_pem_bytes.split(b"-----END CERTIFICATE-----"):
                if b"-----BEGIN CERTIFICATE-----" not in block:
                    continue
                ca_certs.append(x509.load_pem_x509_certificate(
                    block + b"-----END CERTIFICATE-----\n", default_backend()))
            ca_verified = False
            for ca in ca_certs:
                if cert.issuer != ca.subject:
                    continue
                try:
                    ca.public_key().verify(
                        cert.signature, cert.tbs_certificate_bytes,
                        padding.PKCS1v15(), cert.signature_hash_algorithm or hashes.SHA256(),
                    )
                    ca_verified = True
                    break
                except Exception:  # noqa: BLE001
                    continue
    except InvalidSignature:
        sig_detail = "CMS signature does not verify"
    except Exception as exc:  # noqa: BLE001
        sig_detail = f"signature check failed: {exc}"

    return {
        "available": True,
        "imprint_matches": imprint_matches,
        "signature_verified": signature_verified,
        "ca_verified": ca_verified,
        "gen_time": gen_time_iso,
        "detail": sig_detail,
    }


