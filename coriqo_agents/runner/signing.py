"""Ed25519 device signing for runner uploads — the SAME wire scheme
api.domains.agents.device_auth already verifies enforcement requests with
(the {body_sha256, method, path, public_key, timestamp} tuple), not a
second scheme invented for this endpoint.

This module cannot import that server module directly (the runner ships
standalone into customer environments and must not depend on the API's
internal package layout), so `signing_payload` below reproduces its
documented format byte for byte. The two MUST stay identical — see
device_auth.py's own docstring for the authoritative definition this
mirrors. A drift here fails every runner upload with an opaque "signature
did not verify," so treat any change to that tuple as a breaking change to
this file too.

Requires the `runner` extra (`pip install coriqo-agents[runner]`) —
`cryptography` is not a base dependency of this SDK (see pyproject.toml).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _canonical_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def signing_payload(*, method: str, path: str, body: bytes, timestamp: str, public_key: str) -> bytes:
    """Mirrors api.domains.agents.device_auth.signing_payload exactly."""
    return _canonical_bytes({
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "method": method.upper(),
        "path": path,
        "public_key": public_key.lower(),
        "timestamp": timestamp,
    })


@dataclass
class DeviceSigner:
    """Wraps an enrolled device's Ed25519 private key and produces the four
    headers current_device expects: X-Coriqo-Public-Key, X-Coriqo-Signature,
    X-Coriqo-Timestamp, plus Content-Type. Construct with the key bytes this
    host received at enrollment (POST /api/v1/agent-runtime/devices) — this
    class does not enroll a device itself, it only signs with a key that
    already is one.
    """
    private_key: Any  # cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey

    @classmethod
    def generate(cls) -> "DeviceSigner":
        """For local development / tests only. A real deployment loads the
        key issued at enrollment; it does not mint its own."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return cls(private_key=Ed25519PrivateKey.generate())

    @property
    def public_key_hex(self) -> str:
        return self.private_key.public_key().public_bytes_raw().hex()

    def headers(self, *, method: str, path: str, body: bytes) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = signing_payload(
            method=method, path=path, body=body, timestamp=timestamp, public_key=self.public_key_hex,
        )
        signature = self.private_key.sign(payload).hex()
        return {
            "X-Coriqo-Public-Key": self.public_key_hex,
            "X-Coriqo-Signature": signature,
            "X-Coriqo-Timestamp": timestamp,
            "Content-Type": "application/json",
        }
