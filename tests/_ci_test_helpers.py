"""Shared fake-httpx-transport plumbing for `coriqo_agents.runner.ci` tests.

`test_ci_no_secrets.py`, `test_ci_gate_availability.py`, and
`test_ci_error_paths.py` each hand-rolled their own near-identical
`_FakeResponse`, `_Monkeypatch`, `_install_fake_transport`, `_write_device_key`,
`_bundle`, and `_run_ci` helpers (flagged in review:
a future change to ci.py's httpx-facing shape had to be mirrored in three
places instead of one). This module is the single copy; the three files
above import from it instead of redefining their own.

`test_ci_gate_logic.py` calls `_gate()` directly with hand-built dicts and
never touches httpx, so it has nothing to share here.

Every one of these test files runs standalone (`python test_ci_*.py`), the
same convention as every other pure-stdlib SDK test in this directory, so
this module stays pure-stdlib-plus-`cryptography` too — no pytest import.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    HAVE_CRYPTOGRAPHY = True
except ImportError:
    HAVE_CRYPTOGRAPHY = False


def write_device_key(path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)


def bundle() -> dict:
    return {
        "id": "proto_1", "version": "1.0", "bundle_hash": "sha256:deadbeef",
        "template": {"protocol": {"validators": [{"id": "tool_authorization", "version": "1.0.0"}]}},
    }


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, *, raw_text: str | None = None):
        self.status_code = status_code
        self._body = body
        self._raw_text = raw_text

    def json(self) -> dict:
        if self._raw_text is not None:
            # Mirrors what httpx does on a non-JSON body: json.loads raises
            # json.JSONDecodeError, a subclass of ValueError.
            return json.loads(self._raw_text)
        return self._body

    @property
    def text(self) -> str:
        return self._raw_text if self._raw_text is not None else json.dumps(self._body)


class Monkeypatch:
    """A tiny stand-in for pytest's `monkeypatch` fixture — these suites run
    standalone (`python test_ci_*.py`), so they can't depend on pytest being
    the runner."""

    def __init__(self):
        self._restores: list[tuple[object, str, object]] = []

    def setattr(self, obj, name, value):
        self._restores.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._restores):
            setattr(obj, name, value)


def install_fake_transport(
    monkeypatch: Monkeypatch,
    ci_module,
    *,
    execution_id: str = "ex_ci_1",
    poll_behavior: Callable[[str], FakeResponse] | None = None,
    diff_behavior: Callable[[str], FakeResponse] | None = None,
    trace_policy: dict | None = None,
) -> list[tuple[str, str]]:
    """Stubs every network call `ci_module.main()` makes: campaign lookup,
    execution create, completion poll, diff fetch, and the device-signed
    upload. `poll_behavior`/`diff_behavior` default to canned success so a
    test overrides only the one call it's exercising; every call is recorded
    into the returned list in order, for tests that want to assert on it."""
    calls: list[tuple[str, str]] = []

    def default_poll(url):
        return FakeResponse(200, {"id": execution_id, "status": "completed"})

    def default_diff(url):
        return FakeResponse(200, {
            "against_execution_id": "ex_prev", "counts": {"regressed": 0},
            "control_determinations": [], "validators": [],
        })

    poll = poll_behavior or default_poll
    diff = diff_behavior or default_diff

    def fake_get(url, headers=None, timeout=None):
        calls.append(("GET", url))
        if url.endswith("/campaigns/c1"):
            return FakeResponse(200, {
                "id": "c1", "status": "approved", "agent_model_id": "m1", "agent_version_ref": "v1",
            })
        if url.endswith(f"/assurance/{execution_id}"):
            return poll(url)
        if url.endswith(f"/{execution_id}/diff"):
            return diff(url)
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(("POST", url))
        if url.endswith("/campaigns/c1/executions"):
            return FakeResponse(201, {"execution": {"id": execution_id}, "plan": {}})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(ci_module.httpx, "get", fake_get)
    monkeypatch.setattr(ci_module.httpx, "post", fake_post)

    from coriqo_agents.runner import uploader as uploader_module

    class _FakeClient:
        def __init__(self, base_url=""):
            self.base_url = base_url

        def post(self, path, content=None, headers=None):
            calls.append(("UPLOAD", path))
            return FakeResponse(201, {"ok": True})

        def get(self, path, headers=None):
            calls.append(("DEVICE_GET", path))
            if path.endswith("/trace-policy"):
                return FakeResponse(200, trace_policy or {"enabled": False, "days": 400, "max_mb": 25})
            raise AssertionError(f"unexpected device GET {path}")

        def close(self):
            pass

    monkeypatch.setattr(uploader_module.httpx, "Client", _FakeClient)
    return calls


def run_ci(
    ci_module,
    *,
    poll_behavior: Callable[[str], FakeResponse] | None = None,
    diff_behavior: Callable[[str], FakeResponse] | None = None,
    execution_id: str = "ex_ci_1",
    extra_args: list[str] | None = None,
    junit_path: Path | None = None,
    trace_policy: dict | None = None,
) -> tuple[int, str, list[tuple[str, str]]]:
    """Runs `ci_module.main()` end to end against the fake transport above,
    inside a fresh temp dir holding a bundle/trace/device-key. Returns the
    exit code, combined stdout+stderr, and the ordered call log."""
    mp = Monkeypatch()
    try:
        calls = install_fake_transport(
            mp, ci_module, execution_id=execution_id, poll_behavior=poll_behavior, diff_behavior=diff_behavior,
            trace_policy=trace_policy,
        )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            bundle_path = d / "bundle.json"
            bundle_path.write_text(json.dumps(bundle()), encoding="utf-8")
            trace_path = d / "trace.json"
            trace_path.write_text(json.dumps({"resourceSpans": []}), encoding="utf-8")
            device_key_path = d / "device.pem"
            write_device_key(device_key_path)

            argv = [
                "--campaign", "c1", "--agent-version", "v1",
                "--bundle", str(bundle_path), "--trace", str(trace_path),
                "--base-url", "https://coriqo.example", "--tenant", "acme_bank",
                "--api-key", "sk-fake", "--device-key", str(device_key_path),
            ]
            if junit_path is not None:
                argv += ["--junit", str(junit_path)]
            argv += extra_args or []

            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = ci_module.main(argv)
            return exit_code, stdout.getvalue() + stderr.getvalue(), calls
    finally:
        mp.undo()


def run_tests(namespace: dict[str, Any]) -> int:
    """Shared `if __name__ == "__main__":` runner body — every standalone
    test file in this directory does the same "collect test_* callables, run
    them, print a tally" dance."""
    failures = 0
    tests = [obj for name, obj in list(namespace.items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  ✗ {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0
