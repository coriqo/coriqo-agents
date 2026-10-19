"""Uploads a manifest + its results to Coriqo, device-signed.

Deliberately separate from CoriqoAgentClient (../client.py): that class
authenticates with X-API-Key/X-Tenant-Slug and explicitly does not sign
requests ("This SDK does not sign; byoai-runtime does" — see its comment
above `_runs_path`). The runner is the thing that signs, so its own thin
poster lives here rather than bolting signing onto a client built around a
different credential.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .signing import DeviceSigner


@dataclass
class ExecutionUploader:
    base_url: str
    signer: DeviceSigner
    tenant_slug: str
    http_client: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        return self.http_client or httpx.Client(base_url=self.base_url.rstrip("/"))

    def _post(self, path: str, body: dict[str, Any]) -> dict:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers = self.signer.headers(method="POST", path=path, body=raw)
        headers["X-Tenant-Slug"] = self.tenant_slug
        client = self._client()
        try:
            resp = client.post(path, content=raw, headers=headers)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not reach Coriqo to upload {path}: {exc}") from exc
        finally:
            if self.http_client is None:
                client.close()
        if resp.status_code >= 400:
            raise RuntimeError(f"Coriqo rejected the upload ({resp.status_code}): {resp.text}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Coriqo accepted the upload to {path} ({resp.status_code}) but returned a "
                f"non-JSON response: {exc}"
            ) from exc

    def _get(self, path: str) -> dict:
        headers = self.signer.headers(method="GET", path=path, body=b"")
        headers["X-Tenant-Slug"] = self.tenant_slug
        client = self._client()
        try:
            resp = client.get(path, headers=headers)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not reach Coriqo to read {path}: {exc}") from exc
        finally:
            if self.http_client is None:
                client.close()
        if resp.status_code >= 400:
            raise RuntimeError(f"Coriqo refused the request ({resp.status_code}): {resp.text}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Coriqo answered {path} with a non-JSON response: {exc}") from exc

    def get_trace_policy(self, execution_id: str) -> dict:
        """Whether this workspace keeps traces: {"enabled", "days", "max_mb", ...}."""
        return self._get(f"/api/v1/executions/{execution_id}/trace-policy")

    def upload_trace(self, execution_id: str, trace: dict) -> dict:
        """Send the trace the validators judged, for workspaces that keep
        traces. The server accepts it only if its hash is one the signed
        manifest committed to."""
        return self._post(f"/api/v1/executions/{execution_id}/trace", trace)

    def upload_monitoring_windows(self, execution_id: str, windows: list[dict]) -> dict:
        """A batch of evaluation windows for a standing execution.

        Counters and hashed failure digests only — never a payload, never a
        raw trace id (see `monitoring._failure_digest`). Safe to retry: the
        server keys windows on a content-addressed id it recomputes itself,
        so a replayed window is reported as a duplicate rather than counted
        twice."""
        return self._post(
            f"/api/v1/executions/{execution_id}/monitoring/windows",
            {"windows": windows},
        )

    def upload_manifest(self, execution_id: str, manifest: dict) -> dict:
        return self._post(f"/api/v1/executions/{execution_id}/manifest", manifest)

    def upload_results(
        self, execution_id: str, results: list[dict], evidence: dict[str, dict] | None = None,
        failure_scenarios: list[dict] | None = None,
    ) -> dict:
        return self._post(
            f"/api/v1/executions/{execution_id}/results",
            {"results": results, "evidence": evidence or {}, "failure_scenarios": failure_scenarios or []},
        )
