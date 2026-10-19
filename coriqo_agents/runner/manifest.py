"""Canonical hashing + the signed execution manifest.

`canonical_bytes` mirrors api.chain.canonical_bytes byte for byte (sorted
keys, compact separators, UTF-8) — this package cannot import server code
(it ships standalone to customer environments), so the format is
reimplemented here rather than shared, but it MUST stay identical: a runner
manifest hashed one way and re-verified another way is exactly the class of
bug canonical signing exists to prevent. If api.chain.canonical_bytes ever
changes, this function changes with it.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .contract import ValidatorResult


def canonical_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_value(obj: Any) -> str:
    """sha256 hex over an arbitrary JSON-serializable value's canonical
    bytes — used for `configuration_snapshot_hash` and
    `input_dataset_hashes`. Coriqo never receives the raw configuration or
    input datasets, only these hashes, matching the SDK's existing
    never-send-raw-payloads convention (see ../hashing.py's hash_value,
    which this deliberately mirrors rather than imports, since a customer
    may install the `runner` extra without the base client's dependency
    surface some day)."""
    return sha256_hex(canonical_bytes({"value": obj}))


def build_evidence_envelope(execution_id: str, validator_id: str, raw_item: dict[str, Any]) -> dict[str, str]:
    """Wrap one produce_evidence() item into the closed, four-field shape
    the server's evidence_records substrate can validate without knowing
    every validator's own evidence shape ahead of time (that substrate
    rejects unknown keys at every level, so a raw validator-specific dict with
    arbitrary fields cannot be validated directly). `raw_item` itself is
    preserved byte-for-byte inside `raw_json`, not restructured, so nothing
    the validator produced is lost.

    `execution_id` is part of the envelope, not just caller context, because
    `evidence_content_id` hashes the envelope to get the id — without
    execution_id, two DIFFERENT executions whose validators observe
    byte-identical trace events (the same tool call, same payload — not a
    hypothetical: two runs against a replayed/fixture trace, or two clean
    runs of the same protocol against the same agent snapshot) would derive
    the SAME content-addressed id and collide on EvidenceRecord's primary
    key at insert. Caught live by two tests in this SDK's own pytest suite
    sharing one trace fixture."""
    return {
        "execution_id": execution_id,
        "validator_id": validator_id,
        "kind": str(raw_item.get("type", "unknown")),
        "raw_json": json.dumps(raw_item, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    }


#: `kind` on the evidence envelope carrying an engine's own output. The
#: server's evidence record type does not enumerate kinds, so this string is
#: the only thing that marks an envelope as a raw engine artifact — keep it
#: in step with Coriqo's API, which reads it to
#: set `assurance_executions.raw_artifact_evidence_id`.
RAW_ENGINE_ARTIFACT_KIND = "raw_engine_artifact"

#: Above this, the artifact's bytes are not uploaded — only its hash, its
#: length and a truncation flag. A Promptfoo run over a few hundred test
#: cases produces a results file in the low megabytes; one over a large
#: generated corpus does not, and an evidence row is not a blob store. The
#: hash is over the FULL bytes either way, so the manifest's claim stays
#: exact and a reviewer can still verify a copy of the file held customer
#: side against it.
MAX_INLINE_ARTIFACT_BYTES = 2 * 1024 * 1024


def raw_engine_artifact_item(
    artifact: bytes,
    *,
    engine_id: str,
    engine_version: str,
    plan_hash: str,
    max_inline_bytes: int = MAX_INLINE_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """The evidence payload for an engine's verbatim output.

    Carries the hash, the byte length and the engine/plan identity always,
    and the content itself only when it fits under `max_inline_bytes`.
    `truncated` says which of those happened, so "the file was too big to
    ship" never reads as "the tool produced nothing".

    Bytes that are not valid UTF-8 are base64'd rather than dropped or
    lossily decoded — `encoding` says which, and the hash is over the
    original bytes in both cases, so what a reviewer re-hashes is what the
    tool wrote.
    """
    digest = sha256_hex(artifact)
    item: dict[str, Any] = {
        "type": RAW_ENGINE_ARTIFACT_KIND,
        "engine": engine_id,
        "engine_version": engine_version,
        "plan_hash": plan_hash,
        "sha256": digest,
        "byte_length": len(artifact),
    }
    if len(artifact) > max_inline_bytes:
        item.update({"truncated": True, "encoding": None, "content": None})
        return item
    try:
        item.update({"truncated": False, "encoding": "utf-8", "content": artifact.decode("utf-8")})
    except UnicodeDecodeError:
        item.update({
            "truncated": False,
            "encoding": "base64",
            "content": base64.b64encode(artifact).decode("ascii"),
        })
    return item


def evidence_content_id(envelope: dict[str, str]) -> str:
    """Content-addressed evidence id: both this package and
    api.domains.assurance.manifest_service compute the SAME id from the
    SAME envelope bytes, so a result's evidence_ids can be embedded in the
    manifest's per-result hash before any network round trip ever
    happens — there is no server-assigned id to wait for. The server then
    independently re-derives this id from the uploaded payload rather than
    trusting whatever string the runner claims; a payload that doesn't hash
    back to its own claimed id is a disclosed mismatch, not a silently
    trusted assertion. Keep in lockstep with manifest_service.py's function
    of the same name, same reasoning as canonical_result_hash above."""
    return "ev_" + sha256_hex(canonical_bytes(envelope))[:32]


def canonical_result_hash(result: ValidatorResult | dict[str, Any]) -> str:
    """The exact field set + order api.domains.assurance.manifest_service.
    _canonical_result_hash recomputes server-side. Both sides MUST hash the
    same six fields — validator_id, version, status, score, severity,
    observations, evidence_ids — or every manifest fails verification
    regardless of whether the run was legitimate. This is the one place
    that contract lives on the runner side; keep it in lockstep with the
    server function of the same name."""
    d = result.to_dict() if isinstance(result, ValidatorResult) else result
    fields = {
        "validator_id": d.get("validator_id"),
        "version": d.get("version"),
        "status": d.get("status"),
        "score": d.get("score"),
        "severity": d.get("severity"),
        "observations": d.get("observations") or [],
        "evidence_ids": d.get("evidence_ids") or [],
    }
    return sha256_hex(canonical_bytes(fields))


def build_manifest(
    *,
    runner_version: str,
    protocol_id: str,
    protocol_version: str,
    bundle_hash: str,
    validators: list[dict[str, str]],
    system_version_ref: str | None,
    configuration_snapshot: dict[str, Any] | None,
    input_datasets: dict[str, Any] | None,
    results: list[ValidatorResult],
    started_at: datetime,
    completed_at: datetime,
    trace_provenance: dict[str, Any] | None = None,
    engine: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the signed execution manifest:

        runner version, validator ids+versions+bundle hash, protocol
        id+version, system version ref, configuration snapshot hash, input
        dataset hashes, per-result hashes, timestamps

    `validators` is [{"id": ..., "version": ...}, ...] — the validators
    actually run, which the server checks against the compiled bundle's own
    validator list. `results` must be in the exact order they will be
    uploaded to POST /executions/{id}/results — `result_hashes` here is
    positional, and the server re-verifies index-for-index.

    `trace_provenance` is a deliberate exception to `input_dataset_hashes`'
    never-send-raw rule: `RunnerEngine.run()` derives it (`{"source": ...,
    "event_count": ...}`) from `context.inputs["trace"]` when that trace
    carries `otel_ingest.py`'s `"source"` marker, and it is embedded here
    verbatim, not hashed — the whole point is for a human reviewing the
    execution to see, without re-deriving anything, whether it evaluated a
    real OTel export or a hand-built trace. No field discloses any trace
    CONTENT (no payload text, no tool names) — only that a trace existed,
    where it came from, how many events it had, and, when events carry
    timestamps, `first_event_at`, `last_event_at` and `max_gap_seconds` (the
    longest silence between two events) as completeness signals. Those three
    are omitted when no event has a readable time, and a manifest without them
    verifies exactly as before.

    `engine` is the same kind of exception, added for third-party
    engine adapters: `{"id", "version", "plan_hash", "raw_artifact_sha256"}`,
    embedded verbatim. `None` for the native validator path, which is what
    `RunnerEngine.run()` still passes — an execution that ran before any
    engine adapter existed was produced by no recorded engine, and writing
    `native` onto it after the fact would be provenance nobody observed (the
    same reasoning migration 0276 gives for not backfilling the column). The
    two hashes are what make the block checkable: `plan_hash` covers the
    configuration the adapter wrote, `raw_artifact_sha256` covers the exact
    bytes the tool emitted, which are uploaded separately as evidence. No
    part of it may carry a credential — the plan hashes an environment
    variable's NAME, never its value.

    `capabilities` is `capabilities.report()` — what
    this HOST can run, not what this run ran. Embedded verbatim for the same
    reason `trace_provenance` is: the campaign builder has to grey out an
    engine nobody's device can execute and say WHY, and a hash of that fact
    would tell it nothing. `None` for a caller that does not collect it, so
    a manifest built the old way is unchanged.
    """
    for ts in (started_at, completed_at):
        if ts.tzinfo is None:
            raise ValueError("started_at/completed_at must be timezone-aware")

    return {
        "runner_version": runner_version,
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "bundle_hash": bundle_hash,
        "validators": validators,
        "system_version_ref": system_version_ref,
        # `is not None`, not truthy: ValidatorContext.configuration
        # defaults to `{}` (an explicit, present-but-empty configuration),
        # which a truthy check hashed identically to "no configuration
        # snapshot was ever captured" (None) — indistinguishable in a
        # signed record whose whole purpose is to make that distinction.
        "configuration_snapshot_hash": (
            hash_value(configuration_snapshot) if configuration_snapshot is not None else None
        ),
        "input_dataset_hashes": (
            [hash_value(v) for v in input_datasets.values()] if input_datasets else []
        ),
        "result_hashes": [canonical_result_hash(r) for r in results],
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "trace_provenance": trace_provenance,
        "engine": engine,
        "capabilities": capabilities,
    }
