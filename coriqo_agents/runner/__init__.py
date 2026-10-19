"""The Coriqo assurance runner — the customer-side execution engine for
assurance protocols. Ships on the same byoai-runtime / coriqo-agents SDK
channel that already carries telemetry collection, because telemetry
collection and validator execution ship together in the customer
environment.

This package holds the validator contract, the execution engine that runs
a compiled protocol bundle's validators and builds the signed manifest, and
the Ed25519 signing helper that reuses the SAME device-key mechanism the
rest of byoai-runtime signs with, not a second scheme. The real validators
live under `.validators`; `stub_validators.py` stays in the tree only as a labeled example of the contract's minimal shape; nothing in this
package resolves "tool_authorization" to the stub anymore.

Nothing in this package imports an LLM client or calls out to one: the runner must ship with zero LLM dependencies so a customer never
needs to hand it API keys. Its only extra dependency (see this SDK's
pyproject.toml `runner` extra) is `cryptography`, for Ed25519 signing.
"""
from .contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from .engine import RunnerEngine
from .manifest import build_manifest, canonical_result_hash
from .otel_ingest import (
    MultipleAgentsInBatchError,
    otlp_json_bytes_to_trace,
    otlp_json_bytes_to_traces,
    otlp_json_to_trace,
    otlp_json_to_traces,
)
from .signing import DeviceSigner
from .uploader import ExecutionUploader
from .validators import VALIDATOR_REGISTRY, resolve_validators

__all__ = [
    "Severity",
    "Status",
    "Validator",
    "ValidatorContext",
    "ValidatorResult",
    "RunnerEngine",
    "build_manifest",
    "canonical_result_hash",
    "otlp_json_to_trace",
    "otlp_json_bytes_to_trace",
    "otlp_json_to_traces",
    "otlp_json_bytes_to_traces",
    "MultipleAgentsInBatchError",
    "DeviceSigner",
    "ExecutionUploader",
    "VALIDATOR_REGISTRY",
    "resolve_validators",
]
