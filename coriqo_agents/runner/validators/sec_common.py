"""security assurance slice: shared input shape and helpers.

Every security validator reads `context.inputs["security"]`, facts the
harness/customer supplies about the system under test — never a verdict
the harness computed itself:

    {
      "bom": {                                # supply_chain_bom_integrity
        "declared_components": [
          {"id": str, "type": "base_model" | "adapter" | "prompt" |
                              "tool_package" | "corpus" | "third_party_api",
           "version": str, "hash": str},
        ],
        "observed_components": [{"id": str, "hash": str}],
      },
      "extraction": {                         # extraction_query_pattern
        "queries": [{"principal_id": str, "query_count": int,
                     "input_space_coverage": float}],   # 0.0-1.0
        "max_queries_per_principal": int,
        "max_input_space_coverage": float,               # 0.0-1.0
      },
      "poisoning": {                          # training_data_provenance
        "datasets": [{"id": str, "signed_source_record": bool}],
      },
    }

A missing top-level key means "not supplied" (INCONCLUSIVE — cannot
assess that check), distinct from a supplied-but-empty collection
(NOT_APPLICABLE — nothing to check), the same split every other MODEL/RAG/GenAI/security
validator draws.
"""
from __future__ import annotations

from typing import Any


def security_input(context) -> dict[str, Any] | None:
    """The `inputs["security"]` dict, or None when the run carried none."""
    security = (context.inputs or {}).get("security")
    return security if isinstance(security, dict) else None
