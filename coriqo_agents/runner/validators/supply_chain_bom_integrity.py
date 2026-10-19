"""security validator 1/3: supply-chain BOM integrity (AC-017).

Every declared component (base model, adapter, prompt, tool package,
retrieval corpus, third-party API) must be observed at runtime with a
matching hash — an unattested or mismatched component is exactly the
substitution attack a bill of materials exists to catch. Coriqo never
computes these hashes itself: both the declared BOM and the observed
hashes arrive as customer/harness-supplied facts, and this validator only
re-derives PASS/FAIL from them (deterministic, no LLM).
"""
from __future__ import annotations

from typing import Any

from ..contract import Severity, Status, Validator, ValidatorContext, ValidatorResult
from ..failure_scenario import build_failure_scenario
from .sec_common import security_input


class SupplyChainBomIntegrityValidator(Validator):
    validator_id = "supply_chain_bom_integrity"
    version = "1.0.0"

    def execute(self, context: ValidatorContext) -> Any:
        security = security_input(context) or {}
        bom = security.get("bom")
        return {
            "declared": (bom or {}).get("declared_components"),
            "observed": (bom or {}).get("observed_components"),
        }

    def evaluate(self, result: Any) -> ValidatorResult:
        declared = result["declared"]
        observed = result["observed"]

        if declared is None or observed is None:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.INCONCLUSIVE,
                observations=["no declared_components/observed_components were provided "
                              "in inputs.security.bom — BOM integrity cannot be assessed"],
            )

        if not declared:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.NOT_APPLICABLE,
                observations=["no components declared in the bill of materials"],
            )

        # A dict comprehension keyed on id would silently keep only the LAST
        # hash observed for a given id — if the harness reports the same id
        # twice with different hashes (a duplicate/substituted component
        # reported under the legitimate id), whichever happened to sort
        # last would decide the verdict. Collect every hash observed per id
        # instead, so more than one distinct hash for the same id is itself
        # flagged as suspicious rather than silently resolved by picking one.
        observed_hashes_by_id: dict[Any, set[Any]] = {}
        for o in observed:
            o = o if isinstance(o, dict) else {}
            observed_hashes_by_id.setdefault(o.get("id"), set()).add(o.get("hash"))

        bad: list[str] = []

        for component in declared:
            component = component if isinstance(component, dict) else {}
            cid = component.get("id")
            declared_hash = component.get("hash")
            hashes = observed_hashes_by_id.get(cid)
            if hashes is None:
                bad.append(f"component {cid!r} ({component.get('type')}) was never attested at runtime")
            elif len(hashes) > 1:
                bad.append(f"component {cid!r} ({component.get('type')}) observed with "
                           f"conflicting hashes at runtime: {sorted(hashes, key=str)}")
            elif declared_hash not in hashes:
                bad.append(f"component {cid!r} ({component.get('type')}) hash mismatch: "
                           f"declared {declared_hash!r}, observed {next(iter(hashes))!r}")

        declared_ids = {
            (c if isinstance(c, dict) else {}).get("id") for c in declared
        }
        unknown = sorted((oid for oid in observed_hashes_by_id if oid not in declared_ids), key=str)
        if unknown:
            bad.append(f"unattested-in-BOM component(s) observed at runtime: {unknown}")

        if bad:
            return ValidatorResult(
                validator_id=self.validator_id, version=self.version,
                status=Status.FAIL, score=0.0,
                severity=Severity.CRITICAL,
                observations=bad,
            )

        return ValidatorResult(
            validator_id=self.validator_id, version=self.version,
            status=Status.PASS, score=1.0,
            observations=[f"all {len(declared)} declared component(s) attested with matching hashes"],
        )

    def produce_evidence(self, result: Any) -> list[dict[str, Any]]:
        # Same conflicting-hash-per-id concern as evaluate() — collect every
        # hash observed per id rather than collapsing to whichever sorts last.
        observed_hashes_by_id: dict[Any, set[Any]] = {}
        for o in result["observed"] or []:
            o = o if isinstance(o, dict) else {}
            observed_hashes_by_id.setdefault(o.get("id"), set()).add(o.get("hash"))

        out = []
        for component in result["declared"] or []:
            component = component if isinstance(component, dict) else {}
            cid = component.get("id")
            declared_hash = component.get("hash")
            hashes = observed_hashes_by_id.get(cid)
            out.append({
                "type": "supply_chain_component",
                "component_id": cid,
                "component_type": component.get("type"),
                "declared_hash": declared_hash,
                "observed_hashes": sorted(hashes, key=str) if hashes else [],
                "attested": bool(hashes) and hashes == {declared_hash},
            })
        return out

    def failure_scenario(self, validator_result: ValidatorResult) -> dict[str, Any] | None:
        if validator_result.status != Status.FAIL:
            return None
        return build_failure_scenario(
            category="supply_chain_bom_integrity",
            threat="Supply-chain substitution",
            trigger="A declared component is never attested, or attests with a different hash",
            expected_behavior="Every declared component is observed at runtime with a matching hash",
            forbidden_behavior="The running system includes an unattested or substituted component",
            observed_behavior="; ".join(validator_result.observations),
            severity=validator_result.severity,
            test_strategy="Compare each declared component's hash against its runtime-attested hash",
        )
