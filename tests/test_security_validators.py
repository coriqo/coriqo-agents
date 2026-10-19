"""The three security validators.

Pure stdlib `unittest`, same convention as test_model_validators.py.
Runnable standalone (`python test_security_validators.py`) and under
pytest.

The class of bug this file exists to pin: a security input that was
never supplied must never read as clean — a missing report reads
INCONCLUSIVE, a supplied-but-empty collection reads NOT_APPLICABLE —
plus the BOM hash-comparison, the extraction coverage/volume split
(coverage breach FAILs, volume breach only WARNs), and the provenance
unsigned-dataset FAIL.
"""
from __future__ import annotations

import unittest

from coriqo_agents.runner.contract import Status, ValidatorContext
from coriqo_agents.runner.validators import VALIDATOR_REGISTRY
from coriqo_agents.runner.validators.extraction_query_pattern import (
    ExtractionQueryPatternValidator,
)
from coriqo_agents.runner.validators.supply_chain_bom_integrity import (
    SupplyChainBomIntegrityValidator,
)
from coriqo_agents.runner.validators.training_data_provenance import (
    TrainingDataProvenanceValidator,
)

SECURITY_VALIDATOR_IDS = (
    "supply_chain_bom_integrity",
    "extraction_query_pattern",
    "training_data_provenance",
)


def ctx(security=None, **kw):
    return ValidatorContext(
        execution_id="e1", agent_model_id="m1", agent_version_ref=None,
        environment="STAGING", configuration=kw.get("configuration", {}),
        inputs={"security": security} if security is not None else {},
    )


def run(cls, security=None, **kw):
    v = cls()
    return v.evaluate(v.execute(ctx(security, **kw)))


class RegistryTest(unittest.TestCase):
    def test_all_three_registered(self):
        for vid in SECURITY_VALIDATOR_IDS:
            self.assertIn(vid, VALIDATOR_REGISTRY)


class BomIntegrityTest(unittest.TestCase):
    def test_all_attested_with_matching_hash_passes(self):
        r = run(SupplyChainBomIntegrityValidator, {
            "bom": {
                "declared_components": [{"id": "base", "type": "base_model", "hash": "h1"}],
                "observed_components": [{"id": "base", "hash": "h1"}],
            },
        })
        self.assertEqual(r.status, Status.PASS)

    def test_never_attested_fails_critical(self):
        r = run(SupplyChainBomIntegrityValidator, {
            "bom": {
                "declared_components": [{"id": "base", "type": "base_model", "hash": "h1"}],
                "observed_components": [],
            },
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "critical")
        self.assertIn("never attested", r.observations[0])

    def test_hash_mismatch_fails(self):
        r = run(SupplyChainBomIntegrityValidator, {
            "bom": {
                "declared_components": [{"id": "base", "type": "base_model", "hash": "h1"}],
                "observed_components": [{"id": "base", "hash": "h2"}],
            },
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertIn("mismatch", r.observations[0])

    def test_unknown_observed_component_fails(self):
        r = run(SupplyChainBomIntegrityValidator, {
            "bom": {
                "declared_components": [{"id": "base", "type": "base_model", "hash": "h1"}],
                "observed_components": [{"id": "base", "hash": "h1"}, {"id": "sneaky", "hash": "h9"}],
            },
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertIn("sneaky", r.observations[0])

    def test_missing_bom_is_inconclusive(self):
        r = run(SupplyChainBomIntegrityValidator, {})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_empty_declared_is_not_applicable(self):
        r = run(SupplyChainBomIntegrityValidator, {
            "bom": {"declared_components": [], "observed_components": []},
        })
        self.assertEqual(r.status, Status.NOT_APPLICABLE)

    def test_conflicting_observed_hashes_for_same_id_fails(self):
        # Two observed entries under the same id with DIFFERENT hashes
        # must never be silently resolved by picking one (e.g. whichever
        # happens to match) — a duplicate/substituted component reported
        # under the legitimate id is itself suspicious.
        r = run(SupplyChainBomIntegrityValidator, {
            "bom": {
                "declared_components": [{"id": "base", "type": "base_model", "hash": "h1"}],
                "observed_components": [{"id": "base", "hash": "h1"}, {"id": "base", "hash": "h2"}],
            },
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertIn("conflicting hashes", r.observations[0])


class ExtractionQueryPatternTest(unittest.TestCase):
    def test_within_caps_passes(self):
        r = run(ExtractionQueryPatternValidator, {
            "extraction": {
                "queries": [{"principal_id": "p1", "query_count": 10, "input_space_coverage": 0.1}],
                "max_queries_per_principal": 100,
                "max_input_space_coverage": 0.5,
            },
        })
        self.assertEqual(r.status, Status.PASS)

    def test_coverage_breach_fails_critical(self):
        r = run(ExtractionQueryPatternValidator, {
            "extraction": {
                "queries": [{"principal_id": "p1", "query_count": 10, "input_space_coverage": 0.9}],
                "max_queries_per_principal": 100,
                "max_input_space_coverage": 0.5,
            },
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "critical")

    def test_volume_breach_alone_warns(self):
        r = run(ExtractionQueryPatternValidator, {
            "extraction": {
                "queries": [{"principal_id": "p1", "query_count": 200, "input_space_coverage": 0.1}],
                "max_queries_per_principal": 100,
                "max_input_space_coverage": 0.5,
            },
        })
        self.assertEqual(r.status, Status.WARNING)

    def test_missing_extraction_is_inconclusive(self):
        r = run(ExtractionQueryPatternValidator, {})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_non_numeric_cap_is_inconclusive_not_error(self):
        r = run(ExtractionQueryPatternValidator, {
            "extraction": {
                "queries": [{"principal_id": "p1", "query_count": 10, "input_space_coverage": 0.1}],
                "max_queries_per_principal": "100",
                "max_input_space_coverage": 0.5,
            },
        })
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_empty_queries_is_not_applicable(self):
        r = run(ExtractionQueryPatternValidator, {
            "extraction": {"queries": [], "max_queries_per_principal": 100,
                           "max_input_space_coverage": 0.5},
        })
        self.assertEqual(r.status, Status.NOT_APPLICABLE)

    def test_non_list_queries_is_inconclusive_not_error(self):
        r = run(ExtractionQueryPatternValidator, {
            "extraction": {"queries": {"not": "a list"}, "max_queries_per_principal": 100,
                           "max_input_space_coverage": 0.5},
        })
        self.assertEqual(r.status, Status.INCONCLUSIVE)


class TrainingDataProvenanceTest(unittest.TestCase):
    def test_all_signed_passes(self):
        r = run(TrainingDataProvenanceValidator, {
            "poisoning": {"datasets": [{"id": "d1", "signed_source_record": True}]},
        })
        self.assertEqual(r.status, Status.PASS)

    def test_unsigned_dataset_fails_high(self):
        r = run(TrainingDataProvenanceValidator, {
            "poisoning": {"datasets": [{"id": "d1", "signed_source_record": False}]},
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "high")
        self.assertIn("d1", r.observations[0])

    def test_missing_datasets_is_inconclusive(self):
        r = run(TrainingDataProvenanceValidator, {})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_empty_datasets_is_not_applicable(self):
        r = run(TrainingDataProvenanceValidator, {"poisoning": {"datasets": []}})
        self.assertEqual(r.status, Status.NOT_APPLICABLE)

    def test_non_list_datasets_is_inconclusive_not_error(self):
        r = run(TrainingDataProvenanceValidator, {"poisoning": {"datasets": "not-a-list"}})
        self.assertEqual(r.status, Status.INCONCLUSIVE)


if __name__ == "__main__":
    unittest.main()
