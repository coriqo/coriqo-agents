"""The four MODEL validators.

Pure stdlib `unittest`, same convention as test_genai_validators.py.
Runnable standalone (`python test_model_validators.py`) and under pytest.

The class of bug this file exists to pin: a MODEL input that was never
supplied must never read as clean — a missing report reads INCONCLUSIVE,
a supplied-but-empty collection reads NOT_APPLICABLE — plus the
threshold-direction split (gte/lte), the fairness group-gap comparison,
the drift per-feature comparison, and the reproducibility per-row
comparison.
"""
from __future__ import annotations

import unittest

from coriqo_agents.runner.contract import Status, ValidatorContext
from coriqo_agents.runner.validators import VALIDATOR_REGISTRY
from coriqo_agents.runner.validators.model_drift_psi import ModelDriftPsiValidator
from coriqo_agents.runner.validators.model_fairness_parity import ModelFairnessParityValidator
from coriqo_agents.runner.validators.model_performance_threshold import (
    ModelPerformanceThresholdValidator,
)
from coriqo_agents.runner.validators.model_reproducibility import ModelReproducibilityValidator

MODEL_VALIDATOR_IDS = (
    "model_performance_threshold",
    "model_fairness_parity",
    "model_drift_psi",
    "model_reproducibility",
)


def ctx(model=None, **kw):
    return ValidatorContext(
        execution_id="e1", agent_model_id="m1", agent_version_ref=None,
        environment="STAGING", configuration=kw.get("configuration", {}),
        inputs={"model": model} if model is not None else {},
    )


def run(cls, model=None, **kw):
    v = cls()
    return v.evaluate(v.execute(ctx(model, **kw)))


class RegistryTest(unittest.TestCase):
    def test_all_four_registered(self):
        for vid in MODEL_VALIDATOR_IDS:
            self.assertIn(vid, VALIDATOR_REGISTRY)


class PerformanceThresholdTest(unittest.TestCase):
    def test_gte_metric_clearing_threshold_passes(self):
        r = run(ModelPerformanceThresholdValidator, {
            "performance": {"metric": "auc", "observed": 0.91, "threshold": 0.85, "direction": "gte"},
        })
        self.assertEqual(r.status, Status.PASS)

    def test_gte_metric_below_threshold_fails_high(self):
        r = run(ModelPerformanceThresholdValidator, {
            "performance": {"metric": "auc", "observed": 0.80, "threshold": 0.85, "direction": "gte"},
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "high")

    def test_lte_metric_above_threshold_fails(self):
        r = run(ModelPerformanceThresholdValidator, {
            "performance": {"metric": "error_rate", "observed": 0.10, "threshold": 0.05, "direction": "lte"},
        })
        self.assertEqual(r.status, Status.FAIL)

    def test_missing_performance_is_inconclusive(self):
        r = run(ModelPerformanceThresholdValidator, {})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_unknown_direction_is_inconclusive_not_error(self):
        r = run(ModelPerformanceThresholdValidator, {
            "performance": {"metric": "auc", "observed": 0.9, "threshold": 0.85, "direction": "sideways"},
        })
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_missing_observed_is_inconclusive_not_error(self):
        r = run(ModelPerformanceThresholdValidator, {
            "performance": {"metric": "auc", "threshold": 0.85, "direction": "gte"},
        })
        self.assertEqual(r.status, Status.INCONCLUSIVE)


class FairnessParityTest(unittest.TestCase):
    def test_gap_within_tolerance_passes(self):
        r = run(ModelFairnessParityValidator, {
            "fairness": {"metric": "tpr", "group_values": {"a": 0.80, "b": 0.78}, "tolerance": 0.05},
        })
        self.assertEqual(r.status, Status.PASS)

    def test_gap_over_tolerance_fails_high(self):
        r = run(ModelFairnessParityValidator, {
            "fairness": {"metric": "tpr", "group_values": {"a": 0.90, "b": 0.60}, "tolerance": 0.05},
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "high")

    def test_missing_fairness_is_inconclusive(self):
        r = run(ModelFairnessParityValidator, {})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_single_group_is_not_applicable(self):
        r = run(ModelFairnessParityValidator, {
            "fairness": {"metric": "tpr", "group_values": {"a": 0.8}, "tolerance": 0.05},
        })
        self.assertEqual(r.status, Status.NOT_APPLICABLE)

    def test_missing_tolerance_is_inconclusive_not_error(self):
        r = run(ModelFairnessParityValidator, {
            "fairness": {"metric": "tpr", "group_values": {"a": 0.8, "b": 0.6}},
        })
        self.assertEqual(r.status, Status.INCONCLUSIVE)


class DriftPsiTest(unittest.TestCase):
    def test_all_features_within_threshold_passes(self):
        r = run(ModelDriftPsiValidator, {
            "drift": {"feature_psi": {"income": 0.05, "age": 0.02}, "threshold": 0.1},
        })
        self.assertEqual(r.status, Status.PASS)

    def test_drifted_feature_fails_high(self):
        r = run(ModelDriftPsiValidator, {
            "drift": {"feature_psi": {"income": 0.25, "age": 0.02}, "threshold": 0.1},
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "high")

    def test_missing_drift_is_inconclusive(self):
        r = run(ModelDriftPsiValidator, {})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_missing_threshold_is_inconclusive_not_error(self):
        r = run(ModelDriftPsiValidator, {"drift": {"feature_psi": {"income": 0.05}}})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_empty_feature_psi_is_not_applicable(self):
        r = run(ModelDriftPsiValidator, {"drift": {"feature_psi": {}, "threshold": 0.1}})
        self.assertEqual(r.status, Status.NOT_APPLICABLE)


class ReproducibilityTest(unittest.TestCase):
    def test_sample_within_tolerance_passes(self):
        r = run(ModelReproducibilityValidator, {
            "reproducibility": {
                "sample": [{"id": "r1", "stored_prediction": 0.5, "recomputed_prediction": 0.501}],
                "tolerance": 0.01,
            },
        })
        self.assertEqual(r.status, Status.PASS)

    def test_sample_outside_tolerance_fails_high(self):
        r = run(ModelReproducibilityValidator, {
            "reproducibility": {
                "sample": [{"id": "r1", "stored_prediction": 0.5, "recomputed_prediction": 0.9}],
                "tolerance": 0.01,
            },
        })
        self.assertEqual(r.status, Status.FAIL)
        self.assertEqual(r.severity, "high")

    def test_missing_reproducibility_is_inconclusive(self):
        r = run(ModelReproducibilityValidator, {})
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_missing_recomputed_prediction_is_inconclusive_not_error(self):
        r = run(ModelReproducibilityValidator, {
            "reproducibility": {
                "sample": [{"id": "r1", "stored_prediction": 0.5}],
                "tolerance": 0.01,
            },
        })
        self.assertEqual(r.status, Status.INCONCLUSIVE)

    def test_empty_sample_is_not_applicable(self):
        r = run(ModelReproducibilityValidator, {"reproducibility": {"sample": [], "tolerance": 0.01}})
        self.assertEqual(r.status, Status.NOT_APPLICABLE)


if __name__ == "__main__":
    unittest.main()
