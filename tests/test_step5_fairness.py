import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.step5_fairness import (
    _equal_group_rate_predictions,
    _equal_opportunity_predictions,
    _wilson_interval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "deliverables" / "05_explainability_ethics_fairness"


class Step5FairnessTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = _wilson_interval(80, 100)
        self.assertLessEqual(low, 0.8)
        self.assertGreaterEqual(high, 0.8)

    def test_equal_group_rate_preserves_exact_capacity(self):
        scores = np.linspace(0, 1, 100)
        groups = np.asarray(["F"] * 60 + ["M"] * 40)
        predictions, _ = _equal_group_rate_predictions(scores, groups, 0.1)
        self.assertEqual(int(predictions.sum()), 10)
        self.assertEqual(int(predictions[groups == "F"].sum()), 6)
        self.assertEqual(int(predictions[groups == "M"].sum()), 4)

    def test_equal_opportunity_search_preserves_exact_capacity(self):
        scores = np.linspace(1, 0, 100)
        target = np.asarray(([1] * 10 + [0] * 40) + ([0] * 10 + [1] * 10 + [0] * 30))
        groups = np.asarray(["F"] * 50 + ["M"] * 50)
        predictions, _, quotas = _equal_opportunity_predictions(
            scores, target, groups, 0.2
        )
        self.assertEqual(int(predictions.sum()), 20)
        self.assertEqual(sum(quotas.values()), 20)


class Step5ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((OUTPUT / "audit_summary.json").read_text())
        cls.mitigation = pd.read_csv(
            OUTPUT / "tables" / "gender_mitigation_comparison.csv"
        )

    def test_test_data_remains_sealed_and_missing_attributes_are_explicit(self):
        protocol = self.summary["protocol"]
        self.assertFalse(protocol["supplied_test_used"])
        self.assertFalse(protocol["race_audit_available"])
        self.assertFalse(protocol["socioeconomic_status_audit_available"])
        self.assertFalse(protocol["city_population_is_socioeconomic_status"])

    def test_mitigation_scenarios_keep_same_review_rate(self):
        baseline = self.summary["mitigation"]["baseline"]["review_rate"]
        equal_rate = self.summary["mitigation"]["equal_rate_scenario"]["metrics"]["review_rate"]
        opportunity = self.summary["mitigation"]["equal_opportunity_scenario"]["metrics"]["review_rate"]
        self.assertAlmostEqual(baseline, equal_rate)
        self.assertAlmostEqual(baseline, opportunity)

    def test_expected_tables_and_figures_exist(self):
        self.assertEqual(len(list((OUTPUT / "figures").glob("*.png"))), 11)
        required_tables = {
            "gender_group_metrics.csv",
            "age_group_metrics.csv",
            "gender_age_intersection_metrics.csv",
            "gender_mitigation_comparison.csv",
            "shap_global_importance.csv",
            "local_shap_cases.csv",
            "temporal_robustness.csv",
        }
        self.assertTrue(required_tables.issubset({path.name for path in (OUTPUT / "tables").glob("*.csv")}))


if __name__ == "__main__":
    unittest.main()
