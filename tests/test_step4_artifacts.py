import json
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "deliverables" / "04_model_implementation"


class Step4ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((OUTPUT / "metrics.json").read_text())
        cls.comparison = pd.read_csv(OUTPUT / "tables" / "model_comparison.csv")
        cls.capacity = pd.read_csv(OUTPUT / "tables" / "capacity_metrics.csv")

    def test_test_file_was_not_used(self):
        self.assertFalse(self.metrics["protocol"]["supplied_test_used"])

    def test_selected_model_matches_highest_primary_metric(self):
        expected = self.comparison.sort_values("average_precision", ascending=False).iloc[0]
        self.assertEqual(self.metrics["selection"]["selected_model"], expected["model"])

    def test_every_model_has_three_capacity_rows_and_saved_artifact(self):
        expected_models = set(self.comparison["model"])
        self.assertEqual(set(self.capacity["model"]), expected_models)
        counts = self.capacity.groupby("model").size()
        self.assertTrue((counts == 3).all())
        for model in expected_models:
            self.assertTrue((PROJECT_ROOT / "models" / "step4" / f"{model}.joblib").exists())

    def test_expected_figures_exist(self):
        figures = list((OUTPUT / "figures").glob("*.png"))
        self.assertEqual(len(figures), 6)


if __name__ == "__main__":
    unittest.main()
