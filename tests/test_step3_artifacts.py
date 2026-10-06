import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP3_DIR = PROJECT_ROOT / "deliverables" / "03_preprocessing_eda_features"


class Step3ArtifactTests(unittest.TestCase):
    def test_split_and_pca_summary_are_consistent(self) -> None:
        summary = json.loads((STEP3_DIR / "analysis_summary.json").read_text())

        self.assertEqual(
            summary["raw_training_rows"],
            summary["development"]["rows"] + summary["validation"]["rows"],
        )
        self.assertLess(
            summary["development"]["end"], summary["validation"]["start"]
        )
        self.assertEqual(summary["cleaning"]["outlier_rows_removed"], 0)
        self.assertEqual(summary["pca"]["input_features"], 20)
        self.assertEqual(summary["pca"]["components_for_threshold"]["0.95"], 13)

    def test_feature_manifest_enforces_identifier_and_audit_policy(self) -> None:
        manifest = json.loads((STEP3_DIR / "feature_manifest.json").read_text())
        primary_features = set(
            manifest["numeric_model_columns"]
            + manifest["one_hot_columns"]
            + manifest["frequency_encoded_columns"]
        )

        self.assertEqual(manifest["encoded_feature_count"], 98)
        self.assertEqual(manifest["filter_selected_feature_count"], 25)
        self.assertTrue(primary_features.isdisjoint(manifest["audit_only_columns"]))
        self.assertTrue(primary_features.isdisjoint(manifest["excluded_direct_identifiers"]))

    def test_expected_figures_and_selected_features_exist(self) -> None:
        figures = list((STEP3_DIR / "figures").glob("*.png"))
        self.assertEqual(len(figures), 11)
        with (STEP3_DIR / "tables" / "selected_features_filter_method.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            selected = list(csv.DictReader(handle))
        self.assertEqual(len(selected), 25)


if __name__ == "__main__":
    unittest.main()

