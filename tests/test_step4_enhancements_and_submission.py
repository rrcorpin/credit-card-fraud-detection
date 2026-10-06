import json
import re
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Step4EnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            ROOT
            / "deliverables/04_model_implementation/enhancements/enhancement_summary.json"
        )
        cls.summary = json.loads(path.read_text(encoding="utf-8"))

    def test_search_protocol_preserves_outer_validation_and_test(self):
        protocol = self.summary["protocol"]
        self.assertFalse(protocol["random_search_uses_outer_validation"])
        self.assertFalse(protocol["early_stopping"])
        self.assertFalse(protocol["test_labels_used_for_model_or_calibration_selection"])
        self.assertEqual(protocol["inner_train_rows"], 829_872)
        self.assertEqual(protocol["inner_tuning_rows"], 207_468)
        self.assertEqual(protocol["outer_validation_rows"], 259_335)

    def test_full_representation_is_declared_validation_winner(self):
        winner = self.summary["representation_winner"]
        self.assertEqual(winner["representation"], "full_98_engineered")
        self.assertEqual(winner["feature_count"], 98)
        self.assertGreater(winner["average_precision"], 0.91)

    def test_isotonic_calibration_improves_proper_scoring_losses(self):
        rows = {row["score_type"]: row for row in self.summary["calibration"]}
        raw = rows["raw_class_weighted_score"]
        calibrated = rows["isotonic_recalibrated_probability"]
        self.assertLess(calibrated["brier_score"], raw["brier_score"])
        self.assertLess(calibrated["log_loss"], raw["log_loss"])

    def test_fairness_mitigation_tradeoff_transfers_to_test(self):
        rows = {row["policy"]: row for row in self.summary["fairness_test_transfer"]}
        global_policy = rows["Global top 1% on test scores"]
        mitigation = rows["Validation-derived equal-opportunity cutoffs"]
        self.assertLess(mitigation["true_positive_rate_gap"], global_policy["true_positive_rate_gap"])
        self.assertLess(mitigation["demographic_parity_ratio_disparate_impact"], global_policy["demographic_parity_ratio_disparate_impact"])
        self.assertLess(mitigation["recall"], global_policy["recall"])


class FinalSubmissionTests(unittest.TestCase):
    def test_final_report_files_exist(self):
        final_dir = ROOT / "deliverables/final_submission"
        for name in (
            "Ryan_Rey_Corpin_Capstone_Final_Report.docx",
            "Ryan_Rey_Corpin_Capstone_Final_Report.pdf",
        ):
            path = final_dir / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 100_000, name)

    def test_presentations_have_expected_slide_counts(self):
        expected = {
            "Ryan_Rey_Corpin_Capstone_Technical_Presentation.pptx": 12,
            "Ryan_Rey_Corpin_Capstone_Business_Presentation.pptx": 10,
        }
        slide_name = re.compile(r"ppt/slides/slide\d+\.xml")
        for filename, slide_count in expected.items():
            for folder in ("deliverables/06_presentations", "deliverables/final_submission"):
                deck = ROOT / folder / filename
                self.assertTrue(deck.is_file(), f"{folder}/{filename}")
                with zipfile.ZipFile(deck) as package:
                    self.assertIsNone(package.testzip(), filename)
                    slides = [n for n in package.namelist() if slide_name.fullmatch(n)]
                self.assertEqual(len(slides), slide_count, f"{folder}/{filename}")
                self.assertTrue(8 <= len(slides) <= 12, filename)

    def test_submission_decks_match_presentation_sources(self):
        for filename in (
            "Ryan_Rey_Corpin_Capstone_Technical_Presentation.pptx",
            "Ryan_Rey_Corpin_Capstone_Business_Presentation.pptx",
        ):
            source = ROOT / "deliverables/06_presentations" / filename
            submitted = ROOT / "deliverables/final_submission" / filename
            self.assertEqual(source.read_bytes(), submitted.read_bytes(), filename)


if __name__ == "__main__":
    unittest.main()
