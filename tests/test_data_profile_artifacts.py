import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP_2_DIR = PROJECT_ROOT / "deliverables" / "02_data_understanding"
PROFILE_PATH = STEP_2_DIR / "data_profile.json"
DICTIONARY_PATH = STEP_2_DIR / "data_dictionary.csv"


class DataProfileArtifactTests(unittest.TestCase):
    def test_profile_internal_consistency(self) -> None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        train = profile["splits"]["train"]
        test = profile["splits"]["test"]
        combined = profile["combined"]

        self.assertEqual(combined["rows"], train["rows"] + test["rows"])
        self.assertEqual(
            combined["fraud_cases"], train["fraud_cases"] + test["fraud_cases"]
        )
        self.assertEqual(
            combined["legitimate_cases"], combined["rows"] - combined["fraud_cases"]
        )
        self.assertEqual(combined["total_missing_values"], 0)
        self.assertEqual(combined["cross_split_transaction_id_overlap"], 0)
        self.assertEqual(train["duplicate_transaction_ids"], 0)
        self.assertEqual(test["duplicate_transaction_ids"], 0)
        self.assertEqual(set(train["invalid_target_values"]), set())
        self.assertEqual(set(test["invalid_target_values"]), set())

    def test_data_dictionary_matches_raw_schema(self) -> None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        with DICTIONARY_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        variables = [row["variable"] for row in rows]
        self.assertEqual(len(variables), 23)
        self.assertEqual(len(set(variables)), 23)
        self.assertEqual(set(variables), set(profile["splits"]["train"]["columns"]))
        self.assertEqual(set(variables), set(profile["splits"]["test"]["columns"]))


if __name__ == "__main__":
    unittest.main()
