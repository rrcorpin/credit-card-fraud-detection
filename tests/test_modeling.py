import unittest

import numpy as np

from src.modeling import evaluate_probabilities, top_capacity_predictions


class ModelingUtilityTests(unittest.TestCase):
    def test_top_capacity_flags_exact_ceiling_count(self):
        scores = np.asarray([0.1, 0.9, 0.7, 0.7, 0.2])
        predictions, cutoff = top_capacity_predictions(scores, 0.4)
        self.assertEqual(int(predictions.sum()), 2)
        self.assertEqual(predictions.tolist(), [0, 1, 1, 0, 0])
        self.assertAlmostEqual(cutoff, 0.7)

    def test_capacity_metrics_are_internally_consistent(self):
        target = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        amounts = np.asarray([10.0, 20.0, 30.0, 70.0])
        _, rows = evaluate_probabilities(target, scores, amounts, [0.5])
        row = rows[0]
        self.assertEqual(row["true_positives"], 2)
        self.assertEqual(row["false_positives"], 0)
        self.assertAlmostEqual(row["recall"], 1.0)
        self.assertAlmostEqual(row["fraud_value_capture_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
