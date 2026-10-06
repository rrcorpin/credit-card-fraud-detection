import unittest

import pandas as pd

from src.features import (
    chronological_development_validation_split,
    engineer_features,
)


def _small_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                ]
            ),
            "cc_num": ["A", "A", "A", "B"],
            "merchant": ["M1", "M1", "M1", "M2"],
            "category": ["shopping", "shopping", "shopping", "grocery"],
            "amt": [10.0, 20.0, 30.0, 5.0],
            "gender": ["F", "F", "F", "M"],
            "city": ["X", "X", "X", "Y"],
            "state": ["AA", "AA", "AA", "BB"],
            "zip": ["00001", "00001", "00001", "00002"],
            "lat": [0.0, 0.0, 0.0, 1.0],
            "long": [0.0, 0.0, 0.0, 1.0],
            "city_pop": [100, 100, 100, 200],
            "job": ["J1", "J1", "J1", "J2"],
            "dob": pd.to_datetime(["1990-01-01"] * 3 + ["1980-01-01"]),
            "unix_time": [1577836800, 1577836800, 1577840400, 1577844000],
            "merch_lat": [0.1, 0.1, 0.1, 1.1],
            "merch_long": [0.1, 0.1, 0.1, 1.1],
            "is_fraud": [0, 0, 1, 0],
        }
    )


class FeatureEngineeringTests(unittest.TestCase):
    def test_simultaneous_transactions_do_not_enter_each_others_history(self) -> None:
        engineered = engineer_features(_small_transactions())
        simultaneous = engineered.loc[
            engineered["trans_date_trans_time"] == pd.Timestamp("2020-01-01 00:00:00")
        ]
        later = engineered.loc[
            engineered["trans_date_trans_time"] == pd.Timestamp("2020-01-01 01:00:00")
        ].iloc[0]

        self.assertTrue((simultaneous["card_prior_txn_count"] == 0).all())
        self.assertTrue((simultaneous["merchant_prior_txn_count"] == 0).all())
        self.assertEqual(later["card_prior_txn_count"], 2)
        self.assertAlmostEqual(later["card_prior_mean_amt"], 15.0)
        self.assertEqual(later["card_category_prior_txn_count"], 2)

    def test_chronological_split_keeps_timestamp_ties_together(self) -> None:
        engineered = engineer_features(_small_transactions())
        split = chronological_development_validation_split(engineered, 0.50)

        self.assertLess(
            split.development["trans_date_trans_time"].max(),
            split.validation["trans_date_trans_time"].min(),
        )
        development_times = set(split.development["trans_date_trans_time"])
        validation_times = set(split.validation["trans_date_trans_time"])
        self.assertTrue(development_times.isdisjoint(validation_times))


if __name__ == "__main__":
    unittest.main()
