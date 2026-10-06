"""Leakage-conscious feature engineering and preprocessing for Sparkov data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_USECOLS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

RAW_DTYPES = {
    "cc_num": "string",
    "merchant": "category",
    "category": "category",
    "amt": "float64",
    "gender": "category",
    "city": "category",
    "state": "category",
    "zip": "string",
    "lat": "float32",
    "long": "float32",
    "city_pop": "int32",
    "job": "category",
    "unix_time": "int64",
    "merch_lat": "float32",
    "merch_long": "float32",
    "is_fraud": "int8",
}

DIRECT_IDENTIFIERS_EXCLUDED = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "trans_num",
    "cc_num",
]

AUDIT_ONLY_COLUMNS = ["gender", "age_at_transaction", "age_band"]

NUMERIC_MODEL_COLUMNS = [
    "log_amt",
    "log_city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "customer_merchant_distance_km",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "is_night",
    "is_first_card_transaction",
    "log_card_prior_txn_count",
    "log_seconds_since_prior_card_txn",
    "log_card_prior_mean_amt",
    "log_amount_to_prior_mean_ratio",
    "log_merchant_prior_txn_count",
    "log_card_category_prior_txn_count",
]

LOW_CARDINALITY_MODEL_COLUMNS = [
    "category",
    "state",
    "amount_band",
    "hour_band",
]

HIGH_CARDINALITY_MODEL_COLUMNS = ["merchant", "city", "zip", "job"]


@dataclass(frozen=True)
class ChronologicalSplit:
    development: pd.DataFrame
    validation: pd.DataFrame
    boundary_timestamp: pd.Timestamp


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Training-frequency encoding with unseen categories mapped to zero."""

    def __init__(self) -> None:
        self.feature_names_in_: np.ndarray | None = None
        self.mappings_: list[dict[str, float]] = []

    def fit(self, X: pd.DataFrame, y: object = None) -> "FrequencyEncoder":
        frame = self._as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.mappings_ = []
        denominator = len(frame)
        for column in frame.columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            self.mappings_.append(
                (values.value_counts(dropna=False) / denominator).to_dict()
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._as_frame(X, columns=self.feature_names_in_)
        encoded = []
        for column, mapping in zip(frame.columns, self.mappings_, strict=True):
            values = frame[column].astype("string").fillna("__MISSING__")
            encoded.append(values.map(mapping).fillna(0.0).to_numpy(dtype=float))
        return np.column_stack(encoded)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        features = (
            np.asarray(list(input_features), dtype=object)
            if input_features is not None
            else self.feature_names_in_
        )
        if features is None:
            raise RuntimeError("FrequencyEncoder has not been fitted.")
        return np.asarray([f"{feature}_frequency" for feature in features], dtype=object)

    @staticmethod
    def _as_frame(
        X: pd.DataFrame | np.ndarray,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=columns)


def load_raw_transactions(path: Path) -> pd.DataFrame:
    """Load only columns needed for EDA, feature engineering, and auditing."""
    frame = pd.read_csv(
        path,
        usecols=RAW_USECOLS,
        dtype=RAW_DTYPES,
        parse_dates=["trans_date_trans_time", "dob"],
        low_memory=False,
    )
    frame["cc_num"] = frame["cc_num"].astype("category")
    frame["zip"] = frame["zip"].astype("category")
    return frame


def _haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(lat1.astype(float).to_numpy())
    lon1_rad = np.radians(lon1.astype(float).to_numpy())
    lat2_rad = np.radians(lat2.astype(float).to_numpy())
    lon2_rad = np.radians(lon2.astype(float).to_numpy())
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(haversine))


def _strict_prior_aggregates(
    frame: pd.DataFrame,
    group_columns: list[str],
    prefix: str,
    include_amount: bool = False,
) -> pd.DataFrame:
    """Aggregate only transactions with timestamps strictly before each row.

    Transactions sharing the same timestamp within a group receive identical
    history, so simultaneous events never leak into one another.
    """
    keys = group_columns + ["trans_date_trans_time"]
    aggregations: dict[str, tuple[str, str]] = {"events_at_time": ("amt", "size")}
    if include_amount:
        aggregations["amount_at_time"] = ("amt", "sum")
    grouped = (
        frame.groupby(keys, observed=True, sort=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values(keys, kind="mergesort")
    )
    groupby = grouped.groupby(group_columns, observed=True, sort=False)
    grouped[f"{prefix}_prior_txn_count"] = (
        groupby["events_at_time"].cumsum() - grouped["events_at_time"]
    )
    if include_amount:
        grouped[f"{prefix}_prior_amount_sum"] = (
            groupby["amount_at_time"].cumsum() - grouped["amount_at_time"]
        )
    grouped[f"{prefix}_prior_timestamp"] = groupby["trans_date_trans_time"].shift(1)

    keep = keys + [f"{prefix}_prior_txn_count", f"{prefix}_prior_timestamp"]
    if include_amount:
        keep.append(f"{prefix}_prior_amount_sum")
    return frame.merge(grouped[keep], on=keys, how="left", sort=False, validate="many_to_one")


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create row-level and strict-past behavioral features without labels."""
    frame = raw.sort_values("trans_date_trans_time", kind="mergesort").reset_index(drop=True).copy()

    timestamp = frame["trans_date_trans_time"]
    frame["transaction_month"] = timestamp.dt.to_period("M").astype(str)
    frame["hour"] = timestamp.dt.hour.astype("int8")
    frame["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int8")
    frame["is_night"] = frame["hour"].between(0, 5).astype("int8")
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["day_of_week_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["hour_band"] = pd.cut(
        frame["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    )

    age = (timestamp - frame["dob"]).dt.total_seconds() / (365.2425 * 24 * 60 * 60)
    frame["age_at_transaction"] = age.astype("float32")
    frame["age_band"] = pd.cut(
        age,
        bins=[-np.inf, 17, 24, 34, 44, 54, 64, np.inf],
        labels=["under_18", "18_24", "25_34", "35_44", "45_54", "55_64", "65_plus"],
    )
    frame["amount_band"] = pd.cut(
        frame["amt"],
        bins=[-np.inf, 10, 50, 100, 500, np.inf],
        labels=["under_10", "10_50", "50_100", "100_500", "over_500"],
    )
    frame["log_amt"] = np.log1p(frame["amt"])
    frame["log_city_pop"] = np.log1p(frame["city_pop"])
    frame["customer_merchant_distance_km"] = _haversine_km(
        frame["lat"], frame["long"], frame["merch_lat"], frame["merch_long"]
    )

    frame = _strict_prior_aggregates(frame, ["cc_num"], "card", include_amount=True)
    frame = _strict_prior_aggregates(frame, ["merchant"], "merchant")
    frame = _strict_prior_aggregates(frame, ["cc_num", "category"], "card_category")

    frame["seconds_since_prior_card_txn"] = (
        frame["trans_date_trans_time"] - frame["card_prior_timestamp"]
    ).dt.total_seconds()
    frame["is_first_card_transaction"] = (frame["card_prior_txn_count"] == 0).astype("int8")
    prior_count = frame["card_prior_txn_count"].astype(float)
    frame["card_prior_mean_amt"] = np.where(
        prior_count > 0,
        frame["card_prior_amount_sum"] / prior_count,
        0.0,
    )
    frame["amount_to_prior_mean_ratio"] = np.where(
        prior_count > 0,
        frame["amt"] / np.maximum(frame["card_prior_mean_amt"], 0.01),
        1.0,
    )

    frame["log_card_prior_txn_count"] = np.log1p(frame["card_prior_txn_count"])
    frame["log_seconds_since_prior_card_txn"] = np.log1p(
        frame["seconds_since_prior_card_txn"].clip(lower=0).fillna(0)
    )
    frame["log_card_prior_mean_amt"] = np.log1p(frame["card_prior_mean_amt"])
    frame["log_amount_to_prior_mean_ratio"] = np.log1p(
        frame["amount_to_prior_mean_ratio"].clip(lower=0)
    )
    frame["log_merchant_prior_txn_count"] = np.log1p(frame["merchant_prior_txn_count"])
    frame["log_card_category_prior_txn_count"] = np.log1p(
        frame["card_category_prior_txn_count"]
    )
    return frame


def chronological_development_validation_split(
    frame: pd.DataFrame,
    development_fraction: float,
) -> ChronologicalSplit:
    """Split on a timestamp boundary while keeping tied timestamps together."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between zero and one")
    ordered = frame.sort_values("trans_date_trans_time", kind="mergesort").reset_index(drop=True)
    boundary_position = int(len(ordered) * development_fraction)
    boundary_timestamp = ordered.loc[boundary_position, "trans_date_trans_time"]
    development = ordered.loc[ordered["trans_date_trans_time"] < boundary_timestamp].copy()
    validation = ordered.loc[ordered["trans_date_trans_time"] >= boundary_timestamp].copy()
    if development.empty or validation.empty:
        raise RuntimeError("Chronological split produced an empty partition")
    return ChronologicalSplit(development, validation, boundary_timestamp)


def build_preprocessor() -> ColumnTransformer:
    """Return a training-fitted encoder/scaler for the candidate feature set."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_MODEL_COLUMNS),
            ("one_hot", categorical_pipeline, LOW_CARDINALITY_MODEL_COLUMNS),
            ("frequency", FrequencyEncoder(), HIGH_CARDINALITY_MODEL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

