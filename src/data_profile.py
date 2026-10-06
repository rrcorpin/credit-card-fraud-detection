"""Create a reproducible profile of the raw Sparkov transaction files.

The profiler reads CSV files in chunks to limit memory use. It writes a JSON
profile and a CSV data dictionary under the Step 2 deliverables directory. It
does not modify the source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_DICTIONARY = [
    {
        "variable": "Unnamed: 0",
        "role": "raw row index",
        "type": "integer identifier",
        "unit_or_format": "sequential integer",
        "allowed_values_or_range": "0 to rows_in_file - 1",
        "description": "CSV export index; not a predictive feature and will be removed.",
    },
    {
        "variable": "trans_date_trans_time",
        "role": "predictor / ordering",
        "type": "datetime",
        "unit_or_format": "YYYY-MM-DD HH:MM:SS",
        "allowed_values_or_range": "observed dataset period",
        "description": "Transaction timestamp used for chronological splitting and temporal features.",
    },
    {
        "variable": "cc_num",
        "role": "entity identifier",
        "type": "integer identifier",
        "unit_or_format": "card number",
        "allowed_values_or_range": "synthetic card identifiers",
        "description": "Synthetic cardholder account identifier; excluded as a raw model feature but usable to construct past-only behavior.",
    },
    {
        "variable": "merchant",
        "role": "predictor",
        "type": "categorical string",
        "unit_or_format": "merchant name",
        "allowed_values_or_range": "observed merchant categories",
        "description": "Synthetic merchant associated with the transaction.",
    },
    {
        "variable": "category",
        "role": "predictor",
        "type": "categorical string",
        "unit_or_format": "merchant category",
        "allowed_values_or_range": "observed category labels",
        "description": "Type of goods or services associated with the merchant.",
    },
    {
        "variable": "amt",
        "role": "predictor / value KPI",
        "type": "continuous numeric",
        "unit_or_format": "US dollars",
        "allowed_values_or_range": "non-negative transaction amount",
        "description": "Monetary value of the transaction.",
    },
    {
        "variable": "first",
        "role": "direct identifier",
        "type": "categorical string",
        "unit_or_format": "given name",
        "allowed_values_or_range": "synthetic names",
        "description": "Synthetic cardholder first name; excluded from modelling.",
    },
    {
        "variable": "last",
        "role": "direct identifier",
        "type": "categorical string",
        "unit_or_format": "family name",
        "allowed_values_or_range": "synthetic names",
        "description": "Synthetic cardholder last name; excluded from modelling.",
    },
    {
        "variable": "gender",
        "role": "protected audit attribute",
        "type": "binary categorical",
        "unit_or_format": "category",
        "allowed_values_or_range": "F, M",
        "description": "Synthetic cardholder gender; retained for fairness auditing only and excluded from model inputs.",
    },
    {
        "variable": "street",
        "role": "direct identifier",
        "type": "categorical string",
        "unit_or_format": "street address",
        "allowed_values_or_range": "synthetic addresses",
        "description": "Synthetic cardholder street address; excluded from modelling.",
    },
    {
        "variable": "city",
        "role": "predictor / quasi-identifier",
        "type": "categorical string",
        "unit_or_format": "city name",
        "allowed_values_or_range": "observed cities",
        "description": "Cardholder home city; evaluated carefully because geography may proxy demographic characteristics.",
    },
    {
        "variable": "state",
        "role": "predictor / quasi-identifier",
        "type": "categorical string",
        "unit_or_format": "US postal abbreviation",
        "allowed_values_or_range": "observed state abbreviations",
        "description": "Cardholder home state.",
    },
    {
        "variable": "zip",
        "role": "predictor / quasi-identifier",
        "type": "categorical integer",
        "unit_or_format": "US ZIP code",
        "allowed_values_or_range": "observed synthetic ZIP codes",
        "description": "Cardholder home ZIP code; treated as categorical, not continuous.",
    },
    {
        "variable": "lat",
        "role": "predictor",
        "type": "continuous numeric",
        "unit_or_format": "decimal degrees",
        "allowed_values_or_range": "-90 to 90",
        "description": "Latitude of the cardholder location.",
    },
    {
        "variable": "long",
        "role": "predictor",
        "type": "continuous numeric",
        "unit_or_format": "decimal degrees",
        "allowed_values_or_range": "-180 to 180",
        "description": "Longitude of the cardholder location.",
    },
    {
        "variable": "city_pop",
        "role": "predictor",
        "type": "non-negative integer",
        "unit_or_format": "people",
        "allowed_values_or_range": "non-negative population",
        "description": "Population of the cardholder city.",
    },
    {
        "variable": "job",
        "role": "predictor / quasi-identifier",
        "type": "categorical string",
        "unit_or_format": "occupation title",
        "allowed_values_or_range": "observed occupation labels",
        "description": "Synthetic cardholder occupation; assessed for proxy-bias risk.",
    },
    {
        "variable": "dob",
        "role": "protected-feature source",
        "type": "date",
        "unit_or_format": "YYYY-MM-DD",
        "allowed_values_or_range": "date before transaction timestamp",
        "description": "Synthetic date of birth; converted to age at transaction for analysis and fairness auditing.",
    },
    {
        "variable": "trans_num",
        "role": "transaction identifier",
        "type": "categorical string",
        "unit_or_format": "32-character hexadecimal identifier",
        "allowed_values_or_range": "synthetic unique transaction IDs",
        "description": "Transaction identifier used for integrity checks; excluded from modelling.",
    },
    {
        "variable": "unix_time",
        "role": "ordering / derived-feature source",
        "type": "integer timestamp",
        "unit_or_format": "seconds since Unix epoch",
        "allowed_values_or_range": "observed dataset period",
        "description": "Machine-readable transaction time; redundant with the human-readable timestamp.",
    },
    {
        "variable": "merch_lat",
        "role": "predictor",
        "type": "continuous numeric",
        "unit_or_format": "decimal degrees",
        "allowed_values_or_range": "-90 to 90",
        "description": "Latitude of the merchant location.",
    },
    {
        "variable": "merch_long",
        "role": "predictor",
        "type": "continuous numeric",
        "unit_or_format": "decimal degrees",
        "allowed_values_or_range": "-180 to 180",
        "description": "Longitude of the merchant location.",
    },
    {
        "variable": "is_fraud",
        "role": "target",
        "type": "binary integer",
        "unit_or_format": "class label",
        "allowed_values_or_range": "0 = legitimate, 1 = fraudulent",
        "description": "Ground-truth transaction fraud label.",
    },
]


def _json_ready(value: Any) -> Any:
    """Convert pandas/NumPy values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _numeric_summary(frames: list[pd.DataFrame], outlier_columns: list[str]) -> dict[str, Any]:
    numeric = pd.concat(frames, ignore_index=True)
    description = numeric.describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]
    ).transpose()
    description = description.rename(columns={"50%": "median"})
    result: dict[str, Any] = {
        "descriptive_statistics": description.to_dict(orient="index"),
        "iqr_outliers": {},
    }
    for column in outlier_columns:
        series = numeric[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        result["iqr_outliers"][column] = {
            "method": "values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR",
            "lower_fence": lower,
            "upper_fence": upper,
            "count": int(mask.sum()),
            "rate": float(mask.mean()),
            "interpretation": "distributional flag, not automatic evidence of invalid data",
        }
    return result


def profile_file(
    path: Path,
    config: dict[str, Any],
    known_transaction_hashes: set[int] | None = None,
) -> tuple[dict[str, Any], set[int]]:
    chunk_size = int(config["chunk_size"])
    timestamp_column = config["timestamp_column"]
    target_column = config["target_column"]
    transaction_id_column = config["transaction_id_column"]
    categorical_columns = config["categorical_columns"]
    numeric_columns = config["numeric_profile_columns"]

    rows = 0
    fraud_cases = 0
    missing = Counter()
    categorical_values: dict[str, Counter] = {
        column: Counter() for column in categorical_columns
    }
    transaction_hashes: set[int] = set()
    duplicate_transaction_ids = 0
    cross_file_transaction_id_overlap = 0
    raw_dtypes: dict[str, str] = {}
    columns: list[str] = []
    min_timestamp: pd.Timestamp | None = None
    max_timestamp: pd.Timestamp | None = None
    invalid_target_values: Counter = Counter()
    validation_counts: Counter = Counter()
    age_min: float | None = None
    age_max: float | None = None
    numeric_frames: list[pd.DataFrame] = []
    index_min: int | None = None
    index_max: int | None = None
    index_count = 0
    index_sum = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    ):
        if chunk_number == 0:
            columns = chunk.columns.tolist()
            raw_dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}

        rows += len(chunk)
        missing.update(chunk.isna().sum().astype(int).to_dict())

        target_counts = chunk[target_column].value_counts(dropna=False)
        fraud_cases += int(target_counts.get(1, 0))
        for value, count in target_counts.items():
            if value not in (0, 1):
                invalid_target_values[str(value)] += int(count)

        parsed_time = pd.to_datetime(chunk[timestamp_column], errors="coerce")
        validation_counts["invalid_transaction_timestamp"] += int(parsed_time.isna().sum())
        current_min = parsed_time.min()
        current_max = parsed_time.max()
        if pd.notna(current_min):
            min_timestamp = current_min if min_timestamp is None else min(min_timestamp, current_min)
        if pd.notna(current_max):
            max_timestamp = current_max if max_timestamp is None else max(max_timestamp, current_max)

        parsed_dob = pd.to_datetime(chunk["dob"], errors="coerce")
        validation_counts["invalid_date_of_birth"] += int(parsed_dob.isna().sum())
        valid_age = (parsed_time - parsed_dob).dt.total_seconds() / (365.2425 * 24 * 60 * 60)
        validation_counts["date_of_birth_after_transaction"] += int((valid_age < 0).sum())
        validation_counts["derived_age_under_18"] += int((valid_age < 18).sum())
        validation_counts["derived_age_over_100"] += int((valid_age > 100).sum())
        current_age_min = valid_age.min()
        current_age_max = valid_age.max()
        if pd.notna(current_age_min):
            age_min = float(current_age_min) if age_min is None else min(age_min, float(current_age_min))
        if pd.notna(current_age_max):
            age_max = float(current_age_max) if age_max is None else max(age_max, float(current_age_max))

        amount = pd.to_numeric(chunk["amt"], errors="coerce")
        city_population = pd.to_numeric(chunk["city_pop"], errors="coerce")
        validation_counts["nonpositive_amount"] += int((amount <= 0).sum())
        validation_counts["nonpositive_city_population"] += int((city_population <= 0).sum())
        validation_counts["invalid_cardholder_latitude"] += int(
            (~pd.to_numeric(chunk["lat"], errors="coerce").between(-90, 90)).sum()
        )
        validation_counts["invalid_cardholder_longitude"] += int(
            (~pd.to_numeric(chunk["long"], errors="coerce").between(-180, 180)).sum()
        )
        validation_counts["invalid_merchant_latitude"] += int(
            (~pd.to_numeric(chunk["merch_lat"], errors="coerce").between(-90, 90)).sum()
        )
        validation_counts["invalid_merchant_longitude"] += int(
            (~pd.to_numeric(chunk["merch_long"], errors="coerce").between(-180, 180)).sum()
        )

        for column in categorical_columns:
            categorical_values[column].update(
                chunk[column].dropna().astype(str).value_counts().to_dict()
            )

        hashed_ids = pd.util.hash_pandas_object(
            chunk[transaction_id_column].astype(str), index=False
        ).astype("uint64")
        hashed_list = [int(value) for value in hashed_ids]
        duplicate_transaction_ids += len(hashed_list) - len(set(hashed_list))
        duplicate_transaction_ids += sum(value in transaction_hashes for value in set(hashed_list))
        if known_transaction_hashes is not None:
            cross_file_transaction_id_overlap += sum(
                value in known_transaction_hashes for value in set(hashed_list)
            )
        transaction_hashes.update(hashed_list)

        raw_index = pd.to_numeric(chunk[config["raw_row_index_column"]], errors="coerce")
        index_count += int(raw_index.notna().sum())
        index_sum += int(raw_index.dropna().sum())
        current_index_min = raw_index.min()
        current_index_max = raw_index.max()
        if pd.notna(current_index_min):
            value = int(current_index_min)
            index_min = value if index_min is None else min(index_min, value)
        if pd.notna(current_index_max):
            value = int(current_index_max)
            index_max = value if index_max is None else max(index_max, value)

        numeric_frames.append(chunk[numeric_columns].copy())

    expected_index_sum = rows * (rows - 1) // 2
    index_is_sequential = (
        index_count == rows
        and index_min == 0
        and index_max == rows - 1
        and index_sum == expected_index_sum
    )

    categorical_summary = {}
    for column, counts in categorical_values.items():
        ordered = counts.most_common()
        categorical_summary[column] = {
            "cardinality": len(counts),
            "allowed_values": sorted(counts) if len(counts) <= 60 else None,
            "top_values": [
                {"value": value, "count": int(count), "rate": count / rows}
                for value, count in ordered[:10]
            ],
        }

    profile = {
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "columns_count": len(columns),
        "columns": columns,
        "raw_dtypes": raw_dtypes,
        "missing_counts": dict(missing),
        "total_missing_values": int(sum(missing.values())),
        "fraud_cases": fraud_cases,
        "legitimate_cases": rows - fraud_cases,
        "fraud_rate": fraud_cases / rows,
        "timestamp_min": min_timestamp,
        "timestamp_max": max_timestamp,
        "invalid_target_values": dict(invalid_target_values),
        "validation_counts": dict(validation_counts),
        "derived_age_at_transaction_range_years": {
            "min": age_min,
            "max": age_max,
        },
        "duplicate_transaction_ids": duplicate_transaction_ids,
        "cross_file_transaction_id_overlap": cross_file_transaction_id_overlap,
        "raw_index_is_sequential": index_is_sequential,
        "categorical_summary": categorical_summary,
        "numeric_summary": _numeric_summary(numeric_frames, config["iqr_outlier_columns"]),
    }
    return _json_ready(profile), transaction_hashes


def build_profile(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = project_root / "configs" / "data_profile.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    output_dir = project_root / "deliverables" / "02_data_understanding"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = project_root / config["files"]["train"]
    test_path = project_root / config["files"]["test"]
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required raw data file not found: {path}")

    train_profile, train_hashes = profile_file(train_path, config)
    test_profile, _ = profile_file(test_path, config, known_transaction_hashes=train_hashes)

    combined_rows = train_profile["rows"] + test_profile["rows"]
    combined_fraud = train_profile["fraud_cases"] + test_profile["fraud_cases"]
    profile = {
        "profile_version": 1,
        "dataset_name": config["dataset_name"],
        "configuration": str(config_path.relative_to(project_root)),
        "methodology": {
            "read_strategy": f"pandas chunked CSV reads of {config['chunk_size']:,} rows",
            "missingness": "exact null counts from all rows",
            "duplicate_check": "exact 64-bit pandas hashes of transaction IDs; collision risk is negligible but non-zero",
            "outlier_rule": "exact per-file Tukey 1.5*IQR flags for amount and city population; flags are not automatically removed",
            "test_data_policy": "descriptive profiling only; test labels are prohibited from model, feature, hyperparameter, or threshold selection",
        },
        "combined": {
            "rows": combined_rows,
            "fraud_cases": combined_fraud,
            "legitimate_cases": combined_rows - combined_fraud,
            "fraud_rate": combined_fraud / combined_rows,
            "timestamp_min": train_profile["timestamp_min"],
            "timestamp_max": test_profile["timestamp_max"],
            "total_missing_values": train_profile["total_missing_values"]
            + test_profile["total_missing_values"],
            "cross_split_transaction_id_overlap": test_profile[
                "cross_file_transaction_id_overlap"
            ],
        },
        "splits": {"train": train_profile, "test": test_profile},
    }

    profile_path = output_dir / "data_profile.json"
    profile_path.write_text(
        json.dumps(_json_ready(profile), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    dictionary_path = output_dir / "data_dictionary.csv"
    pd.DataFrame(DATA_DICTIONARY).to_csv(dictionary_path, index=False)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Capstone project root (defaults to the parent of src/).",
    )
    args = parser.parse_args()
    profile = build_profile(args.project_root)
    combined = profile["combined"]
    print(
        "Profile complete: "
        f"{combined['rows']:,} rows, "
        f"{combined['fraud_cases']:,} fraud cases, "
        f"{combined['fraud_rate']:.4%} fraud rate."
    )
    print(
        args.project_root
        / "deliverables"
        / "02_data_understanding"
        / "data_profile.json"
    )


if __name__ == "__main__":
    main()
