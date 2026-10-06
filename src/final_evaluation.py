"""Refit the frozen winner, evaluate once on Sparkov test, and model cost scenarios."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(8, os.cpu_count() or 1)))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.features import build_preprocessor, engineer_features, load_raw_transactions
from src.modeling import build_model, evaluate_probabilities, top_capacity_predictions
from src.step5_fairness import _fairness_summary, _group_metrics


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _transform_selected(preprocessor, frame, positions, chunk_size):
    output = np.empty((len(frame), len(positions)), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        transformed = preprocessor.transform(frame.iloc[start:stop])
        output[start:stop] = np.asarray(transformed[:, positions], dtype=np.float32)
    return output


def _cost_sensitivity(capacity_metrics, cost_config):
    rows = []
    for metric in capacity_metrics:
        reviewed = metric["true_positives"] + metric["false_positives"]
        for recovery_rate in cost_config["fraud_value_recovery_rates"]:
            for review_cost in cost_config["review_costs_usd"]:
                for friction_cost in cost_config["false_positive_friction_costs_usd"]:
                    avoided_loss = metric["fraud_value_captured"] * recovery_rate
                    operating_cost = reviewed * review_cost + metric["false_positives"] * friction_cost
                    net_benefit = avoided_loss - operating_cost
                    rows.append(
                        {
                            "review_capacity": metric["review_capacity"],
                            "fraud_value_recovery_rate": recovery_rate,
                            "review_cost_usd": review_cost,
                            "false_positive_friction_cost_usd": friction_cost,
                            "reviewed_transactions": reviewed,
                            "false_positives": metric["false_positives"],
                            "fraud_value_captured_usd": metric["fraud_value_captured"],
                            "avoided_loss_usd": avoided_loss,
                            "operating_cost_usd": operating_cost,
                            "net_benefit_usd": net_benefit,
                            "benefit_cost_ratio": avoided_loss / operating_cost if operating_cost else float("inf"),
                            "roi": net_benefit / operating_cost if operating_cost else float("inf"),
                        }
                    )
    return pd.DataFrame(rows)


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = json.loads((root / "configs" / "final_evaluation.json").read_text())
    model_config = json.loads((root / "configs" / "step4_models.json").read_text())
    seed = int(config["random_seed"])
    np.random.seed(seed)

    output = root / "deliverables" / "04_model_implementation" / "final_test"
    table_dir = output / "tables"
    model_dir = root / "models" / "final"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_raw_transactions(root / "data" / "raw" / "sparkov" / "fraudTrain.csv")
    test_raw = load_raw_transactions(root / "data" / "raw" / "sparkov" / "fraudTest.csv")
    train_raw = train_raw.assign(_source="train", _source_row=np.arange(len(train_raw)))
    test_raw = test_raw.assign(_source="test", _source_row=np.arange(len(test_raw)))
    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    engineered = engineer_features(combined)
    train = engineered.loc[engineered["_source"].eq("train")].sort_values("_source_row").reset_index(drop=True)
    test = engineered.loc[engineered["_source"].eq("test")].sort_values("_source_row").reset_index(drop=True)

    preprocessor = build_preprocessor()
    preprocessor.fit(train)
    all_names = list(preprocessor.get_feature_names_out())
    selected_names = json.loads(
        (root / "models" / "step4" / "selected_features.json").read_text()
    )
    positions = np.asarray([all_names.index(name) for name in selected_names])
    chunk_size = int(config["transform_chunk_size"])
    X_train = _transform_selected(preprocessor, train, positions, chunk_size)
    X_test = _transform_selected(preprocessor, test, positions, chunk_size)
    y_train = train["is_fraud"].to_numpy(dtype=np.int8)
    y_test = test["is_fraud"].to_numpy(dtype=np.int8)

    specification = model_config["models"][config["model_name"]]
    model = build_model(specification, seed)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    capacities = [float(value) for value in config["review_capacities"]]
    probability_metrics, capacity_rows = evaluate_probabilities(
        y_test, scores, test["amt"].to_numpy(dtype=float), capacities
    )
    capacity_table = pd.DataFrame(capacity_rows)
    capacity_table.to_csv(table_dir / "final_test_capacity_metrics.csv", index=False)

    recommended_capacity = float(config["recommended_review_capacity"])
    recommended_predictions, recommended_cutoff = top_capacity_predictions(
        scores, recommended_capacity
    )
    test_audit = test.copy()
    test_audit["gender"] = test_audit["gender"].astype("string")
    test_audit["age_band"] = test_audit["age_band"].astype("string")
    gender = _group_metrics(test_audit, "gender", recommended_predictions, scores)
    age = _group_metrics(test_audit, "age_band", recommended_predictions, scores)
    gender.to_csv(table_dir / "final_test_gender_metrics.csv", index=False)
    age.to_csv(table_dir / "final_test_age_metrics.csv", index=False)
    eligible_age_groups = age.loc[age["fraud_cases"] > 0, "group"]
    eligible_age_mask = test_audit["age_band"].isin(eligible_age_groups).to_numpy()
    fairness = {
        "gender": _fairness_summary(y_test, recommended_predictions, test_audit["gender"], gender),
        "age_band_eligible": _fairness_summary(
            y_test[eligible_age_mask],
            recommended_predictions[eligible_age_mask],
            test_audit.loc[eligible_age_mask, "age_band"],
            age[age["fraud_cases"] > 0],
        ),
        "excluded_age_groups": age.loc[age["fraud_cases"] == 0, "group"].tolist(),
        "age_exclusion_reason": "Fairlearn equalized-odds metrics are undefined for groups with zero positive labels.",
    }

    cost_table = _cost_sensitivity(capacity_rows, config["cost_sensitivity"])
    cost_table.to_csv(table_dir / "cost_sensitivity.csv", index=False)
    cost = config["cost_sensitivity"]
    base = cost_table[
        np.isclose(cost_table["review_capacity"], recommended_capacity)
        & np.isclose(cost_table["fraud_value_recovery_rate"], cost["base_recovery_rate"])
        & np.isclose(cost_table["review_cost_usd"], cost["base_review_cost_usd"])
        & np.isclose(
            cost_table["false_positive_friction_cost_usd"],
            cost["base_false_positive_friction_cost_usd"],
        )
    ].iloc[0].to_dict()
    sensitivity_at_recommended = cost_table[
        np.isclose(cost_table["review_capacity"], recommended_capacity)
    ]
    sensitivity_range = {
        "minimum_net_benefit_usd": float(sensitivity_at_recommended["net_benefit_usd"].min()),
        "maximum_net_benefit_usd": float(sensitivity_at_recommended["net_benefit_usd"].max()),
        "minimum_benefit_cost_ratio": float(sensitivity_at_recommended["benefit_cost_ratio"].min()),
        "maximum_benefit_cost_ratio": float(sensitivity_at_recommended["benefit_cost_ratio"].max()),
        "all_scenarios_positive": bool((sensitivity_at_recommended["net_benefit_usd"] > 0).all()),
    }

    pd.DataFrame(
        {
            "trans_date_trans_time": test["trans_date_trans_time"].astype(str),
            "is_fraud": y_test,
            "amt": test["amt"].to_numpy(dtype=float),
            "score": scores.astype(np.float32),
            "reviewed_at_1pct": recommended_predictions,
            "gender": test_audit["gender"],
            "age_band": test_audit["age_band"],
        }
    ).to_csv(table_dir / "final_test_scores.csv.gz", index=False, compression="gzip")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib", compress=3)
    joblib.dump(model, model_dir / f"{config['model_name']}.joblib", compress=3)
    (model_dir / "selected_features.json").write_text(
        json.dumps(selected_names, indent=2) + "\n", encoding="utf-8"
    )

    recommended_metrics = capacity_table[
        np.isclose(capacity_table["review_capacity"], recommended_capacity)
    ].iloc[0].to_dict()
    summary = {
        "protocol": {
            "model": config["model_name"],
            "hyperparameters_frozen_from_validation": True,
            "feature_list_frozen_from_development": True,
            "model_refit_on_complete_fraudTrain": True,
            "test_evaluated_once": True,
            "train_rows": len(train),
            "train_fraud_cases": int(y_train.sum()),
            "test_rows": len(test),
            "test_fraud_cases": int(y_test.sum()),
            "test_period_start": test["trans_date_trans_time"].min(),
            "test_period_end": test["trans_date_trans_time"].max(),
            "recommended_review_capacity": recommended_capacity,
            "recommended_test_score_cutoff": recommended_cutoff,
        },
        "ranking_metrics": {
            "average_precision": float(average_precision_score(y_test, scores)),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "brier_score": float(brier_score_loss(y_test, scores)),
            "log_loss": float(log_loss(y_test, np.clip(scores, 1e-12, 1 - 1e-12))),
        },
        "recommended_policy_metrics": recommended_metrics,
        "fairness_at_recommended_policy": fairness,
        "cost_scenario": {
            "currency": "USD",
            "comparison": "illustrative avoided-loss scenario versus no intervention",
            "base_assumptions": {
                "fraud_value_recovery_rate": cost["base_recovery_rate"],
                "review_cost_usd": cost["base_review_cost_usd"],
                "false_positive_friction_cost_usd": cost["base_false_positive_friction_cost_usd"],
            },
            "base_result": base,
            "sensitivity_at_1pct": sensitivity_range,
            "excluded_costs": [
                "model development and integration",
                "infrastructure and monitoring",
                "regulatory and governance review",
                "fraud recovery uncertainty beyond the scenario rate",
                "customer lifetime and reputational effects",
            ],
            "production_roi_claim": False,
        },
    }
    (output / "final_test_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary), indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
