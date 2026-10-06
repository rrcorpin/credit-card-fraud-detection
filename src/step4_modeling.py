"""Train, compare, and persist Step 4 fraud-classification models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT_DEFAULT / ".cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(8, os.cpu_count() or 1)))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from src.features import (
    build_preprocessor,
    chronological_development_validation_split,
    engineer_features,
    load_raw_transactions,
)
from src.modeling import build_model, evaluate_probabilities, top_capacity_predictions


def _display_name(name: str) -> str:
    labels = {
        "dummy_prior": "Dummy prior",
        "logistic_c0_1": "Logistic regression (C=0.1)",
        "logistic_c1_0": "Logistic regression (C=1.0)",
        "decision_tree_depth8": "Decision tree (depth 8)",
        "decision_tree_depth12": "Decision tree (depth 12)",
        "random_forest": "Random forest",
        "hist_gradient_boosting_conservative": "Histogram boosting (conservative)",
        "hist_gradient_boosting_flexible": "Histogram boosting (flexible)",
    }
    return labels.get(name, name.replace("_", " "))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _transform_selected(
    preprocessor,
    frame: pd.DataFrame,
    selected_positions: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Transform in chunks to cap peak memory, retaining selected columns only."""
    output = np.empty((len(frame), len(selected_positions)), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        transformed = preprocessor.transform(frame.iloc[start:stop])
        output[start:stop] = np.asarray(transformed[:, selected_positions], dtype=np.float32)
    return output


def _save_curves(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    figure_dir: Path,
    dpi: int = 180,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, scores in predictions.items():
        PrecisionRecallDisplay.from_predictions(y_true, scores, name=_display_name(name), ax=ax)
    ax.set_title("Validation precision–recall curves")
    ax.set_ylim(0, 1.02)
    fig.savefig(figure_dir / "01_precision_recall_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, scores in predictions.items():
        RocCurveDisplay.from_predictions(y_true, scores, name=_display_name(name), ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Validation ROC curves")
    fig.savefig(figure_dir / "02_roc_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_comparison_figures(
    comparison: pd.DataFrame,
    capacity: pd.DataFrame,
    best_name: str,
    y_true: np.ndarray,
    best_scores: np.ndarray,
    figure_dir: Path,
    dpi: int = 180,
) -> None:
    ordered = comparison.sort_values("average_precision", ascending=True).copy()
    ordered["display_model"] = ordered["model"].map(_display_name)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=ordered, x="average_precision", y="display_model", color="#4C78A8", ax=ax)
    baseline = float(y_true.mean())
    ax.axvline(baseline, color="#E45756", linestyle="--", label=f"Prevalence baseline ({baseline:.3%})")
    ax.set_title("Primary model comparison on chronological validation data")
    ax.set_xlabel("Average precision (PR-AUC)")
    ax.set_ylabel("Model")
    ax.legend()
    fig.savefig(figure_dir / "03_model_pr_auc_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    one_percent = capacity[np.isclose(capacity["review_capacity"], 0.01)].copy()
    one_percent = one_percent.sort_values("recall", ascending=False)
    one_percent["display_model"] = one_percent["model"].map(_display_name)
    plot_data = one_percent.melt(
        id_vars="display_model",
        value_vars=["precision", "recall", "fraud_value_capture_rate"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_data["metric"] = plot_data["metric"].map(
        {
            "precision": "Precision",
            "recall": "Fraud recall",
            "fraud_value_capture_rate": "Fraud-value recall",
        }
    )
    sns.barplot(data=plot_data, y="display_model", x="value", hue="metric", ax=ax)
    ax.set_title("Performance when exactly 1% of transactions are reviewed")
    ax.set_xlabel("Metric value")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1)
    fig.savefig(figure_dir / "04_operational_comparison_at_1pct.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fraction_positive, mean_predicted = calibration_curve(
        y_true, best_scores, n_bins=10, strategy="uniform"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(mean_predicted, fraction_positive, marker="o", label=_display_name(best_name))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.set_title("Validation calibration diagnostic")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud fraction")
    ax.legend()
    fig.savefig(figure_dir / "05_best_model_calibration.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    predictions, _ = top_capacity_predictions(best_scores, 0.01)
    matrix = pd.crosstab(
        pd.Series(y_true, name="Actual"),
        pd.Series(predictions, name="Predicted"),
        dropna=False,
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticklabels(["Legitimate", "Fraud"], rotation=0)
    ax.set_title(f"{_display_name(best_name)}: confusion matrix at 1% review capacity")
    fig.savefig(figure_dir / "06_best_model_confusion_matrix_1pct.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = json.loads((project_root / "configs" / "step4_models.json").read_text())
    seed = int(config["random_seed"])
    np.random.seed(seed)

    output_dir = project_root / "deliverables" / "04_model_implementation"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    model_dir = project_root / "models" / "step4"
    for directory in (output_dir, table_dir, figure_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_raw_transactions(project_root / "data" / "raw" / "sparkov" / "fraudTrain.csv")
    engineered = engineer_features(raw)
    split = chronological_development_validation_split(
        engineered, float(config["development_fraction"])
    )
    development, validation = split.development, split.validation

    preprocessor = build_preprocessor()
    preprocessor.fit(development)
    all_feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    manifest = json.loads(
        (project_root / "deliverables" / "03_preprocessing_eda_features" / "feature_manifest.json").read_text()
    )
    selected_names = manifest["filter_selected_features"][: int(config["selected_feature_count"])]
    feature_lookup = {name: index for index, name in enumerate(all_feature_names)}
    missing = [name for name in selected_names if name not in feature_lookup]
    if missing:
        raise RuntimeError(f"Selected features missing after preprocessing: {missing}")
    selected_positions = np.asarray([feature_lookup[name] for name in selected_names])
    chunk_size = int(config["transform_chunk_size"])
    X_development = _transform_selected(preprocessor, development, selected_positions, chunk_size)
    X_validation = _transform_selected(preprocessor, validation, selected_positions, chunk_size)
    y_development = development["is_fraud"].to_numpy(dtype=np.int8)
    y_validation = validation["is_fraud"].to_numpy(dtype=np.int8)
    validation_amounts = validation["amt"].to_numpy(dtype=float)

    joblib.dump(preprocessor, model_dir / "preprocessor.joblib", compress=3)
    (model_dir / "selected_features.json").write_text(
        json.dumps(selected_names, indent=2) + "\n", encoding="utf-8"
    )

    comparison_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    detailed: dict[str, Any] = {}
    for name, specification in config["models"].items():
        model = build_model(specification, seed)
        started = time.perf_counter()
        model.fit(X_development, y_development)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = model.predict_proba(X_validation)[:, 1]
        score_seconds = time.perf_counter() - started
        metrics, model_capacity_rows = evaluate_probabilities(
            y_validation,
            scores,
            validation_amounts,
            [float(value) for value in config["review_capacities"]],
        )
        comparison_rows.append(
            {
                "model": name,
                "family": specification["family"],
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "f1_optimal_threshold": metrics["f1_optimal_threshold"],
                "f1_optimal_score": metrics["f1_optimal"]["f1"],
                "fit_seconds": fit_seconds,
                "score_seconds": score_seconds,
            }
        )
        for row in model_capacity_rows:
            capacity_rows.append({"model": name, **row})
        predictions_by_model[name] = scores.astype(np.float32)
        detailed[name] = metrics
        joblib.dump(model, model_dir / f"{name}.joblib", compress=3)

    comparison = pd.DataFrame(comparison_rows).sort_values("average_precision", ascending=False)
    capacities = pd.DataFrame(capacity_rows)
    best_name = str(comparison.iloc[0]["model"])
    operating_capacity = float(config["operating_review_capacity"])
    operating = capacities[
        (capacities["model"] == best_name)
        & np.isclose(capacities["review_capacity"], operating_capacity)
    ].iloc[0].to_dict()

    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    capacities.to_csv(table_dir / "capacity_metrics.csv", index=False)
    pd.DataFrame(
        {
            "trans_date_trans_time": validation["trans_date_trans_time"].astype(str),
            "is_fraud": y_validation,
            "amt": validation_amounts,
            "score": predictions_by_model[best_name],
        }
    ).to_csv(table_dir / "best_model_validation_scores.csv.gz", index=False, compression="gzip")

    artifact_rows = []
    for artifact in sorted(model_dir.glob("*")):
        if artifact.is_file():
            artifact_rows.append(
                {
                    "artifact": artifact.name,
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            )
    pd.DataFrame(artifact_rows).to_csv(table_dir / "model_artifact_manifest.csv", index=False)

    display_predictions = {
        name: predictions_by_model[name]
        for name in comparison.groupby("family", sort=False).head(1)["model"]
    }
    _save_curves(y_validation, display_predictions, figure_dir)
    _save_comparison_figures(
        comparison,
        capacities,
        best_name,
        y_validation,
        predictions_by_model[best_name],
        figure_dir,
    )

    summary = {
        "protocol": {
            "target": "is_fraud",
            "primary_metric": config["primary_metric"],
            "selection_split": "chronological validation carved from fraudTrain.csv",
            "supplied_test_used": False,
            "development_rows": len(development),
            "development_fraud_cases": int(y_development.sum()),
            "development_start": development["trans_date_trans_time"].min(),
            "development_end": development["trans_date_trans_time"].max(),
            "validation_rows": len(validation),
            "validation_fraud_cases": int(y_validation.sum()),
            "validation_start": validation["trans_date_trans_time"].min(),
            "validation_end": validation["trans_date_trans_time"].max(),
            "selected_feature_count": len(selected_names),
        },
        "selection": {
            "selected_model": best_name,
            "selection_reason": "highest validation average precision",
            "operating_capacity_scenario": operating_capacity,
            "operating_metrics": operating,
            "probability_calibration_required_before_probability_interpretation": True,
        },
        "models": detailed,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary["selection"]), indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
