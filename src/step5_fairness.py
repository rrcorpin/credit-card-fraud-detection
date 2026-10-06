"""Explain the selected model and audit subgroup fairness on validation data."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT_DEFAULT / ".cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(8, os.cpu_count() or 1)))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.features import (
    chronological_development_validation_split,
    engineer_features,
    load_raw_transactions,
)
from src.modeling import threshold_metrics, top_capacity_predictions


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


def _display_feature(name: str) -> str:
    replacements = {
        "numeric__": "",
        "one_hot__": "",
        "frequency__": "",
        "log_": "log ",
        "_": " ",
    }
    output = name
    for source, target in replacements.items():
        output = output.replace(source, target)
    return output.strip().capitalize()


def _transform_selected(preprocessor, frame, positions, chunk_size):
    output = np.empty((len(frame), len(positions)), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        transformed = preprocessor.transform(frame.iloc[start:stop])
        output[start:stop] = np.asarray(transformed[:, positions], dtype=np.float32)
    return output


def _score_in_chunks(preprocessor, model, frame, positions, chunk_size):
    scores = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        transformed = preprocessor.transform(frame.iloc[start:stop])
        selected = np.asarray(transformed[:, positions], dtype=np.float32)
        scores[start:stop] = model.predict_proba(selected)[:, 1]
    return scores


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _group_metrics(
    frame: pd.DataFrame,
    group_column: str,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    rows = []
    groups = frame[group_column].astype("string").fillna("Missing")
    target = frame["is_fraud"].to_numpy(dtype=np.int8)
    amounts = frame["amt"].to_numpy(dtype=float)
    for group in sorted(groups.unique().tolist()):
        mask = groups.eq(group).to_numpy()
        y_group = target[mask]
        pred_group = predictions[mask]
        amount_group = amounts[mask]
        tn = int(((y_group == 0) & (pred_group == 0)).sum())
        fp = int(((y_group == 0) & (pred_group == 1)).sum())
        fn = int(((y_group == 1) & (pred_group == 0)).sum())
        tp = int(((y_group == 1) & (pred_group == 1)).sum())
        selected = tp + fp
        fraud_value = float(amount_group[y_group == 1].sum())
        captured_value = float(amount_group[(y_group == 1) & (pred_group == 1)].sum())
        selection_ci = _wilson_interval(selected, len(y_group))
        tpr_ci = _wilson_interval(tp, tp + fn)
        fpr_ci = _wilson_interval(fp, fp + tn)
        precision_ci = _wilson_interval(tp, selected)
        rows.append(
            {
                "grouping": group_column,
                "group": str(group),
                "rows": int(len(y_group)),
                "fraud_cases": int((y_group == 1).sum()),
                "legitimate_cases": int((y_group == 0).sum()),
                "selected": selected,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "fraud_prevalence": float(y_group.mean()),
                "mean_score": float(scores[mask].mean()),
                "selection_rate": selected / len(y_group),
                "selection_rate_ci_low": selection_ci[0],
                "selection_rate_ci_high": selection_ci[1],
                "true_positive_rate": tp / (tp + fn) if tp + fn else float("nan"),
                "true_positive_rate_ci_low": tpr_ci[0],
                "true_positive_rate_ci_high": tpr_ci[1],
                "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
                "false_positive_rate_ci_low": fpr_ci[0],
                "false_positive_rate_ci_high": fpr_ci[1],
                "precision": tp / selected if selected else float("nan"),
                "precision_ci_low": precision_ci[0],
                "precision_ci_high": precision_ci[1],
                "fraud_value_recall": captured_value / fraud_value if fraud_value else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _fairness_summary(y_true, predictions, sensitive, group_table) -> dict[str, float]:
    return {
        "demographic_parity_difference": float(
            demographic_parity_difference(y_true, predictions, sensitive_features=sensitive)
        ),
        "demographic_parity_ratio_disparate_impact": float(
            demographic_parity_ratio(y_true, predictions, sensitive_features=sensitive)
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(y_true, predictions, sensitive_features=sensitive)
        ),
        "equalized_odds_ratio": float(
            equalized_odds_ratio(y_true, predictions, sensitive_features=sensitive)
        ),
        "true_positive_rate_gap": float(
            group_table["true_positive_rate"].max() - group_table["true_positive_rate"].min()
        ),
        "false_positive_rate_gap": float(
            group_table["false_positive_rate"].max() - group_table["false_positive_rate"].min()
        ),
        "precision_gap": float(group_table["precision"].max() - group_table["precision"].min()),
    }


def _equal_group_rate_predictions(scores, groups, capacity):
    """Select proportional group quotas while preserving the exact total capacity."""
    groups = pd.Series(groups, dtype="string").fillna("Missing").to_numpy()
    total_selected = max(1, int(np.ceil(len(scores) * capacity)))
    unique, counts = np.unique(groups, return_counts=True)
    raw_quotas = total_selected * counts / len(scores)
    quotas = np.floor(raw_quotas).astype(int)
    remainder = total_selected - int(quotas.sum())
    if remainder:
        order = np.argsort(-(raw_quotas - quotas), kind="mergesort")
        quotas[order[:remainder]] += 1
    predictions = np.zeros(len(scores), dtype=np.int8)
    cutoffs = {}
    for group, quota in zip(unique, quotas, strict=True):
        positions = np.flatnonzero(groups == group)
        ranked = positions[np.argsort(-scores[positions], kind="mergesort")]
        predictions[ranked[:quota]] = 1
        cutoffs[str(group)] = float(scores[ranked[quota - 1]]) if quota else float("nan")
    return predictions, cutoffs


def _equal_opportunity_predictions(scores, target, groups, capacity):
    """Exploratorily allocate a fixed review budget to minimize two-group TPR gap."""
    groups = pd.Series(groups, dtype="string").fillna("Missing").to_numpy()
    unique = np.unique(groups)
    if len(unique) != 2:
        raise ValueError("Equal-opportunity search currently requires exactly two groups")
    total_selected = max(1, int(np.ceil(len(scores) * capacity)))
    ranked = {}
    cumulative_positives = {}
    positive_totals = {}
    for group in unique:
        positions = np.flatnonzero(groups == group)
        ranked[group] = positions[np.argsort(-scores[positions], kind="mergesort")]
        cumulative_positives[group] = np.concatenate(
            [[0], np.cumsum(target[ranked[group]] == 1)]
        )
        positive_totals[group] = int((target[positions] == 1).sum())
    first, second = unique
    lower = max(0, total_selected - len(ranked[second]))
    upper = min(total_selected, len(ranked[first]))
    proportional_first = total_selected * len(ranked[first]) / len(scores)
    candidates = []
    for first_quota in range(lower, upper + 1):
        second_quota = total_selected - first_quota
        first_tpr = cumulative_positives[first][first_quota] / positive_totals[first]
        second_tpr = cumulative_positives[second][second_quota] / positive_totals[second]
        total_tp = (
            cumulative_positives[first][first_quota]
            + cumulative_positives[second][second_quota]
        )
        candidates.append(
            (
                abs(first_tpr - second_tpr),
                -int(total_tp),
                abs(first_quota - proportional_first),
                first_quota,
                second_quota,
            )
        )
    _, _, _, first_quota, second_quota = min(candidates)
    quotas = {first: first_quota, second: second_quota}
    predictions = np.zeros(len(scores), dtype=np.int8)
    cutoffs = {}
    for group in unique:
        quota = quotas[group]
        predictions[ranked[group][:quota]] = 1
        cutoffs[str(group)] = float(scores[ranked[group][quota - 1]]) if quota else float("nan")
    return predictions, cutoffs, {str(group): int(quota) for group, quota in quotas.items()}


def _save_explainability(
    model,
    X_validation,
    y_validation,
    scores,
    selected_names,
    predictions,
    audit_frame,
    config,
    figure_dir,
    table_dir,
):
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)
    positive_positions = np.flatnonzero(y_validation == 1)
    negative_positions = np.flatnonzero(y_validation == 0)
    positive_positions = rng.choice(
        positive_positions,
        size=min(int(config["shap_positive_sample"]), len(positive_positions)),
        replace=False,
    )
    negative_positions = rng.choice(
        negative_positions,
        size=min(int(config["shap_negative_sample"]), len(negative_positions)),
        replace=False,
    )
    explanation_positions = np.concatenate([positive_positions, negative_positions])
    explanation_positions = rng.permutation(explanation_positions)
    X_explain = X_validation[explanation_positions]
    display_names = [_display_feature(name) for name in selected_names]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    importance = pd.DataFrame(
        {
            "feature": selected_names,
            "display_feature": display_names,
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
            "mean_signed_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(table_dir / "shap_global_importance.csv", index=False)

    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=display_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP summary on an enriched validation explanation sample")
    plt.tight_layout()
    plt.savefig(figure_dir / "01_shap_summary.png", dpi=int(config["figure_dpi"]), bbox_inches="tight")
    plt.close()

    pdp_count = min(int(config["pdp_sample"]), len(X_validation))
    pdp_positions = rng.choice(len(X_validation), size=pdp_count, replace=False)
    X_pdp = X_validation[pdp_positions]
    numeric_features = [
        "numeric__log_amt",
        "numeric__log_amount_to_prior_mean_ratio",
        "numeric__hour_cos",
        "numeric__log_card_prior_mean_amt",
    ]
    feature_indices = [selected_names.index(name) for name in numeric_features]
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_pdp,
        features=feature_indices,
        feature_names=display_names,
        kind="average",
        grid_resolution=30,
        n_cols=2,
    )
    display.figure_.set_size_inches(13, 9)
    display.figure_.suptitle("Partial dependence of selected model features", y=1.02)
    display.figure_.savefig(
        figure_dir / "02_partial_dependence.png",
        dpi=int(config["figure_dpi"]),
        bbox_inches="tight",
    )
    plt.close(display.figure_)

    ice_feature = selected_names.index("numeric__log_amt")
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_pdp,
        features=[ice_feature],
        feature_names=display_names,
        kind="both",
        centered=True,
        subsample=int(config["ice_curves"]),
        random_state=seed,
        grid_resolution=40,
    )
    display.figure_.set_size_inches(10, 7)
    display.figure_.suptitle("ICE and average partial dependence for transaction amount", y=1.02)
    display.figure_.savefig(
        figure_dir / "03_ice_transaction_amount.png",
        dpi=int(config["figure_dpi"]),
        bbox_inches="tight",
    )
    plt.close(display.figure_)

    cases = {
        "true_positive": np.flatnonzero((y_validation == 1) & (predictions == 1)),
        "false_positive": np.flatnonzero((y_validation == 0) & (predictions == 1)),
        "false_negative": np.flatnonzero((y_validation == 1) & (predictions == 0)),
    }
    local_rows = []
    expected = np.asarray(explainer.expected_value).reshape(-1)[-1]
    for number, (case_type, positions) in enumerate(cases.items(), start=4):
        if case_type == "false_negative":
            position = positions[np.argmax(scores[positions])]
        else:
            median_score = np.median(scores[positions])
            position = positions[np.argmin(np.abs(scores[positions] - median_score))]
        local_values = explainer.shap_values(X_validation[position : position + 1])
        if isinstance(local_values, list):
            local_values = local_values[-1]
        local_values = np.asarray(local_values)
        if local_values.ndim == 3:
            local_values = local_values[:, :, -1]
        local_values = local_values.reshape(-1)
        explanation = shap.Explanation(
            values=local_values,
            base_values=expected,
            data=X_validation[position],
            feature_names=display_names,
        )
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title(f"Local SHAP explanation: {case_type.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(
            figure_dir / f"{number:02d}_local_shap_{case_type}.png",
            dpi=int(config["figure_dpi"]),
            bbox_inches="tight",
        )
        plt.close()
        ranked = np.argsort(-np.abs(local_values))[:10]
        for rank, feature_index in enumerate(ranked, start=1):
            local_rows.append(
                {
                    "case_type": case_type,
                    "validation_position": int(position),
                    "model_score": float(scores[position]),
                    "actual_label": int(y_validation[position]),
                    "predicted_label": int(predictions[position]),
                    "transaction_time": str(
                        audit_frame.iloc[position]["trans_date_trans_time"]
                    ),
                    "transaction_amount": float(audit_frame.iloc[position]["amt"]),
                    "merchant_category": str(audit_frame.iloc[position]["category"]),
                    "reported_gender": str(audit_frame.iloc[position]["gender"]),
                    "age_band": str(audit_frame.iloc[position]["age_band"]),
                    "rank": rank,
                    "feature": selected_names[feature_index],
                    "feature_value": float(X_validation[position, feature_index]),
                    "shap_value_log_odds": float(local_values[feature_index]),
                }
            )
    pd.DataFrame(local_rows).to_csv(table_dir / "local_shap_cases.csv", index=False)
    return importance


def _save_fairness_figures(gender, age, intersection, mitigation, temporal, figure_dir, dpi):
    sns.set_theme(style="whitegrid", context="talk")
    for name, data, filename in [
        ("Reported gender", gender, "07_gender_audit.png"),
        ("Derived age band", age, "08_age_audit.png"),
    ]:
        plot_data = data.copy()
        plot_data["display_group"] = (
            plot_data["group"].str.replace("_", "–").str.replace("plus", "+")
        )
        metrics = [
            ("true_positive_rate", "Fraud recall (TPR)", "#D28B62"),
            ("selection_rate", "Review selection rate", "#4C78A8"),
            ("false_positive_rate", "False-positive rate", "#59A14F"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        for ax, (metric, title, color) in zip(axes, metrics, strict=True):
            sns.barplot(data=plot_data, x="display_group", y=metric, color=color, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(name)
            ax.set_ylabel("Rate")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.tick_params(axis="x", rotation=40)
            if metric == "true_positive_rate":
                lower = max(0, float(plot_data[metric].min()) - 0.1)
                ax.set_ylim(lower, 1.0)
        fig.suptitle(f"Validation fairness audit by {name.lower()}", y=1.03)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    eligible = intersection[intersection["eligible_for_comparison"]].copy()
    eligible = eligible.sort_values("true_positive_rate")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=eligible,
        x="false_positive_rate",
        y="true_positive_rate",
        size="rows",
        hue="group",
        sizes=(80, 500),
        ax=ax,
    )
    ax.set_title("Eligible gender × age intersections: protection versus burden")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Fraud recall (TPR)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(figure_dir / "09_intersectional_audit.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    performance = mitigation.melt(
        id_vars="policy",
        value_vars=["overall_precision", "overall_recall"],
        var_name="metric",
        value_name="value",
    )
    performance["metric"] = performance["metric"].map(
        {"overall_precision": "Precision", "overall_recall": "Fraud recall"}
    )
    disparity = mitigation.melt(
        id_vars="policy",
        value_vars=["demographic_parity_difference", "equalized_odds_difference"],
        var_name="metric",
        value_name="value",
    )
    disparity["metric"] = disparity["metric"].map(
        {
            "demographic_parity_difference": "Demographic-parity gap",
            "equalized_odds_difference": "Equalized-odds gap",
        }
    )
    sns.barplot(data=performance, x="metric", y="value", hue="policy", ax=axes[0])
    sns.barplot(data=disparity, x="metric", y="value", hue="policy", ax=axes[1])
    axes[0].set_title("Overall performance (higher is better)")
    axes[1].set_title("Gender disparity (lower is better)")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=11)
    fig.suptitle("Gender post-processing scenarios at unchanged 1% workload", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "10_mitigation_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=temporal, x="month", y="average_precision", marker="o", ax=ax)
    ax.set_title("Validation ranking performance over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average precision (PR-AUC)")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(figure_dir / "11_temporal_robustness.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = json.loads((root / "configs" / "step5_fairness.json").read_text())
    seed = int(config["random_seed"])
    np.random.seed(seed)
    output = root / "deliverables" / "05_explainability_ethics_fairness"
    figure_dir, table_dir = output / "figures", output / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_transactions(root / "data" / "raw" / "sparkov" / "fraudTrain.csv")
    engineered = engineer_features(raw)
    split = chronological_development_validation_split(
        engineered, float(config["development_fraction"])
    )
    development, validation = split.development, split.validation
    model_dir = root / "models" / "step4"
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = joblib.load(model_dir / f"{config['model_name']}.joblib")
    selected_names = json.loads((model_dir / "selected_features.json").read_text())
    all_names = list(preprocessor.get_feature_names_out())
    positions = np.asarray([all_names.index(name) for name in selected_names])
    chunk_size = int(config["transform_chunk_size"])
    X_validation = _transform_selected(preprocessor, validation, positions, chunk_size)
    y_validation = validation["is_fraud"].to_numpy(dtype=np.int8)
    scores = model.predict_proba(X_validation)[:, 1]
    predictions, cutoff = top_capacity_predictions(
        scores, float(config["review_capacity"])
    )

    train_scores = _score_in_chunks(
        preprocessor, model, development, positions, chunk_size
    )
    train_pr_auc = average_precision_score(development["is_fraud"], train_scores)
    validation_pr_auc = average_precision_score(y_validation, scores)

    validation = validation.copy()
    validation["gender"] = validation["gender"].astype("string")
    validation["age_band"] = validation["age_band"].astype("string")
    validation["gender_age"] = validation["gender"] + " × " + validation["age_band"]
    population_edges = development["city_pop"].quantile([1 / 3, 2 / 3]).to_numpy()
    validation["city_population_segment"] = pd.cut(
        validation["city_pop"],
        bins=[-np.inf, population_edges[0], population_edges[1], np.inf],
        labels=["Lower population", "Middle population", "Higher population"],
        include_lowest=True,
    ).astype("string")

    gender = _group_metrics(validation, "gender", predictions, scores)
    age = _group_metrics(validation, "age_band", predictions, scores)
    population = _group_metrics(validation, "city_population_segment", predictions, scores)
    intersection = _group_metrics(validation, "gender_age", predictions, scores)
    intersection["eligible_for_comparison"] = (
        (intersection["rows"] >= int(config["minimum_intersection_rows"]))
        & (intersection["fraud_cases"] >= int(config["minimum_intersection_fraud_cases"]))
    )
    gender.to_csv(table_dir / "gender_group_metrics.csv", index=False)
    age.to_csv(table_dir / "age_group_metrics.csv", index=False)
    population.to_csv(table_dir / "city_population_segment_metrics.csv", index=False)
    intersection.to_csv(table_dir / "gender_age_intersection_metrics.csv", index=False)

    gender_summary = _fairness_summary(
        y_validation, predictions, validation["gender"], gender
    )
    age_summary = _fairness_summary(
        y_validation, predictions, validation["age_band"], age
    )
    population_summary = _fairness_summary(
        y_validation, predictions, validation["city_population_segment"], population
    )

    mitigated_predictions, group_cutoffs = _equal_group_rate_predictions(
        scores, validation["gender"], float(config["review_capacity"])
    )
    mitigated_gender = _group_metrics(validation, "gender", mitigated_predictions, scores)
    mitigated_summary = _fairness_summary(
        y_validation, mitigated_predictions, validation["gender"], mitigated_gender
    )
    amounts = validation["amt"].to_numpy(dtype=float)
    baseline_overall = threshold_metrics(y_validation, predictions, amounts)
    mitigated_overall = threshold_metrics(y_validation, mitigated_predictions, amounts)
    opportunity_predictions, opportunity_cutoffs, opportunity_quotas = _equal_opportunity_predictions(
        scores,
        y_validation,
        validation["gender"],
        float(config["review_capacity"]),
    )
    opportunity_gender = _group_metrics(
        validation, "gender", opportunity_predictions, scores
    )
    opportunity_summary = _fairness_summary(
        y_validation, opportunity_predictions, validation["gender"], opportunity_gender
    )
    opportunity_overall = threshold_metrics(y_validation, opportunity_predictions, amounts)
    mitigation = pd.DataFrame(
        [
            {
                "policy": "Global top 1%",
                "overall_precision": baseline_overall["precision"],
                "overall_recall": baseline_overall["recall"],
                "fraud_value_recall": baseline_overall["fraud_value_capture_rate"],
                **gender_summary,
            },
            {
                "policy": "Equal 1% within gender",
                "overall_precision": mitigated_overall["precision"],
                "overall_recall": mitigated_overall["recall"],
                "fraud_value_recall": mitigated_overall["fraud_value_capture_rate"],
                **mitigated_summary,
            },
            {
                "policy": "Exploratory equal-opportunity thresholds",
                "overall_precision": opportunity_overall["precision"],
                "overall_recall": opportunity_overall["recall"],
                "fraud_value_recall": opportunity_overall["fraud_value_capture_rate"],
                **opportunity_summary,
            },
        ]
    )
    mitigation.to_csv(table_dir / "gender_mitigation_comparison.csv", index=False)
    mitigated_gender.to_csv(table_dir / "gender_metrics_after_equal_rate.csv", index=False)
    opportunity_gender.to_csv(
        table_dir / "gender_metrics_after_equal_opportunity.csv", index=False
    )

    shap_importance = _save_explainability(
        model,
        X_validation,
        y_validation,
        scores,
        selected_names,
        predictions,
        validation,
        config,
        figure_dir,
        table_dir,
    )

    temporal_rows = []
    validation_month = validation["trans_date_trans_time"].dt.to_period("M").astype(str)
    for month in sorted(validation_month.unique()):
        mask = validation_month.eq(month).to_numpy()
        temporal_rows.append(
            {
                "month": month,
                "rows": int(mask.sum()),
                "fraud_cases": int(y_validation[mask].sum()),
                "fraud_prevalence": float(y_validation[mask].mean()),
                "average_precision": float(average_precision_score(y_validation[mask], scores[mask])),
                "recall_at_global_cutoff": float(
                    recall_score(y_validation[mask], predictions[mask], zero_division=0)
                ),
                "precision_at_global_cutoff": float(
                    precision_score(y_validation[mask], predictions[mask], zero_division=0)
                ),
            }
        )
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(table_dir / "temporal_robustness.csv", index=False)
    _save_fairness_figures(
        gender,
        age,
        intersection,
        mitigation,
        temporal,
        figure_dir,
        int(config["figure_dpi"]),
    )

    summary = {
        "protocol": {
            "model": config["model_name"],
            "audit_split": "chronological validation",
            "supplied_test_used": False,
            "review_capacity": float(config["review_capacity"]),
            "global_score_cutoff": cutoff,
            "validation_rows": len(validation),
            "validation_fraud_cases": int(y_validation.sum()),
            "race_audit_available": False,
            "socioeconomic_status_audit_available": False,
            "city_population_is_socioeconomic_status": False,
        },
        "overfitting_diagnostic": {
            "development_pr_auc": train_pr_auc,
            "validation_pr_auc": validation_pr_auc,
            "absolute_gap": train_pr_auc - validation_pr_auc,
        },
        "global_policy_metrics": baseline_overall,
        "fairness": {
            "gender": gender_summary,
            "age_band": age_summary,
            "city_population_context": population_summary,
        },
        "mitigation": {
            "equal_rate_scenario": {
                "method": "equal review rate within reported gender at unchanged total capacity",
                "group_score_cutoffs": group_cutoffs,
                "metrics": {**mitigated_overall, **mitigated_summary},
            },
            "equal_opportunity_scenario": {
                "method": "validation-tuned group thresholds minimizing the gender TPR gap at unchanged total capacity",
                "group_score_cutoffs": opportunity_cutoffs,
                "group_review_quotas": opportunity_quotas,
                "metrics": {**opportunity_overall, **opportunity_summary},
                "optimism_warning": "Thresholds were optimized and assessed on the same validation data; this is a diagnostic upper bound, not a deployable estimate.",
            },
            "baseline": {**baseline_overall, **gender_summary},
            "validation_labels_used_for_assessment_only": True,
        },
        "explainability": {
            "method": "TreeSHAP plus partial dependence and ICE",
            "shap_sample": int(config["shap_positive_sample"]) + int(config["shap_negative_sample"]),
            "top_features": shap_importance.head(10)[
                ["feature", "mean_absolute_shap"]
            ].to_dict(orient="records"),
            "shap_units": "model log-odds",
            "causal_interpretation": False,
        },
    }
    (output / "audit_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary["fairness"]), indent=2))
    print(json.dumps(_json_ready(summary["mitigation"]), indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
