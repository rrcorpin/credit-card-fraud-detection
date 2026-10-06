"""Run reproducible Step 3 EDA, feature engineering, selection, and PCA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT_DEFAULT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from src.features import (
    AUDIT_ONLY_COLUMNS,
    DIRECT_IDENTIFIERS_EXCLUDED,
    HIGH_CARDINALITY_MODEL_COLUMNS,
    LOW_CARDINALITY_MODEL_COLUMNS,
    NUMERIC_MODEL_COLUMNS,
    build_preprocessor,
    chronological_development_validation_split,
    engineer_features,
    load_raw_transactions,
)


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


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _enriched_sample(frame: pd.DataFrame, negative_count: int, seed: int) -> pd.DataFrame:
    positives = frame.loc[frame["is_fraud"] == 1]
    negatives = frame.loc[frame["is_fraud"] == 0]
    negative_count = min(negative_count, len(negatives))
    sampled_negatives = negatives.sample(n=negative_count, random_state=seed)
    return (
        pd.concat([positives, sampled_negatives], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _create_eda_figures(
    development: pd.DataFrame,
    figure_dir: Path,
    dpi: int,
    seed: int,
) -> dict[str, Any]:
    sns.set_theme(style="whitegrid", context="talk")
    palette = {0: "#4C78A8", 1: "#E45756"}

    class_counts = development["is_fraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Legitimate", "Fraud"], class_counts.values, color=[palette[0], palette[1]])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Severe class imbalance in the development period")
    for bar, value in zip(bars, class_counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:,}", ha="center")
    _save_figure(fig, figure_dir / "01_class_imbalance.png", dpi)

    monthly = (
        development.groupby("transaction_month", observed=True)["is_fraud"]
        .agg(transactions="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="transaction_month", y="fraud_rate", marker="o", ax=ax, color="#E45756")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("Transaction month")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Fraud prevalence changes over time")
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.2f}%")
    _save_figure(fig, figure_dir / "02_monthly_fraud_rate.png", dpi)

    amount_plot = _enriched_sample(development, negative_count=100_000, seed=seed)
    amount_plot = amount_plot.assign(log10_amount=np.log10(amount_plot["amt"]))
    amount_plot["class"] = amount_plot["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        data=amount_plot,
        x="log10_amount",
        hue="class",
        common_norm=False,
        palette={"Legitimate": palette[0], "Fraud": palette[1]},
        fill=False,
        ax=ax,
    )
    ax.set_xlabel("Transaction amount (log10 USD)")
    ax.set_title("Fraud and legitimate transactions have different amount profiles")
    _save_figure(fig, figure_dir / "03_amount_distribution_by_class.png", dpi)

    category = (
        development.groupby("category", observed=True)["is_fraud"]
        .agg(transactions="size", fraud_cases="sum", fraud_rate="mean")
        .sort_values("fraud_rate", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=category,
        y="category",
        x="fraud_rate",
        order=category["category"].astype(str).tolist(),
        color="#E45756",
        ax=ax,
    )
    ax.set_xlabel("Fraud rate")
    ax.set_ylabel("Merchant category")
    ax.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.1f}%")
    ax.set_title("Fraud prevalence varies substantially by merchant category")
    _save_figure(fig, figure_dir / "04_category_fraud_rate.png", dpi)

    temporal = development.pivot_table(
        index="day_of_week",
        columns="hour",
        values="is_fraud",
        aggfunc="mean",
        observed=True,
    ).reindex(index=range(7), columns=range(24))
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(100 * temporal, cmap="mako", ax=ax, cbar_kws={"label": "Fraud rate (%)"})
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title("Fraud rate by day and hour")
    _save_figure(fig, figure_dir / "05_day_hour_fraud_heatmap.png", dpi)

    relationship_sample = _enriched_sample(development, negative_count=40_000, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(
        data=relationship_sample,
        x="is_fraud",
        y="log_amt",
        hue="is_fraud",
        palette=palette,
        legend=False,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_xticks([0, 1], ["Legitimate", "Fraud"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("log(1 + amount)")
    axes[0].set_title("Transaction amount")
    sns.boxplot(
        data=relationship_sample,
        x="is_fraud",
        y="customer_merchant_distance_km",
        hue="is_fraud",
        palette=palette,
        legend=False,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_xticks([0, 1], ["Legitimate", "Fraud"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Customer–merchant distance (km)")
    axes[1].set_title("Geographic distance")
    fig.suptitle("Domain-derived relationships with the fraud label")
    _save_figure(fig, figure_dir / "06_domain_feature_relationships.png", dpi)

    gender = (
        development.groupby("gender", observed=True)["is_fraud"]
        .agg(transactions="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
    )
    age = (
        development.groupby("age_band", observed=True)["is_fraud"]
        .agg(transactions="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=gender, x="gender", y="fraud_rate", color="#72B7B2", ax=axes[0])
    sns.barplot(data=age, x="age_band", y="fraud_rate", color="#54A24B", ax=axes[1])
    for ax in axes:
        ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.2f}%")
        ax.set_ylabel("Observed fraud rate")
    axes[0].set_title("By reported gender")
    axes[1].set_title("By derived age band")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Preliminary subgroup prevalence (not a model fairness audit)")
    _save_figure(fig, figure_dir / "07_subgroup_prevalence.png", dpi)

    correlation_columns = NUMERIC_MODEL_COLUMNS + ["is_fraud"]
    correlation_sample = development[correlation_columns].sample(
        n=min(200_000, len(development)), random_state=seed
    )
    correlations = correlation_sample.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlations, cmap="vlag", center=0, ax=ax, square=False)
    ax.set_title("Spearman correlations among engineered numeric features")
    _save_figure(fig, figure_dir / "08_numeric_correlation_heatmap.png", dpi)

    return {
        "class_counts": class_counts.to_dict(),
        "monthly": monthly,
        "category": category,
        "gender": gender,
        "age": age,
        "correlations": correlations,
    }


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = json.loads(
        (project_root / "configs" / "step3_analysis.json").read_text(encoding="utf-8")
    )
    seed = int(config["random_seed"])
    np.random.seed(seed)

    output_dir = project_root / "deliverables" / "03_preprocessing_eda_features"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    raw_path = project_root / "data" / "raw" / "sparkov" / "fraudTrain.csv"
    raw = load_raw_transactions(raw_path)
    engineered = engineer_features(raw)
    split = chronological_development_validation_split(
        engineered, float(config["development_fraction"])
    )
    development = split.development
    validation = split.validation

    eda = _create_eda_figures(
        development, figure_dir, int(config["figure_dpi"]), seed
    )
    eda["monthly"].to_csv(table_dir / "monthly_fraud_rate.csv", index=False)
    eda["category"].to_csv(table_dir / "category_fraud_rate.csv", index=False)
    eda["gender"].to_csv(table_dir / "gender_prevalence.csv", index=False)
    eda["age"].to_csv(table_dir / "age_band_prevalence.csv", index=False)
    eda["correlations"].to_csv(table_dir / "numeric_spearman_correlations.csv")

    diagnostic = _enriched_sample(
        development, int(config["diagnostic_negative_sample"]), seed
    )
    preprocessor = build_preprocessor()
    preprocessor.fit(development)
    encoded = preprocessor.transform(diagnostic)
    feature_names = preprocessor.get_feature_names_out()
    target = diagnostic["is_fraud"].to_numpy()

    mi_count = min(int(config["mutual_information_sample"]), len(diagnostic))
    mi_positions = np.random.default_rng(seed).choice(len(diagnostic), size=mi_count, replace=False)
    mi_values = mutual_info_classif(
        encoded[mi_positions],
        target[mi_positions],
        random_state=seed,
        n_neighbors=3,
    )
    mutual_information = pd.DataFrame(
        {"feature": feature_names, "mutual_information": mi_values}
    ).sort_values("mutual_information", ascending=False)
    mutual_information.to_csv(table_dir / "mutual_information.csv", index=False)

    top_k = min(int(config["feature_selection_k"]), len(mutual_information))
    selected = mutual_information.head(top_k).copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))
    selected.to_csv(table_dir / "selected_features_filter_method.csv", index=False)

    mi_plot = mutual_information.head(20).copy()
    mi_plot["display_feature"] = (
        mi_plot["feature"]
        .str.replace(r"^(numeric|one_hot|frequency)__", "", regex=True)
        .str.replace("_", " ")
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(
        data=mi_plot,
        y="display_feature",
        x="mutual_information",
        color="#4C78A8",
        ax=ax,
    )
    ax.set_title("Filter selection: top mutual-information features")
    ax.set_xlabel("Mutual information with fraud label")
    ax.set_ylabel("")
    _save_figure(fig, figure_dir / "09_mutual_information.png", int(config["figure_dpi"]))

    diagnostic_model = ExtraTreesClassifier(
        n_estimators=int(config["extra_trees_estimators"]),
        max_depth=18,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    diagnostic_model.fit(encoded, target)
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": diagnostic_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(table_dir / "extra_trees_feature_importance.csv", index=False)
    importance_plot = importance.head(20).copy()
    importance_plot["display_feature"] = (
        importance_plot["feature"]
        .str.replace(r"^(numeric|one_hot|frequency)__", "", regex=True)
        .str.replace("_", " ")
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(
        data=importance_plot,
        y="display_feature",
        x="importance",
        color="#F58518",
        ax=ax,
    )
    ax.set_title("Diagnostic embedded importance from Extra Trees")
    ax.set_xlabel("Mean decrease in impurity importance")
    ax.set_ylabel("")
    _save_figure(fig, figure_dir / "10_model_based_feature_importance.png", int(config["figure_dpi"]))

    pca_fit_count = min(int(config["pca_fit_sample"]), len(development))
    pca_fit = development.sample(n=pca_fit_count, random_state=seed)
    pca_scaler = StandardScaler()
    pca_matrix = pca_scaler.fit_transform(pca_fit[NUMERIC_MODEL_COLUMNS])
    pca = PCA(random_state=seed)
    pca.fit(pca_matrix)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pca_table = pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    )
    pca_table.to_csv(table_dir / "pca_explained_variance.csv", index=False)

    pca_plot = _enriched_sample(
        development, int(config["pca_plot_negative_sample"]), seed
    )
    projected = pca.transform(pca_scaler.transform(pca_plot[NUMERIC_MODEL_COLUMNS]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(pca_table["component"], cumulative, marker="o", color="#4C78A8")
    for threshold in (0.80, 0.90, 0.95):
        axes[0].axhline(threshold, linestyle="--", linewidth=1, color="grey")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Principal components")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].set_title("PCA compression of numeric engineered features")
    scatter = axes[1].scatter(
        projected[:, 0],
        projected[:, 1],
        c=pca_plot["is_fraud"].map({0: "#4C78A8", 1: "#E45756"}),
        s=np.where(pca_plot["is_fraud"].to_numpy() == 1, 12, 4),
        alpha=np.where(pca_plot["is_fraud"].to_numpy() == 1, 0.65, 0.15),
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("PCA projection (fraud-enriched display sample)")
    _save_figure(fig, figure_dir / "11_pca_analysis.png", int(config["figure_dpi"]))

    thresholds = {}
    for threshold in (0.80, 0.90, 0.95):
        thresholds[str(threshold)] = int(np.searchsorted(cumulative, threshold) + 1)

    feature_manifest = {
        "excluded_direct_identifiers": DIRECT_IDENTIFIERS_EXCLUDED,
        "audit_only_columns": AUDIT_ONLY_COLUMNS,
        "numeric_model_columns": NUMERIC_MODEL_COLUMNS,
        "one_hot_columns": LOW_CARDINALITY_MODEL_COLUMNS,
        "frequency_encoded_columns": HIGH_CARDINALITY_MODEL_COLUMNS,
        "encoded_feature_count": len(feature_names),
        "filter_selected_feature_count": top_k,
        "filter_selected_features": selected["feature"].tolist(),
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "random_seed": seed,
        "raw_training_rows": len(raw),
        "development_fraction_requested": config["development_fraction"],
        "split_boundary_timestamp": split.boundary_timestamp,
        "development": {
            "rows": len(development),
            "start": development["trans_date_trans_time"].min(),
            "end": development["trans_date_trans_time"].max(),
            "fraud_cases": int(development["is_fraud"].sum()),
            "fraud_rate": float(development["is_fraud"].mean()),
        },
        "validation": {
            "rows": len(validation),
            "start": validation["trans_date_trans_time"].min(),
            "end": validation["trans_date_trans_time"].max(),
            "fraud_cases": int(validation["is_fraud"].sum()),
            "fraud_rate": float(validation["is_fraud"].mean()),
            "policy": "held out from Step 3 relationship EDA, selection, importance, and PCA",
        },
        "cleaning": {
            "missing_values_removed": 0,
            "duplicate_transaction_ids_removed": 0,
            "outlier_rows_removed": 0,
            "outlier_strategy": "retain; log-transform amount and city population; robust tree models evaluated later",
        },
        "diagnostic_sample": {
            "rows": len(diagnostic),
            "fraud_cases": int(diagnostic["is_fraud"].sum()),
            "purpose": "feature ranking and diagnostic importance only; not unbiased performance estimation",
        },
        "feature_selection": {
            "method": "mutual-information filter on an enriched development-only sample",
            "top_k": top_k,
            "top_features": selected.head(10).to_dict(orient="records"),
        },
        "model_based_importance": {
            "model": "ExtraTreesClassifier diagnostic model",
            "top_features": importance.head(10).to_dict(orient="records"),
            "limitation": "impurity importance is diagnostic and not final-model SHAP",
        },
        "pca": {
            "input_features": len(NUMERIC_MODEL_COLUMNS),
            "components_for_threshold": thresholds,
            "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
            "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
        },
    }
    summary = _json_ready(summary)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
