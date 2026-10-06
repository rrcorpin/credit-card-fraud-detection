"""Reviewer-driven Step 4 robustness experiments on chronological development data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(8, os.cpu_count() or 1)))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler

from src.features import build_preprocessor, chronological_development_validation_split, engineer_features, load_raw_transactions
from src.modeling import threshold_metrics, top_capacity_predictions
from src.step5_fairness import _fairness_summary, _group_metrics


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _transform(preprocessor, frame, positions, chunk_size):
    output = np.empty((len(frame), len(positions)), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        block = preprocessor.transform(frame.iloc[start:stop])[:, positions]
        if hasattr(block, "toarray"):
            block = block.toarray()
        output[start:stop] = np.asarray(block, dtype=np.float32)
    return output


def _score(y_true, scores):
    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    return {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "brier_score": brier_score_loss(y_true, clipped),
        "log_loss": log_loss(y_true, clipped, labels=[0, 1]),
    }


def _build(family, params, seed):
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            **params, class_weight="balanced", early_stopping=False, random_state=seed
        )
    return RandomForestClassifier(
        **params, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
    )


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = json.loads((root / "configs" / "step4_enhancements.json").read_text())
    seed = int(config["random_seed"])
    output = root / "deliverables" / "04_model_implementation" / "enhancements"
    tables = output / "tables"
    models = root / "models" / "step4_enhancements"
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    raw = load_raw_transactions(root / "data" / "raw" / "sparkov" / "fraudTrain.csv")
    engineered = engineer_features(raw)
    outer = chronological_development_validation_split(engineered, float(config["outer_development_fraction"]))
    inner = chronological_development_validation_split(outer.development, float(config["inner_training_fraction"]))
    inner_train, inner_tune, outer_validation = inner.development, inner.validation, outer.validation

    preprocessor = build_preprocessor()
    preprocessor.fit(inner_train)
    all_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    selected_names = json.loads((root / "models" / "step4" / "selected_features.json").read_text())
    lookup = {name: i for i, name in enumerate(all_names)}
    selected_positions = np.asarray([lookup[name] for name in selected_names])
    chunk_size = int(config["transform_chunk_size"])
    X_inner_train = _transform(preprocessor, inner_train, selected_positions, chunk_size)
    X_inner_tune = _transform(preprocessor, inner_tune, selected_positions, chunk_size)
    X_outer_inner_prep = _transform(preprocessor, outer_validation, selected_positions, chunk_size)
    y_inner_train = inner_train["is_fraud"].to_numpy(dtype=np.int8)
    y_inner_tune = inner_tune["is_fraud"].to_numpy(dtype=np.int8)
    y_outer = outer_validation["is_fraud"].to_numpy(dtype=np.int8)

    search_rows = []
    best_by_family = {}
    for family, space_key in (("hist_gradient_boosting", "hist_gradient_boosting_space"), ("random_forest", "random_forest_space")):
        candidates = list(ParameterSampler(config[space_key], n_iter=int(config["search_iterations"][family]), random_state=seed))
        if family == "hist_gradient_boosting":
            candidates.append({"learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 63, "min_samples_leaf": 25, "l2_regularization": 0.5})
        for candidate_id, params in enumerate(candidates, 1):
            model = _build(family, params, seed)
            started = time.perf_counter()
            model.fit(X_inner_train, y_inner_train)
            scores = model.predict_proba(X_inner_tune)[:, 1]
            search_rows.append({"family": family, "candidate": candidate_id, **params, **_score(y_inner_tune, scores), "fit_seconds": time.perf_counter() - started})
    search = pd.DataFrame(search_rows).sort_values(["family", "average_precision"], ascending=[True, False])
    search.to_csv(tables / "chronological_randomized_search.csv", index=False)
    for family in search["family"].unique():
        row = search[search["family"].eq(family)].iloc[0]
        keys = config[f"{family}_space"].keys()
        best_by_family[family] = {key: row[key].item() if hasattr(row[key], "item") else row[key] for key in keys}
        for key in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "n_estimators", "max_depth"):
            if key in best_by_family[family]:
                best_by_family[family][key] = int(best_by_family[family][key])

    outer_preprocessor = build_preprocessor()
    outer_preprocessor.fit(outer.development)
    outer_names = np.asarray(outer_preprocessor.get_feature_names_out(), dtype=object)
    outer_lookup = {name: i for i, name in enumerate(outer_names)}
    outer_selected = np.asarray([outer_lookup[name] for name in selected_names])
    all_positions = np.arange(len(outer_names))
    X_dev_selected = _transform(outer_preprocessor, outer.development, outer_selected, chunk_size)
    X_val_selected = _transform(outer_preprocessor, outer_validation, outer_selected, chunk_size)
    X_dev_full = _transform(outer_preprocessor, outer.development, all_positions, chunk_size)
    X_val_full = _transform(outer_preprocessor, outer_validation, all_positions, chunk_size)
    y_dev = outer.development["is_fraud"].to_numpy(dtype=np.int8)

    best_hgb = best_by_family["hist_gradient_boosting"]
    representation_rows = []
    for representation, X_train, X_valid in (("top_25_mutual_information", X_dev_selected, X_val_selected), (f"full_{len(outer_names)}_engineered", X_dev_full, X_val_full)):
        model = _build("hist_gradient_boosting", best_hgb, seed)
        started = time.perf_counter()
        model.fit(X_train, y_dev)
        scores = model.predict_proba(X_valid)[:, 1]
        predictions, cutoff = top_capacity_predictions(scores, float(config["review_capacity"]))
        representation_rows.append({"representation": representation, "feature_count": X_train.shape[1], **_score(y_outer, scores), **{f"at_1pct_{k}": v for k, v in threshold_metrics(y_outer, predictions, outer_validation["amt"].to_numpy(dtype=float)).items()}, "at_1pct_score_cutoff": cutoff, "fit_seconds": time.perf_counter() - started})
        joblib.dump(model, models / f"hgb_{representation}.joblib", compress=3)
    representations = pd.DataFrame(representation_rows).sort_values("average_precision", ascending=False)
    representations.to_csv(tables / "representation_comparison.csv", index=False)

    family_rows = []
    for family, params in best_by_family.items():
        model = _build(family, params, seed)
        started = time.perf_counter()
        model.fit(X_dev_selected, y_dev)
        scores = model.predict_proba(X_val_selected)[:, 1]
        family_rows.append({"family": family, **params, **_score(y_outer, scores), "fit_seconds": time.perf_counter() - started})
    pd.DataFrame(family_rows).sort_values("average_precision", ascending=False).to_csv(tables / "tuned_family_outer_validation.csv", index=False)

    calibration_base = _build("hist_gradient_boosting", best_hgb, seed)
    calibration_base.fit(X_inner_train, y_inner_train)
    tune_raw = calibration_base.predict_proba(X_inner_tune)[:, 1]
    outer_raw = calibration_base.predict_proba(X_outer_inner_prep)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(tune_raw, y_inner_tune)
    outer_calibrated = calibrator.predict(outer_raw)
    calibration = pd.DataFrame([{"score_type": "raw_class_weighted_score", **_score(y_outer, outer_raw)}, {"score_type": "isotonic_recalibrated_probability", **_score(y_outer, outer_calibrated)}])
    calibration.to_csv(tables / "calibration_comparison.csv", index=False)
    joblib.dump(calibration_base, models / "calibration_base_model.joblib", compress=3)
    joblib.dump(calibrator, models / "isotonic_calibrator.joblib", compress=3)
    joblib.dump(preprocessor, models / "inner_preprocessor.joblib", compress=3)

    audit = json.loads((root / "deliverables" / "05_explainability_ethics_fairness" / "audit_summary.json").read_text())
    cutoffs = audit["mitigation"]["equal_opportunity_scenario"]["group_score_cutoffs"]
    final_scores = pd.read_csv(root / "deliverables" / "04_model_implementation" / "final_test" / "tables" / "final_test_scores.csv.gz")
    test_raw = load_raw_transactions(root / "data" / "raw" / "sparkov" / "fraudTest.csv")
    if len(test_raw) != len(final_scores):
        raise RuntimeError("Final test scores do not align with raw test rows.")
    genders = test_raw["gender"].astype("string")
    test_scores = final_scores["score"].to_numpy(dtype=float)
    test_y = final_scores["is_fraud"].to_numpy(dtype=np.int8)
    test_amounts = final_scores["amt"].to_numpy(dtype=float)
    mitigation_predictions = np.asarray([score >= float(cutoffs[str(group)]) for score, group in zip(test_scores, genders, strict=True)], dtype=np.int8)
    global_predictions, global_cutoff = top_capacity_predictions(test_scores, float(config["review_capacity"]))
    mitigation_rows = []
    audit_frame = pd.DataFrame({"gender": genders, "is_fraud": test_y, "amt": test_amounts})
    for policy, predictions in (("Global top 1% on test scores", global_predictions), ("Validation-derived equal-opportunity cutoffs", mitigation_predictions)):
        grouped = _group_metrics(audit_frame, "gender", predictions, test_scores)
        mitigation_rows.append({"policy": policy, **threshold_metrics(test_y, predictions, test_amounts), **_fairness_summary(test_y, predictions, genders, grouped)})
    mitigation_test = pd.DataFrame(mitigation_rows)
    mitigation_test.to_csv(tables / "fairness_mitigation_test_transfer.csv", index=False)

    summary = {
        "protocol": {"inner_train_rows": len(inner_train), "inner_tuning_rows": len(inner_tune), "outer_validation_rows": len(outer_validation), "random_search_uses_outer_validation": False, "early_stopping": False, "test_labels_used_for_model_or_calibration_selection": False},
        "best_parameters": best_by_family,
        "representation_winner": representations.iloc[0].to_dict(),
        "calibration": calibration.to_dict(orient="records"),
        "fairness_test_transfer": mitigation_test.to_dict(orient="records"),
        "global_test_score_cutoff_is_period_relative": global_cutoff,
        "pca_decision": "Rejected before Step 4: 13/20 components were needed for 95% numeric variance, PC1-PC2 had substantial class overlap, and PCA would discard interpretable mixed-type signals. PCA remains a visualization diagnostic, not a model representation."
    }
    (output / "enhancement_summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    print(json.dumps(_json_ready(summary), indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
