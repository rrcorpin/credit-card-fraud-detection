"""Reusable model construction and rare-event evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


def build_model(specification: dict[str, Any], random_seed: int):
    """Instantiate a supported classifier from a JSON-compatible specification."""
    params = dict(specification)
    family = params.pop("family")
    if family == "dummy":
        return DummyClassifier(**params, random_state=random_seed)
    if family == "logistic_regression":
        return LogisticRegression(**params, random_state=random_seed)
    if family == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=random_seed)
    if family == "random_forest":
        return RandomForestClassifier(**params, random_state=random_seed)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=random_seed)
    raise ValueError(f"Unsupported model family: {family}")


def top_capacity_predictions(scores: np.ndarray, capacity: float) -> tuple[np.ndarray, float]:
    """Flag exactly the highest-scored capacity share, with deterministic ties."""
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    scores = np.asarray(scores, dtype=float)
    review_count = max(1, int(np.ceil(len(scores) * capacity)))
    ranked = np.argsort(-scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=np.int8)
    predictions[ranked[:review_count]] = 1
    return predictions, float(scores[ranked[review_count - 1]])


def threshold_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    amounts: np.ndarray,
) -> dict[str, float | int]:
    """Calculate classification and fraud-value metrics at a decision threshold."""
    y_true = np.asarray(y_true, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fraud_value = float(np.asarray(amounts)[y_true == 1].sum())
    captured_value = float(np.asarray(amounts)[(y_true == 1) & (predictions == 1)].sum())
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "review_rate": float(predictions.mean()),
        "fraud_value_capture_rate": captured_value / fraud_value if fraud_value else 0.0,
        "fraud_value_captured": captured_value,
        "fraud_value_total": fraud_value,
    }


def evaluate_probabilities(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    capacities: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate ranking, calibration, F1-optimal, and capacity-constrained behavior."""
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.clip(np.asarray(scores, dtype=float), 1e-12, 1 - 1e-12)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_position = int(np.nanargmax(f1_values))
    best_threshold = float(thresholds[best_position])
    best_predictions = (scores >= best_threshold).astype(np.int8)
    summary: dict[str, Any] = {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "f1_optimal_threshold": best_threshold,
        "f1_optimal": threshold_metrics(y_true, best_predictions, amounts),
        "threshold_0_5": threshold_metrics(y_true, (scores >= 0.5).astype(np.int8), amounts),
    }
    capacity_rows: list[dict[str, Any]] = []
    for capacity in capacities:
        predictions, threshold = top_capacity_predictions(scores, capacity)
        row = {
            "review_capacity": float(capacity),
            "score_cutoff": threshold,
            **threshold_metrics(y_true, predictions, amounts),
        }
        capacity_rows.append(row)
    return summary, capacity_rows
