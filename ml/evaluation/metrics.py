"""
ml/evaluation/metrics.py
─────────────────────────
Evaluation Metrics

Provides deterministic, unit-tested implementations of:
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error) — for reference only; not primary metric

All values are computed from ACTUAL predictions and actual demand values.
NEVER fabricate metric values.
"""

from typing import Union
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : Array of actual demand values.
    y_pred : Array of predicted demand values.

    Returns
    -------
    float — MAE in the same units as y (MW for this dataset).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : Array of actual demand values.
    y_pred : Array of predicted demand values.

    Returns
    -------
    float — RMSE in the same units as y (MW).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.

    Note: MAPE is numerically unstable when y_true is near zero.
    epsilon is added to avoid division by zero.

    Parameters
    ----------
    y_true   : Array of actual demand values.
    y_pred   : Array of predicted demand values.
    epsilon  : Small constant to avoid division by zero.

    Returns
    -------
    float — MAPE as a percentage (e.g., 5.0 means 5%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (Coefficient of Determination) regression score.

    1 - (SS_res / SS_tot). Best possible score is 1.0.

    Parameters
    ----------
    y_true : Array of actual values.
    y_pred : Array of predicted values.

    Returns
    -------
    float — R-squared score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def forecast_accuracy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Overall forecast accuracy defined as (100 - MAPE), clamped between 0% and 100%.

    Parameters
    ----------
    y_true  : Array of actual values.
    y_pred  : Array of predicted values.
    epsilon : Small constant for MAPE calculation.

    Returns
    -------
    float — Forecast accuracy percentage.
    """
    val = 100.0 - mape(y_true, y_pred, epsilon=epsilon)
    return float(np.clip(val, 0.0, 100.0))


def coverage_score(y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """
    Empirical prediction interval coverage probability (PICP).

    Measures the percentage of actual observations falling within [lower_bound, upper_bound].

    Parameters
    ----------
    y_true      : Actual values.
    lower_bound : Lower prediction bound (e.g. 95% lower PI).
    upper_bound : Upper prediction bound (e.g. 95% upper PI).

    Returns
    -------
    float — Empirical coverage percentage (e.g., 95.2%).
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower_bound, dtype=float)
    upper = np.asarray(upper_bound, dtype=float)
    if not (len(y_true) == len(lower) == len(upper)):
        raise ValueError("Length mismatch between y_true, lower_bound, and upper_bound")
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside) * 100.0)


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classification accuracy score (percentage of exact matches).

    Parameters
    ----------
    y_true : Ground truth labels.
    y_pred : Predicted labels.

    Returns
    -------
    float — Accuracy percentage between 0.0 and 100.0.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true={len(y_t)}, y_pred={len(y_p)}")
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p) * 100.0)


def confusion_matrix_anomalies(y_true: np.ndarray, y_pred: np.ndarray, pos_label: str = "ANOMALY") -> dict:
    """
    Compute binary confusion matrix components for anomaly detection.

    Parameters
    ----------
    y_true    : Ground truth labels (e.g. "ANOMALY", "NORMAL").
    y_pred    : Predicted labels.
    pos_label : The positive class label for anomalies.

    Returns
    -------
    dict with keys 'tp', 'fp', 'tn', 'fn'.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true={len(y_t)}, y_pred={len(y_p)}")

    is_true_pos = (y_t == pos_label)
    is_pred_pos = (y_p == pos_label)

    tp = int(np.sum(is_true_pos & is_pred_pos))
    fp = int(np.sum((~is_true_pos) & is_pred_pos))
    tn = int(np.sum((~is_true_pos) & (~is_pred_pos)))
    fn = int(np.sum(is_true_pos & (~is_pred_pos)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: str = "ANOMALY") -> float:
    """
    Precision = TP / (TP + FP). Clamped to 0.0 if denominator is 0.
    """
    cm = confusion_matrix_anomalies(y_true, y_pred, pos_label=pos_label)
    denom = cm["tp"] + cm["fp"]
    if denom == 0:
        return 0.0
    return float(cm["tp"] / denom)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: str = "ANOMALY") -> float:
    """
    Recall = TP / (TP + FN). Clamped to 0.0 if denominator is 0.
    """
    cm = confusion_matrix_anomalies(y_true, y_pred, pos_label=pos_label)
    denom = cm["tp"] + cm["fn"]
    if denom == 0:
        return 0.0
    return float(cm["tp"] / denom)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: str = "ANOMALY") -> float:
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall).
    """
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    if (prec + rec) == 0.0:
        return 0.0
    return float(2.0 * (prec * rec) / (prec + rec))

