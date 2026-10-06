"""
ml/tests/test_metrics.py
────────────────────────
Unit tests for regression and classification evaluation metrics.
"""

import numpy as np
import pytest

from ml.evaluation.metrics import (
    mae,
    rmse,
    mape,
    r2_score,
    forecast_accuracy,
    coverage_score,
    classification_accuracy,
    confusion_matrix_anomalies,
    precision_score,
    recall_score,
    f1_score,
)


def test_regression_metrics_perfect():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])

    assert mae(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0
    assert mape(y_true, y_pred) == 0.0
    assert r2_score(y_true, y_pred) == 1.0
    assert forecast_accuracy(y_true, y_pred) == 100.0


def test_r2_score_known_values():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    r2 = r2_score(y_true, y_pred)
    assert 0.90 < r2 < 1.0


def test_coverage_score():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([5.0, 15.0, 25.0, 50.0])  # 4th is outside (40 < 50)
    upper = np.array([15.0, 25.0, 35.0, 60.0])

    cov = coverage_score(y_true, lower, upper)
    assert cov == 75.0


def test_classification_metrics():
    y_true = np.array(["ANOMALY", "NORMAL", "ANOMALY", "NORMAL", "NORMAL"])
    y_pred = np.array(["ANOMALY", "NORMAL", "NORMAL", "NORMAL", "ANOMALY"])

    # True Positives: 1 (index 0)
    # False Positives: 1 (index 4)
    # True Negatives: 2 (indices 1, 3)
    # False Negatives: 1 (index 2)
    cm = confusion_matrix_anomalies(y_true, y_pred, pos_label="ANOMALY")
    assert cm == {"tp": 1, "fp": 1, "tn": 2, "fn": 1}

    acc = classification_accuracy(y_true, y_pred)
    assert acc == 60.0

    prec = precision_score(y_true, y_pred, pos_label="ANOMALY")
    assert prec == 0.5

    rec = recall_score(y_true, y_pred, pos_label="ANOMALY")
    assert rec == 0.5

    f1 = f1_score(y_true, y_pred, pos_label="ANOMALY")
    assert f1 == 0.5
