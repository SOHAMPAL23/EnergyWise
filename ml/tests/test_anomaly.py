"""
ml/tests/test_anomaly.py
─────────────────────────
Anomaly Detection Tests

Tests:
  - NORMAL classification (actual within bounds)
  - SPIKE_ANOMALY classification (actual > upper_bound)
  - DROP_ANOMALY classification (actual < lower_bound)
  - Boundary observations (exactly at bounds)
  - Sensitivity change effect on alert rate
"""

import pytest
import numpy as np
import pandas as pd

from ml.anomaly.detector import detect_anomalies
from ml.configs.config import DS_COL, Y_COL


def make_test_forecast(actuals, yhat, lower, upper):
    """Helper to build test/forecast DataFrames."""
    n = len(actuals)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    test_df = pd.DataFrame({DS_COL: dates, Y_COL: actuals})
    forecast_df = pd.DataFrame({
        DS_COL: dates,
        "yhat": yhat,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })
    return test_df, forecast_df


# ─── Normal observation ────────────────────────────────────────────────────────

def test_normal_classification():
    actual = [100.0]
    yhat = [100.0]
    lower = [80.0]
    upper = [120.0]
    test_df, forecast_df = make_test_forecast(actual, yhat, lower, upper)
    result = detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)
    assert result["classification"].iloc[0] == "NORMAL"


# ─── Spike anomaly ─────────────────────────────────────────────────────────────

def test_spike_anomaly():
    actual = [150.0]
    yhat = [100.0]
    lower = [80.0]
    upper = [120.0]
    test_df, forecast_df = make_test_forecast(actual, yhat, lower, upper)
    result = detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)
    assert result["classification"].iloc[0] == "SPIKE_ANOMALY"


# ─── Drop anomaly ──────────────────────────────────────────────────────────────

def test_drop_anomaly():
    actual = [50.0]
    yhat = [100.0]
    lower = [80.0]
    upper = [120.0]
    test_df, forecast_df = make_test_forecast(actual, yhat, lower, upper)
    result = detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)
    assert result["classification"].iloc[0] == "DROP_ANOMALY"


# ─── Boundary observations ─────────────────────────────────────────────────────

def test_exactly_at_lower_bound_is_normal():
    """Exactly at lower bound = NORMAL (not anomaly)."""
    actual = [80.0]
    yhat = [100.0]
    lower = [80.0]
    upper = [120.0]
    test_df, forecast_df = make_test_forecast(actual, yhat, lower, upper)
    result = detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)
    assert result["classification"].iloc[0] == "NORMAL"


def test_exactly_at_upper_bound_is_normal():
    """Exactly at upper bound = NORMAL (not anomaly)."""
    actual = [120.0]
    yhat = [100.0]
    lower = [80.0]
    upper = [120.0]
    test_df, forecast_df = make_test_forecast(actual, yhat, lower, upper)
    result = detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)
    assert result["classification"].iloc[0] == "NORMAL"


# ─── Sensitivity change tests ──────────────────────────────────────────────────

def test_higher_sensitivity_fewer_anomalies():
    """Wider bounds (sensitivity > 1.0) should produce fewer anomalies."""
    n = 50
    actuals = np.random.uniform(50, 150, n)
    yhat = np.full(n, 100.0)
    lower = np.full(n, 80.0)
    upper = np.full(n, 120.0)

    test_df, forecast_df = make_test_forecast(actuals, yhat, lower, upper)

    result_narrow = detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)
    result_wide = detect_anomalies(test_df, forecast_df, sensitivity=2.0, save=False)

    n_anomalies_narrow = (result_narrow["classification"] != "NORMAL").sum()
    n_anomalies_wide = (result_wide["classification"] != "NORMAL").sum()

    assert n_anomalies_wide <= n_anomalies_narrow, \
        "Higher sensitivity should produce equal or fewer anomalies"


def test_lower_sensitivity_more_anomalies():
    """Narrower bounds (sensitivity < 1.0) should produce more anomalies."""
    n = 50
    np.random.seed(42)
    actuals = np.random.uniform(50, 150, n)
    yhat = np.full(n, 100.0)
    lower = np.full(n, 80.0)
    upper = np.full(n, 120.0)

    test_df, forecast_df = make_test_forecast(actuals, yhat, lower, upper)

    result_normal = detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)
    result_sensitive = detect_anomalies(test_df, forecast_df, sensitivity=0.5, save=False)

    n_anomalies_normal = (result_normal["classification"] != "NORMAL").sum()
    n_anomalies_sensitive = (result_sensitive["classification"] != "NORMAL").sum()

    assert n_anomalies_sensitive >= n_anomalies_normal, \
        "Lower sensitivity should produce equal or more anomalies"


# ─── Output schema tests ───────────────────────────────────────────────────────

def test_anomaly_output_columns():
    actual = [100.0, 150.0, 50.0]
    yhat = [100.0, 100.0, 100.0]
    lower = [80.0, 80.0, 80.0]
    upper = [120.0, 120.0, 120.0]
    test_df, forecast_df = make_test_forecast(actual, yhat, lower, upper)
    result = detect_anomalies(test_df, forecast_df, save=False)
    required = {"ds", "actual", "yhat", "lower_bound", "upper_bound",
                "deviation", "deviation_magnitude", "classification"}
    assert required.issubset(set(result.columns))


def test_anomaly_deviation_correct():
    actual = [130.0]
    yhat = [100.0]
    lower = [80.0]
    upper = [120.0]
    test_df, forecast_df = make_test_forecast(actual, yhat, lower, upper)
    result = detect_anomalies(test_df, forecast_df, save=False)
    assert result["deviation"].iloc[0] == pytest.approx(30.0)
    assert result["deviation_magnitude"].iloc[0] == pytest.approx(30.0)
