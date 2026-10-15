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


# ─── Spec fields, asymmetric intervals, validation ─────────────────────────────

def test_required_anomaly_fields_present():
    """timestamp, actual demand, lower, upper, deviation magnitude, classification."""
    test_df, forecast_df = make_test_forecast([100.0, 150.0, 50.0], [100.0] * 3, [80.0] * 3, [120.0] * 3)
    r = detect_anomalies(test_df, forecast_df, save=False)
    assert {"timestamp", "actual", "lower_bound", "upper_bound",
            "deviation_magnitude", "classification"} <= set(r.columns)
    assert list(r["classification"]) == ["NORMAL", "SPIKE_ANOMALY", "DROP_ANOMALY"]
    assert list(r["breach_magnitude"]) == [0.0, 30.0, 30.0]
    assert list(r["deviation_magnitude"]) == [0.0, 50.0, 50.0]


def test_sensitivity_scales_each_side_of_interval():
    test_df, forecast_df = make_test_forecast([100.0], [100.0], [90.0], [130.0])   # asymmetric
    r = detect_anomalies(test_df, forecast_df, sensitivity=2.0, save=False)
    assert r["lower_bound"].iloc[0] == pytest.approx(80.0)
    assert r["upper_bound"].iloc[0] == pytest.approx(160.0)


def test_sensitivity_boundary_flips_classification():
    test_df, forecast_df = make_test_forecast([125.0], [100.0], [80.0], [120.0])
    assert detect_anomalies(test_df, forecast_df, sensitivity=1.0, save=False)["classification"].iloc[0] == "SPIKE_ANOMALY"
    assert detect_anomalies(test_df, forecast_df, sensitivity=1.5, save=False)["classification"].iloc[0] == "NORMAL"


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_invalid_sensitivity_rejected(bad):
    test_df, forecast_df = make_test_forecast([100.0], [100.0], [80.0], [120.0])
    with pytest.raises(ValueError):
        detect_anomalies(test_df, forecast_df, sensitivity=bad, save=False)


def test_misaligned_forecast_rejected():
    test_df, forecast_df = make_test_forecast([100.0, 100.0], [100.0] * 2, [80.0] * 2, [120.0] * 2)
    with pytest.raises(ValueError):
        detect_anomalies(test_df, forecast_df.iloc[:1], save=False)
