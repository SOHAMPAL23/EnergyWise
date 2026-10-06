"""
ml/tests/test_baselines.py
────────────────────────────
Baseline Model Tests

Tests:
  - Naive prediction (constant output = last training value)
  - Seasonal-Naive prediction (seasonal lag lookup)
  - MAE computation
  - RMSE computation
"""

import pytest
import numpy as np
import pandas as pd

from ml.baselines.naive import NaiveForecaster
from ml.baselines.seasonal_naive import SeasonalNaiveForecaster
from ml.evaluation.metrics import mae, rmse
from ml.configs.config import DS_COL, Y_COL


@pytest.fixture
def train_df():
    dates = pd.date_range("2015-01-01", periods=100, freq="D")
    vals = np.arange(100, dtype=float) * 100 + 40000  # monotonically increasing
    return pd.DataFrame({DS_COL: dates, Y_COL: vals})


@pytest.fixture
def test_df():
    dates = pd.date_range("2015-04-12", periods=30, freq="D")
    vals = np.full(30, 48000.0)
    return pd.DataFrame({DS_COL: dates, Y_COL: vals})


# ─── Naive forecaster ──────────────────────────────────────────────────────────

def test_naive_fit_stores_last_value(train_df):
    model = NaiveForecaster()
    model.fit(train_df)
    # Last value is 40000 + 99 * 100 = 49900
    assert model.last_value == pytest.approx(49900.0)


def test_naive_predict_constant(train_df, test_df):
    model = NaiveForecaster().fit(train_df)
    preds = model.predict(test_df)
    assert preds["yhat"].nunique() == 1
    assert preds["yhat"].iloc[0] == pytest.approx(train_df[Y_COL].iloc[-1])


def test_naive_predict_length(train_df, test_df):
    model = NaiveForecaster().fit(train_df)
    preds = model.predict(test_df)
    assert len(preds) == len(test_df)


def test_naive_not_fitted_raises(test_df):
    model = NaiveForecaster()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(test_df)


def test_naive_empty_train_raises():
    df = pd.DataFrame({DS_COL: [], Y_COL: []})
    with pytest.raises(ValueError, match="empty"):
        NaiveForecaster().fit(df)


# ─── Seasonal-Naive forecaster ─────────────────────────────────────────────────

def test_snaive_fit_stores_values(train_df):
    model = SeasonalNaiveForecaster(period=7)
    model.fit(train_df)
    assert model._train_values is not None
    assert len(model._train_values) == len(train_df)


def test_snaive_predict_length(train_df, test_df):
    model = SeasonalNaiveForecaster(period=7).fit(train_df)
    preds = model.predict(test_df)
    assert len(preds) == len(test_df)


def test_snaive_seasonal_cycle():
    """Predictions should repeat the last full seasonal cycle."""
    train_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 5, dtype=float)
    train = pd.DataFrame({
        DS_COL: pd.date_range("2015-01-01", periods=len(train_vals), freq="D"),
        Y_COL: train_vals,
    })
    test = pd.DataFrame({
        DS_COL: pd.date_range("2015-04-12", periods=7, freq="D"),
        Y_COL: np.zeros(7),
    })
    model = SeasonalNaiveForecaster(period=7).fit(train)
    preds = model.predict(test)
    # Should repeat values 1..7 (last full seasonal cycle)
    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    np.testing.assert_array_almost_equal(preds["yhat"].values, expected)


def test_snaive_invalid_period():
    with pytest.raises(ValueError, match="Period must be"):
        SeasonalNaiveForecaster(period=0)


# ─── Metrics tests ─────────────────────────────────────────────────────────────

def test_mae_zero_error():
    y = np.array([100.0, 200.0, 300.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    assert mae(y_true, y_pred) == pytest.approx(10.0)


def test_rmse_zero_error():
    y = np.array([100.0, 200.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # sqrt((9 + 16) / 2) = sqrt(12.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_mae_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_rmse_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
