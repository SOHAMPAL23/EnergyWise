"""
ml/tests/test_reproducibility.py
──────────────────────────────────
Reproducibility Tests

Tests:
  - Fixed split produces same result across runs
  - Fixed synthetic model + input produces deterministic forecast output
  - Model load/save round-trip preserves forecast
"""

import pytest
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from ml.preprocessing.splitting import chronological_train_test_split
from ml.prophet.daily import train_daily_prophet, generate_forecast
from ml.configs.config import DS_COL, Y_COL


@pytest.fixture
def reproducible_df():
    """Fixed seed dataset for determinism tests."""
    np.random.seed(0)
    dates = pd.date_range("2015-01-01", periods=730, freq="D")
    t = np.arange(730)
    y = (50000 + 3000 * np.sin(2 * np.pi * t / 7)
         + 5000 * np.sin(2 * np.pi * t / 365.25)
         + np.random.normal(0, 200, 730))
    return pd.DataFrame({DS_COL: dates, Y_COL: y})


# ─── Deterministic split ───────────────────────────────────────────────────────

def test_split_always_same_boundary(reproducible_df):
    """Running chronological split twice on the same data gives the same boundary."""
    _, _, info1 = chronological_train_test_split(reproducible_df)
    _, _, info2 = chronological_train_test_split(reproducible_df)
    assert info1.split_timestamp == info2.split_timestamp


def test_split_always_same_train_size(reproducible_df):
    train1, _, _ = chronological_train_test_split(reproducible_df)
    train2, _, _ = chronological_train_test_split(reproducible_df)
    assert len(train1) == len(train2)


def test_split_always_same_test_size(reproducible_df):
    _, test1, _ = chronological_train_test_split(reproducible_df)
    _, test2, _ = chronological_train_test_split(reproducible_df)
    assert len(test1) == len(test2)


# ─── Deterministic forecast ────────────────────────────────────────────────────

def test_model_save_load_identical_forecast(reproducible_df, tmp_path):
    """
    Saving and reloading a Prophet model should yield identical predictions.
    """
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    train, test, _ = chronological_train_test_split(reproducible_df)

    # Train original model
    model, _ = train_daily_prophet(train)
    forecast_original = generate_forecast(model, test)

    # Save and reload
    model_path = tmp_path / "repro_model.json"
    with open(model_path, "w") as fh:
        fh.write(model_to_json(model))
    with open(model_path, "r") as fh:
        model_reloaded = model_from_json(fh.read())

    forecast_reloaded = generate_forecast(model_reloaded, test)

    # Forecasts should be numerically identical
    pd.testing.assert_series_equal(
        forecast_original["yhat"].reset_index(drop=True),
        forecast_reloaded["yhat"].reset_index(drop=True),
        check_exact=False,
        rtol=1e-5,
    )


# ─── Artifact integrity tests ──────────────────────────────────────────────────

def test_model_json_is_valid_json(reproducible_df, tmp_path):
    """Serialized model must be valid JSON."""
    import json
    train, _, _ = chronological_train_test_split(reproducible_df)
    model, _ = train_daily_prophet(train)
    model_json = model_to_json(model)
    # Should not raise
    parsed = json.loads(model_json)
    assert isinstance(parsed, dict)


def test_reloaded_model_is_prophet_instance(reproducible_df, tmp_path):
    train, _, _ = chronological_train_test_split(reproducible_df)
    model, _ = train_daily_prophet(train)
    model_path = tmp_path / "integrity_test.json"
    with open(model_path, "w") as fh:
        fh.write(model_to_json(model))
    with open(model_path, "r") as fh:
        reloaded = model_from_json(fh.read())
    assert isinstance(reloaded, Prophet)
