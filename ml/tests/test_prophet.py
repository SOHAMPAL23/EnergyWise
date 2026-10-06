"""
ml/tests/test_prophet.py
─────────────────────────
Prophet Model Tests

Tests:
  - Model training on synthetic data
  - Forecast generation (shape, required columns)
  - Component extraction
  - Model serialization / deserialization
  - Artifact validation
"""

import pytest
import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from ml.prophet.daily import train_daily_prophet, generate_forecast, save_model, load_model
from ml.prophet.components import extract_components
from ml.configs.config import DS_COL, Y_COL, MODELS_DIR


@pytest.fixture
def synthetic_train_df():
    """2 years of synthetic daily data for quick Prophet training tests."""
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    dates = pd.date_range("2015-01-01", periods=730, freq="D")
    # Additive seasonality + noise
    t = np.arange(730)
    y = (50000
         + 3000 * np.sin(2 * np.pi * t / 7)       # weekly
         + 5000 * np.sin(2 * np.pi * t / 365.25)  # yearly
         + np.random.normal(0, 500, 730))
    return pd.DataFrame({DS_COL: dates, Y_COL: y})


@pytest.fixture
def synthetic_test_df():
    dates = pd.date_range("2017-01-01", periods=30, freq="D")
    return pd.DataFrame({DS_COL: dates})


# ─── Training tests ────────────────────────────────────────────────────────────

def test_train_daily_prophet_returns_tuple(synthetic_train_df):
    model, meta = train_daily_prophet(synthetic_train_df)
    assert isinstance(model, Prophet)
    assert isinstance(meta, dict)


def test_train_daily_prophet_meta_keys(synthetic_train_df):
    _, meta = train_daily_prophet(synthetic_train_df)
    required_keys = {"model_id", "granularity", "n_train_rows", "train_start",
                     "train_end", "parameters", "training_duration_seconds"}
    assert required_keys.issubset(set(meta.keys()))


def test_train_daily_prophet_granularity(synthetic_train_df):
    _, meta = train_daily_prophet(synthetic_train_df)
    assert meta["granularity"] == "daily"


def test_train_records_correct_row_count(synthetic_train_df):
    _, meta = train_daily_prophet(synthetic_train_df)
    assert meta["n_train_rows"] == len(synthetic_train_df)


# ─── Forecast tests ────────────────────────────────────────────────────────────

def test_forecast_has_required_columns(synthetic_train_df, synthetic_test_df):
    model, _ = train_daily_prophet(synthetic_train_df)
    forecast = generate_forecast(model, synthetic_test_df)
    for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
        assert col in forecast.columns, f"Missing column: {col}"


def test_forecast_length_matches_test(synthetic_train_df, synthetic_test_df):
    model, _ = train_daily_prophet(synthetic_train_df)
    forecast = generate_forecast(model, synthetic_test_df)
    assert len(forecast) == len(synthetic_test_df)


def test_forecast_uncertainty_bounds_ordered(synthetic_train_df, synthetic_test_df):
    model, _ = train_daily_prophet(synthetic_train_df)
    forecast = generate_forecast(model, synthetic_test_df)
    assert (forecast["yhat_lower"] <= forecast["yhat"]).all()
    assert (forecast["yhat"] <= forecast["yhat_upper"]).all()


def test_forecast_has_no_nulls(synthetic_train_df, synthetic_test_df):
    model, _ = train_daily_prophet(synthetic_train_df)
    forecast = generate_forecast(model, synthetic_test_df)
    assert forecast["yhat"].isna().sum() == 0


# ─── Component extraction tests ────────────────────────────────────────────────

def test_extract_components_returns_dict(synthetic_train_df, synthetic_test_df):
    model, _ = train_daily_prophet(synthetic_train_df)
    forecast = generate_forecast(model, synthetic_test_df)
    components = extract_components(model, forecast, save=False)
    assert isinstance(components, dict)


def test_extract_components_has_trend(synthetic_train_df, synthetic_test_df):
    model, _ = train_daily_prophet(synthetic_train_df)
    forecast = generate_forecast(model, synthetic_test_df)
    components = extract_components(model, forecast, save=False)
    assert "trend" in components
    assert len(components["trend"]) > 0


def test_extract_components_has_changepoints(synthetic_train_df, synthetic_test_df):
    model, _ = train_daily_prophet(synthetic_train_df)
    forecast = generate_forecast(model, synthetic_test_df)
    components = extract_components(model, forecast, save=False)
    assert "changepoints" in components


# ─── Serialization tests ───────────────────────────────────────────────────────

def test_save_and_load_model(synthetic_train_df, tmp_path):
    model, meta = train_daily_prophet(synthetic_train_df)
    model_path = tmp_path / "prophet_test.json"

    # Save manually
    with open(model_path, "w") as fh:
        fh.write(model_to_json(model))

    # Load
    loaded_model = load_model(model_path)
    assert loaded_model is not None
    assert isinstance(loaded_model, Prophet)


def test_loaded_model_can_predict(synthetic_train_df, synthetic_test_df, tmp_path):
    model, _ = train_daily_prophet(synthetic_train_df)
    model_path = tmp_path / "prophet_test_pred.json"

    with open(model_path, "w") as fh:
        fh.write(model_to_json(model))

    loaded = load_model(model_path)
    forecast = generate_forecast(loaded, synthetic_test_df)
    assert len(forecast) == len(synthetic_test_df)
    assert "yhat" in forecast.columns


def test_load_nonexistent_model_raises():
    with pytest.raises(FileNotFoundError):
        load_model(Path("nonexistent_model.json"))


def test_load_corrupted_model_raises(tmp_path):
    bad_file = tmp_path / "bad_model.json"
    bad_file.write_text("not valid json {{{}}}}")
    with pytest.raises(ValueError, match="Corrupted"):
        load_model(bad_file)
