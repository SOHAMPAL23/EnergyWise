"""
ml/prophet/daily.py
────────────────────
Daily Prophet Forecaster

Trains a Prophet model on daily (D) time-series data for Germany energy demand.

Model configuration:
  - Trend:            additive, with changepoints
  - Weekly seasonality: enabled (dominant pattern in daily energy data)
  - Yearly seasonality: enabled if training set spans ≥ 2 years
  - Holidays:         Germany (DE) federal holidays
  - Uncertainty:      95% prediction interval (yhat_lower, yhat_upper)

The model is trained ONLY on the training partition (first 70%).
Test data is NEVER passed to fit().
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from ml.configs.config import (
    DS_COL,
    Y_COL,
    MODELS_DIR,
    MODEL_ID_DAILY,
    MODEL_VERSION_DAILY,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)
from ml.prophet.builder import build_prophet_model, fit_model, predict_seeded
from ml.prophet.holidays import add_holidays, get_holiday_metadata

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def train_daily_prophet(
    train_df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Prophet, Dict[str, Any]]:
    """
    Train a Prophet model on daily energy demand data.

    Parameters
    ----------
    train_df : Daily training DataFrame (ds, y) — ONLY the training partition.
    params   : Optional parameter overrides (e.g., from hyperparameter tuning).
    verbose  : If True, suppress Prophet's internal logging.

    Returns
    -------
    (model, training_meta) — fitted Prophet and metadata dict.

    IMPORTANT: train_df must be the training partition ONLY.
               Never pass test data to this function.
    """
    make_output_dirs()

    n_years_train = (
        train_df[DS_COL].max() - train_df[DS_COL].min()
    ).days / 365.25

    # Suppress Prophet's internal cmdstanpy output
    import logging as _logging
    if not verbose:
        _logging.getLogger("prophet").setLevel(_logging.ERROR)
        _logging.getLogger("cmdstanpy").setLevel(_logging.ERROR)

    # Same builder as CV/tuning -> what was validated is what is shipped
    model, param_dict = build_prophet_model(params, "daily", train_df)
    yearly_seasonality = param_dict["yearly_seasonality"]

    logger.info("[PROPHET-DAILY] Training on %d daily records (%s → %s).",
                len(train_df), train_df[DS_COL].min().date(), train_df[DS_COL].max().date())
    logger.info("[PROPHET-DAILY] Parameters: %s", param_dict)

    # Fit — ONLY on training data
    t0 = time.time()
    fit_model(model, train_df)
    elapsed = round(time.time() - t0, 2)

    logger.info("[PROPHET-DAILY] Training complete in %.2f seconds.", elapsed)
    training_meta = {
        "model_id": MODEL_ID_DAILY,
        "model_version": MODEL_VERSION_DAILY,
        "granularity": "daily",
        "n_train_rows": len(train_df),
        "train_start": str(train_df[DS_COL].min().date()),
        "train_end": str(train_df[DS_COL].max().date()),
        "n_years_train": round(n_years_train, 2),
        "yearly_seasonality_enabled": yearly_seasonality,
        "parameters": {k: v for k, v in param_dict.items()},
        "holiday_metadata": get_holiday_metadata(),
        "training_duration_seconds": elapsed,
    }

    return model, training_meta


def generate_forecast(
    model: Prophet,
    future_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate forecast for a given set of timestamps.

    Parameters
    ----------
    model     : Fitted Prophet model.
    future_df : DataFrame with [ds] column containing timestamps to forecast.

    Returns
    -------
    Forecast DataFrame with [ds, yhat, yhat_lower, yhat_upper, trend,
    weekly, yearly, holidays, ...].
    """
    forecast = predict_seeded(model, future_df)
    logger.info("[PROPHET-DAILY] Forecast generated for %d timestamps.", len(forecast))
    return forecast


def save_model(
    model: Prophet,
    training_meta: Dict[str, Any],
    model_id: str = MODEL_ID_DAILY,
) -> Path:
    """
    Serialize a fitted Prophet model to JSON and save metadata.

    Parameters
    ----------
    model        : Fitted Prophet model.
    training_meta: Training metadata dict.
    model_id     : Unique model identifier for the file name.

    Returns
    -------
    Path — path to the saved model JSON file.
    """
    make_output_dirs()
    model_path = MODELS_DIR / f"{model_id}.json"

    with open(model_path, "w", encoding="utf-8") as fh:
        fh.write(model_to_json(model))
    logger.info("[PROPHET-DAILY] Model saved: %s", model_path)

    # Save metadata alongside
    meta_path = model_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(training_meta, fh, indent=2, default=str)
    logger.info("[PROPHET-DAILY] Metadata saved: %s", meta_path)

    return model_path


def load_model(model_path: Path) -> Prophet:
    """
    Deserialize a Prophet model from JSON.

    Parameters
    ----------
    model_path : Path to the serialized model JSON.

    Returns
    -------
    Prophet — loaded model ready for predict().

    Raises
    ------
    FileNotFoundError if the model file does not exist.
    ValueError        if the JSON is corrupted or cannot be deserialized.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"[PROPHET-DAILY] Model not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            model = model_from_json(fh.read())
        logger.info("[PROPHET-DAILY] Model loaded: %s", path)
        return model
    except Exception as exc:
        raise ValueError(
            f"[PROPHET-DAILY] Corrupted or incompatible model artifact at {path}: {exc}"
        ) from exc
