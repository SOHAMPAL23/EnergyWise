"""
ml/prophet/weekly.py
─────────────────────
Weekly Prophet Forecaster

Trains a Prophet model on weekly (W-MON) time-series data.
Mirrors daily.py structure — same design, different granularity.

Seasonal period differences vs daily:
  - Weekly seasonality: DISABLED (no sub-weekly pattern in weekly aggregated data)
  - Yearly seasonality: ENABLED (52-week annual cycle is the primary seasonality)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from ml.configs.config import (
    DS_COL,
    Y_COL,
    MODELS_DIR,
    MODEL_ID_WEEKLY,
    MODEL_VERSION_WEEKLY,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)
from ml.prophet.configuration import build_prophet_params
from ml.prophet.holidays import add_holidays, get_holiday_metadata

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def train_weekly_prophet(
    train_df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Prophet, Dict[str, Any]]:
    """
    Train a Prophet model on weekly aggregated energy demand data.

    Parameters
    ----------
    train_df : Weekly training DataFrame (ds, y) — ONLY the training partition.
    params   : Optional parameter overrides.
    verbose  : If True, allow Prophet logging.

    Returns
    -------
    (model, training_meta)
    """
    make_output_dirs()

    n_years_train = (
        train_df[DS_COL].max() - train_df[DS_COL].min()
    ).days / 365.25
    yearly_seasonality = n_years_train >= 2.0

    param_dict = build_prophet_params(params)
    param_dict["yearly_seasonality"] = yearly_seasonality
    param_dict["weekly_seasonality"] = False   # No sub-weekly pattern in weekly data

    logger.info("[PROPHET-WEEKLY] Training on %d weekly records (%s → %s).",
                len(train_df), train_df[DS_COL].min().date(), train_df[DS_COL].max().date())

    import logging as _logging
    if not verbose:
        _logging.getLogger("prophet").setLevel(_logging.ERROR)
        _logging.getLogger("cmdstanpy").setLevel(_logging.ERROR)

    model = Prophet(
        changepoint_prior_scale=param_dict["changepoint_prior_scale"],
        seasonality_prior_scale=param_dict["seasonality_prior_scale"],
        holidays_prior_scale=param_dict["holidays_prior_scale"],
        seasonality_mode=param_dict["seasonality_mode"],
        yearly_seasonality=param_dict["yearly_seasonality"],
        weekly_seasonality=param_dict["weekly_seasonality"],
        daily_seasonality=False,
        interval_width=param_dict["interval_width"],
        n_changepoints=param_dict["n_changepoints"],
        changepoint_range=param_dict["changepoint_range"],
    )
    add_holidays(model)

    t0 = time.time()
    model.fit(train_df[[DS_COL, Y_COL]])
    elapsed = round(time.time() - t0, 2)

    logger.info("[PROPHET-WEEKLY] Training complete in %.2f seconds.", elapsed)

    training_meta = {
        "model_id": MODEL_ID_WEEKLY,
        "model_version": MODEL_VERSION_WEEKLY,
        "granularity": "weekly",
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


def generate_weekly_forecast(model: Prophet, future_df: pd.DataFrame) -> pd.DataFrame:
    """Generate weekly forecast."""
    forecast = model.predict(future_df[[DS_COL]])
    logger.info("[PROPHET-WEEKLY] Forecast generated for %d weeks.", len(forecast))
    return forecast


def save_model(model: Prophet, training_meta: Dict, model_id: str = MODEL_ID_WEEKLY) -> Path:
    """Save weekly Prophet model and metadata."""
    make_output_dirs()
    model_path = MODELS_DIR / f"{model_id}.json"
    with open(model_path, "w", encoding="utf-8") as fh:
        fh.write(model_to_json(model))
    meta_path = model_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(training_meta, fh, indent=2, default=str)
    logger.info("[PROPHET-WEEKLY] Model saved: %s", model_path)
    return model_path


def load_model(model_path: Path) -> Prophet:
    """Load weekly Prophet model from JSON."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"[PROPHET-WEEKLY] Model not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return model_from_json(fh.read())
    except Exception as exc:
        raise ValueError(f"[PROPHET-WEEKLY] Corrupted model at {path}: {exc}") from exc
