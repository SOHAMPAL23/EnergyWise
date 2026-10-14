"""
ml/prophet/builder.py
──────────────────────
Single source of truth for constructing a Prophet model.

Cross-validation, tuning and final training ALL call build_prophet_model(), so the
configuration that is validated is exactly the configuration that is shipped.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from ml.configs.config import DS_COL, RANDOM_SEED
from ml.prophet.configuration import build_prophet_params
from ml.prophet.holidays import add_holidays

MIN_YEARS_FOR_YEARLY = 2.0


def build_prophet_model(
    params: Optional[Dict[str, Any]],
    granularity: str,
    train_df: pd.DataFrame,
):
    """Return (unfitted Prophet, resolved_param_dict)."""
    if granularity not in ("daily", "weekly"):
        raise ValueError(f"granularity must be 'daily' or 'weekly', got {granularity!r}")
    p = build_prophet_params(params)
    span_years = (train_df[DS_COL].max() - train_df[DS_COL].min()).days / 365.25
    p["yearly_seasonality"] = bool(span_years >= MIN_YEARS_FOR_YEARLY)
    p["weekly_seasonality"] = granularity == "daily"   # no sub-weekly signal in weekly data
    p["daily_seasonality"] = False

    model = Prophet(
        changepoint_prior_scale=p["changepoint_prior_scale"],
        seasonality_prior_scale=p["seasonality_prior_scale"],
        holidays_prior_scale=p["holidays_prior_scale"],
        seasonality_mode=p["seasonality_mode"],
        yearly_seasonality=p["yearly_seasonality"],
        weekly_seasonality=p["weekly_seasonality"],
        daily_seasonality=p["daily_seasonality"],
        interval_width=p["interval_width"],
        n_changepoints=p["n_changepoints"],
        changepoint_range=p["changepoint_range"],
    )
    add_holidays(model)
    if granularity == "daily":
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model, p


def fit_model(model: Prophet, train_df: pd.DataFrame, seed: int = RANDOM_SEED) -> Prophet:
    """Fit with a fixed optimiser seed."""
    model.fit(train_df[[DS_COL, "y"]], seed=seed)
    return model


def predict_seeded(model: Prophet, df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    predict() draws Monte-Carlo samples for the uncertainty interval from numpy's
    global RNG. Seeding right before the call makes the intervals reproducible.
    """
    np.random.seed(seed)
    return model.predict(df[[DS_COL]])
