"""
backend/api/services/forecast.py
────────────────────────────────
Inference service for generating electricity demand forecasts using Prophet.

Design Principles
─────────────────
* NO training logic lives here. Models are loaded once at startup as immutable
  singletons from the ModelRegistry.
* Stateless execution: Each request constructs a fresh prediction DataFrame and
  produces predictions without mutating model or application state.
* Rigorous input validation: Date range ordering and horizons are verified before
  invoking the model.
* Clean error translation: Missing models raise `ModelNotAvailableError` (HTTP 503),
  and domain validation errors raise `ValidationError` (HTTP 422).
"""

from datetime import date
import logging
from typing import Any, Dict, List, Literal

import pandas as pd

from api.core.config import get_settings
from api.core.exceptions import ForecastError, ValidationError
from api.services.model_registry import get_registry

logger = logging.getLogger(__name__)

# Maximum forecast horizon in days (5 years = 1826 days)
MAX_HORIZON_DAYS = 1826


def generate_forecast(
    granularity: Literal["daily", "weekly"],
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """
    Generate point forecasts and uncertainty intervals for the given date range.

    Parameters
    ----------
    granularity : "daily" or "weekly"
    start_date  : Inclusive start date of forecast period
    end_date    : Inclusive end date of forecast period

    Returns
    -------
    List of forecast dicts matching the ForecastItem contract:
        - timestamp (str, ISO-8601)
        - forecast (float, MW)
        - lower_bound (float, MW)
        - upper_bound (float, MW)
        - granularity (str)
        - model_version (str)

    Raises
    ------
    ValidationError        : If date range is invalid (end_date < start_date) or exceeds limit.
    ModelNotAvailableError : If the model artifact for the requested granularity is not loaded (503).
    ForecastError          : If inference fails unexpectedly (500).
    """
    # ── 1. Validate date range ───────────────────────────────────────────────
    if end_date < start_date:
        raise ValidationError(
            f"Invalid date range: start_date ({start_date}) must be before or equal to end_date ({end_date})."
        )

    horizon_days = (end_date - start_date).days
    if horizon_days > MAX_HORIZON_DAYS:
        raise ValidationError(
            f"Requested forecast horizon ({horizon_days} days) exceeds maximum allowed limit of {MAX_HORIZON_DAYS} days."
        )

    # ── 2. Load model from registry ──────────────────────────────────────────
    registry = get_registry()
    # get_model raises ModelNotAvailableError (HTTP 503) if unavailable
    model = registry.get_model(granularity)

    settings = get_settings()
    default_version = (
        settings.MODEL_VERSION_DAILY
        if granularity == "daily"
        else settings.MODEL_VERSION_WEEKLY
    )
    try:
        model_info = registry.get_model_info(granularity)
        model_version = model_info.get("model_version", default_version)
    except Exception:
        model_version = default_version

    # ── 3. Construct future date index ───────────────────────────────────────
    try:
        if granularity == "daily":
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
        elif granularity == "weekly":
            # Prophet weekly model is anchored to Mondays (W-MON).
            # Snap start_date to the Monday of that week.
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            start_monday = start_dt - pd.Timedelta(days=start_dt.weekday())
            dates = pd.date_range(start=start_monday, end=end_dt, freq="W-MON")
            if len(dates) == 0:
                dates = pd.DatetimeIndex([start_monday])
        else:
            raise ValidationError(
                f"Unsupported granularity '{granularity}'. Supported: daily, weekly."
            )

        future_df = pd.DataFrame({"ds": dates})
    except Exception as exc:
        logger.error("Failed to construct forecast date index: %s", exc)
        raise ValidationError(f"Invalid date specification: {exc}") from exc

    # ── 4. Generate forecast (Stateless Inference) ───────────────────────────
    try:
        logger.debug(
            "Generating %s forecast for %d periods (%s to %s)",
            granularity,
            len(future_df),
            start_date,
            end_date,
        )
        forecast_df = model.predict(future_df)
    except Exception as exc:
        logger.exception("Prophet prediction failed for granularity=%s: %s", granularity, exc)
        raise ForecastError(f"Forecasting computation failed: {exc}") from exc

    # ── 5. Assemble response records ─────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    for _, row in forecast_df.iterrows():
        ts_val = row["ds"]
        if hasattr(ts_val, "strftime"):
            ts_str = ts_val.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            ts_str = str(ts_val)

        results.append({
            "timestamp": ts_str,
            "forecast": float(row["yhat"]),
            "lower_bound": float(row["yhat_lower"]),
            "upper_bound": float(row["yhat_upper"]),
            "granularity": granularity,
            "model_version": str(model_version),
        })

    logger.info(
        "Forecast generated successfully: %d points | granularity=%s | model_version=%s",
        len(results),
        granularity,
        model_version,
    )
    return results
