"""
ml/preprocessing/resampling.py
───────────────────────────────
Hourly → Daily and Weekly Resampling

Converts hourly OPSD load data (MW per hour) to Prophet input format:
  ds = timestamp (datetime64, tz-naive)
  y  = demand (mean MW over the period)

Resampling methodology:
  - Daily  : Resample hourly to daily using MEAN of all non-null hourly values in each day.
             If a day has 0 valid hourly readings (all NaN), the daily value is NaN.
  - Weekly : Resample daily to weekly (week starts Monday, W-MON) using MEAN of non-null
             daily values in each ISO week.

Why mean?
  The raw column is already an AVERAGE LOAD in MW (not energy in MWh).
  Taking the mean preserves the physical interpretation as "average load over the period."

The output DataFrames contain EXACTLY two columns:
  ds : datetime64[ns] (tz-naive UTC)
  y  : float64 (mean load in MW)
"""

import logging
from typing import Optional

import pandas as pd

from ml.configs.config import (
    TIMESTAMP_COL,
    TARGET_COL_RAW,
    DAILY_FREQUENCY,
    WEEKLY_FREQUENCY,
    DS_COL,
    Y_COL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def resample_to_daily(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL_RAW,
) -> pd.DataFrame:
    """
    Resample hourly OPSD data to daily mean load (Prophet format: ds, y).

    Parameters
    ----------
    df            : Cleaned DataFrame with timestamp and demand columns.
    timestamp_col : Datetime column (UTC-aware or naive).
    target_col    : Demand column (MW).

    Returns
    -------
    DataFrame with columns [ds, y]:
      ds — date (daily, tz-naive)
      y  — mean daily load (MW)
    """
    ts = df[timestamp_col].copy()

    # Normalize to tz-naive UTC for Prophet compatibility
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    demand = df[target_col].copy()

    # Build a temporary indexed series for resampling
    temp = pd.Series(demand.values, index=ts, name=target_col)
    temp.index = pd.DatetimeIndex(temp.index)

    # Resample to daily mean (skipna=True is the default: days with partial
    # NaN coverage will still produce a mean from available hours)
    daily = temp.resample(DAILY_FREQUENCY).mean()

    n_null = int(daily.isna().sum())
    logger.info(
        "[RESAMPLE] Hourly → Daily: %d days (%s to %s). Null days: %d.",
        len(daily), daily.index.min().date(), daily.index.max().date(), n_null,
    )

    # Compose Prophet-format DataFrame
    out = pd.DataFrame({
        DS_COL: daily.index,
        Y_COL: daily.values,
    }).reset_index(drop=True)

    return out


def resample_to_weekly(
    daily_df: pd.DataFrame,
    ds_col: str = DS_COL,
    y_col: str = Y_COL,
) -> pd.DataFrame:
    """
    Resample a daily Prophet DataFrame to weekly mean (week starts Monday).

    Parameters
    ----------
    daily_df : Daily Prophet DataFrame with [ds, y].
    ds_col   : Timestamp column name.
    y_col    : Demand column name.

    Returns
    -------
    DataFrame with columns [ds, y]:
      ds — Monday of each week (tz-naive)
      y  — mean of daily values within that week
    """
    temp = daily_df.set_index(ds_col)[y_col].copy()
    temp.index = pd.DatetimeIndex(temp.index)

    weekly = temp.resample(WEEKLY_FREQUENCY).mean()

    n_null = int(weekly.isna().sum())
    logger.info(
        "[RESAMPLE] Daily → Weekly: %d weeks (%s to %s). Null weeks: %d.",
        len(weekly), weekly.index.min().date(), weekly.index.max().date(), n_null,
    )

    out = pd.DataFrame({
        DS_COL: weekly.index,
        Y_COL: weekly.values,
    }).reset_index(drop=True)

    return out
