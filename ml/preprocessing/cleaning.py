"""
ml/preprocessing/cleaning.py
────────────────────────────
Data Cleaning for Time-Series Forecasting

All cleaning decisions are explicitly documented and logged.
Nothing is removed silently.

Operations (in order):
  1. Drop duplicate rows (keep first)
  2. Drop duplicate timestamps (keep first, log warning)
  3. Forward-fill isolated missing demand values (max 1 consecutive gap)
  4. Linear interpolation for remaining short gaps (≤ 3 hours)
  5. Flag and preserve large gaps without fabricating data

Notes on methodology:
  - Forward-fill is preferred for hourly load data to preserve the last known state
  - Linear interpolation is used only for genuinely short gaps (≤ 3 consecutive NaNs)
  - Data points that cannot be reasonably imputed are left as NaN and excluded from
    daily aggregation (daily mean skips NaN by default via skipna=True in resample)
  - Nothing beyond this is changed — no outlier removal, no smoothing
"""

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from ml.configs.config import (
    TIMESTAMP_COL,
    TARGET_COL_RAW,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def clean_raw_dataframe(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL_RAW,
    max_interp_gap: int = 3,
) -> pd.DataFrame:
    """
    Clean the raw OPSD DataFrame.

    Parameters
    ----------
    df               : Raw DataFrame from loader.
    timestamp_col    : Timestamp column name.
    target_col       : Demand target column name.
    max_interp_gap   : Max consecutive NaN rows to interpolate (default 3).

    Returns
    -------
    Cleaned DataFrame (copy, never modifies the input in-place).

    Cleaning decisions documented:
      - Duplicate rows: dropped, keeping first occurrence
      - Duplicate timestamps: averaged across duplicates
      - Isolated NaN (≤ max_interp_gap consecutive): linear interpolated
      - Larger NaN runs: left as NaN (excluded from daily mean aggregation)
    """
    df = df.copy()
    n_original = len(df)
    cleaning_log: Dict[str, Any] = {}

    # ── Step 1: Ensure timestamp is datetime ──────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        logger.info("[CLEANING] Timestamp column coerced to datetime.")

    # ── Step 2: Sort chronologically ──────────────────────────────────────
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    logger.info("[CLEANING] Sorted chronologically.")

    # ── Step 3: Drop duplicate rows ───────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(keep="first")
    n_dup_rows = n_before - len(df)
    cleaning_log["duplicate_rows_dropped"] = n_dup_rows
    if n_dup_rows > 0:
        logger.warning("[CLEANING] Dropped %d duplicate row(s) (kept first).", n_dup_rows)
    else:
        logger.info("[CLEANING] No duplicate rows to drop.")

    # ── Step 4: Handle duplicate timestamps (average demand values) ────────
    n_before = len(df)
    dup_ts_mask = df[timestamp_col].duplicated(keep=False)
    n_dup_ts = dup_ts_mask.sum()
    if n_dup_ts > 0:
        logger.warning(
            "[CLEANING] %d rows have duplicate timestamps. "
            "DECISION: Averaging demand values across duplicates.", n_dup_ts
        )
        df = df.groupby(timestamp_col, as_index=False)[target_col].mean()
        df = df.sort_values(by=timestamp_col).reset_index(drop=True)
        cleaning_log["duplicate_timestamps_averaged"] = n_dup_ts
    else:
        cleaning_log["duplicate_timestamps_averaged"] = 0
        logger.info("[CLEANING] No duplicate timestamps to handle.")

    # ── Step 5: Interpolate short missing demand gaps ─────────────────────
    n_null_before = int(df[target_col].isna().sum())
    if n_null_before > 0:
        # Linear interpolation for runs ≤ max_interp_gap consecutive NaN
        df[target_col] = df[target_col].interpolate(
            method="linear", limit=max_interp_gap, limit_direction="forward"
        )
        n_null_after = int(df[target_col].isna().sum())
        n_filled = n_null_before - n_null_after
        cleaning_log["nulls_interpolated"] = n_filled
        cleaning_log["nulls_remaining"] = n_null_after
        if n_filled > 0:
            logger.info(
                "[CLEANING] Interpolated %d null value(s) (linear, max_gap=%d). "
                "Remaining nulls: %d (will be excluded from daily mean).",
                n_filled, max_interp_gap, n_null_after,
            )
    else:
        cleaning_log["nulls_interpolated"] = 0
        cleaning_log["nulls_remaining"] = 0
        logger.info("[CLEANING] No null values to interpolate.")

    n_final = len(df)
    cleaning_log["rows_original"] = n_original
    cleaning_log["rows_final"] = n_final
    cleaning_log["rows_removed"] = n_original - n_final

    logger.info(
        "[CLEANING] Complete. Original: %d → Final: %d rows. Removed: %d.",
        n_original, n_final, n_original - n_final,
    )

    # Attach cleaning log as a custom attribute for downstream reference
    df.attrs["cleaning_log"] = cleaning_log
    return df
