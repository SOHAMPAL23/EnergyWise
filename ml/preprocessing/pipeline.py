"""
ml/preprocessing/pipeline.py
────────────────────────────
Preprocessing Pipeline Orchestrator

Runs the full preprocessing sequence:
  1. Clean the raw DataFrame
  2. Resample hourly → daily (Prophet format: ds, y)
  3. Resample daily → weekly (Prophet format: ds, y)
  4. Apply chronological 70/30 split
  5. Save processed data to disk

Returns all datasets and split info for downstream consumption.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd

from ml.configs.config import (
    PROCESSED_DIR,
    TIMESTAMP_COL,
    TARGET_COL_RAW,
    TRAIN_RATIO,
    DS_COL,
    make_output_dirs,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)
from ml.preprocessing.cleaning import clean_raw_dataframe
from ml.preprocessing.resampling import resample_to_daily, resample_to_weekly
from ml.preprocessing.splitting import chronological_train_test_split, SplitInfo

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(
    df_raw: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL_RAW,
    train_ratio: float = TRAIN_RATIO,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for energy demand forecasting.

    Parameters
    ----------
    df_raw        : Raw DataFrame from loader (with UTC timestamps).
    timestamp_col : Raw timestamp column name.
    target_col    : Raw demand column name.
    train_ratio   : Fraction of data for training (default 0.70).
    save          : If True, write processed CSVs to PROCESSED_DIR.

    Returns
    -------
    dict with keys:
      df_daily      — Daily Prophet DataFrame (full series)
      df_weekly     — Weekly Prophet DataFrame (full series)
      train_daily   — Daily training set (first 70%)
      test_daily    — Daily test set (last 30%)
      train_weekly  — Weekly training set (first 70%)
      test_weekly   — Weekly test set (last 30%)
      split_daily   — SplitInfo for daily split
      split_weekly  — SplitInfo for weekly split
    """
    make_output_dirs()
    logger.info("[PIPELINE] ═══════════════════════ PREPROCESSING PIPELINE ═══════════════════════")

    # ── Step 1: Clean ─────────────────────────────────────────────────────
    logger.info("[PIPELINE] Step 1/4: Cleaning raw DataFrame ...")
    df_clean = clean_raw_dataframe(df_raw, timestamp_col=timestamp_col, target_col=target_col)

    # ── Step 2: Resample to daily ─────────────────────────────────────────
    logger.info("[PIPELINE] Step 2/4: Resampling to daily resolution ...")
    df_daily = resample_to_daily(df_clean, timestamp_col=timestamp_col, target_col=target_col)

    # Drop any remaining null daily values (cannot impute without fabricating data)
    n_null_daily = int(df_daily[DS_COL].isna().sum() + df_daily["y"].isna().sum())
    if n_null_daily > 0:
        logger.warning(
            "[PIPELINE] %d daily row(s) have null values after resampling. "
            "DECISION: Dropping null rows from daily series (cannot impute without fabricating data).",
            n_null_daily,
        )
        df_daily = df_daily.dropna(subset=["y"]).reset_index(drop=True)

    # ── Step 3: Resample to weekly ────────────────────────────────────────
    logger.info("[PIPELINE] Step 3/4: Resampling to weekly resolution ...")
    df_weekly = resample_to_weekly(df_daily)
    n_null_weekly = int(df_weekly["y"].isna().sum())
    if n_null_weekly > 0:
        logger.warning(
            "[PIPELINE] %d weekly row(s) have null values. Dropping them.",
            n_null_weekly,
        )
        df_weekly = df_weekly.dropna(subset=["y"]).reset_index(drop=True)

    # ── Step 4: Chronological 70/30 split ────────────────────────────────
    logger.info("[PIPELINE] Step 4/4: Applying chronological 70/30 split ...")
    train_daily, test_daily, split_daily = chronological_train_test_split(
        df_daily, train_ratio=train_ratio
    )
    train_weekly, test_weekly, split_weekly = chronological_train_test_split(
        df_weekly, train_ratio=train_ratio
    )

    # ── Save processed data ───────────────────────────────────────────────
    if save:
        _save_processed(df_daily, df_weekly, train_daily, test_daily, train_weekly, test_weekly)

    logger.info("[PIPELINE] ══════════════════════════════════════════════════════════════════════")
    logger.info("[PIPELINE] PREPROCESSING COMPLETE")
    logger.info("[PIPELINE]   Daily  : %d total | %d train | %d test", len(df_daily), len(train_daily), len(test_daily))
    logger.info("[PIPELINE]   Weekly : %d total | %d train | %d test", len(df_weekly), len(train_weekly), len(test_weekly))
    logger.info("[PIPELINE] ══════════════════════════════════════════════════════════════════════")

    return {
        "df_daily": df_daily,
        "df_weekly": df_weekly,
        "train_daily": train_daily,
        "test_daily": test_daily,
        "train_weekly": train_weekly,
        "test_weekly": test_weekly,
        "split_daily": split_daily,
        "split_weekly": split_weekly,
    }


def _save_processed(df_daily, df_weekly, train_daily, test_daily, train_weekly, test_weekly):
    """Save all processed DataFrames to the PROCESSED_DIR."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "daily_full.csv": df_daily,
        "weekly_full.csv": df_weekly,
        "train_daily.csv": train_daily,
        "test_daily.csv": test_daily,
        "train_weekly.csv": train_weekly,
        "test_weekly.csv": test_weekly,
    }
    for fname, df in files.items():
        fpath = PROCESSED_DIR / fname
        df.to_csv(fpath, index=False)
        logger.info("[PIPELINE] Saved: %s (%d rows)", fpath, len(df))
