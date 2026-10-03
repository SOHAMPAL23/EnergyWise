"""
ml/preprocessing/splitting.py
──────────────────────────────
Strict Chronological Train/Test Split

THIS MODULE IS THE MOST CRITICAL DATA-LEAKAGE PREVENTION COMPONENT.

The split MUST be:
  - Chronological (sorted by time before splitting)
  - Deterministic (same result every run for same input)
  - Percentage-based on row count (floor of total * ratio)
  - NEVER shuffled

Returns:
  train_df        — First 70% of the time series
  test_df         — Last 30% of the time series
  split_timestamp — Exact boundary timestamp (first timestamp in test set)
  train_start     — First timestamp in training set
  train_end       — Last timestamp in training set
  test_start      — First timestamp in test set
  test_end        — Last timestamp in test set

Post-split verification:
  - train_df timestamps are ALL strictly less than test_df timestamps
  - No timestamp appears in both sets
  - split is deterministic

NEVER call random_state, shuffle=True, or sklearn.train_test_split.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from ml.configs.config import (
    TRAIN_RATIO,
    DS_COL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """Container for split boundary metadata."""
    split_timestamp: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    train_pct: float
    test_pct: float


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    ds_col: str = DS_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    Split a time-series DataFrame chronologically.

    Parameters
    ----------
    df          : Prophet-format DataFrame with at least a [ds] column.
    train_ratio : Fraction for training (default 0.70).
    ds_col      : Timestamp column name.

    Returns
    -------
    (train_df, test_df, SplitInfo)

    Raises
    ------
    ValueError  if the DataFrame is empty, or if temporal ordering is violated.
    AssertionError  if post-split leakage checks fail.
    """
    if df.empty:
        raise ValueError("[SPLIT] DataFrame is empty — cannot split.")
    if ds_col not in df.columns:
        raise ValueError(f"[SPLIT] Timestamp column '{ds_col}' not found.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"[SPLIT] train_ratio must be in (0, 1). Got: {train_ratio}")

    # ── Sort chronologically (defensive — should already be sorted) ───────
    df = df.sort_values(by=ds_col).reset_index(drop=True)

    n_total = len(df)
    split_idx = int(np.floor(n_total * train_ratio))  # deterministic floor

    if split_idx == 0 or split_idx >= n_total:
        raise ValueError(
            f"[SPLIT] Invalid split at index {split_idx} for {n_total} rows "
            f"with train_ratio={train_ratio}."
        )

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # ── CRITICAL: Post-split leakage checks ───────────────────────────────
    train_max = train_df[ds_col].max()
    test_min = test_df[ds_col].min()

    # Check 1: All train timestamps < all test timestamps
    assert train_max < test_min, (
        f"[LEAKAGE DETECTED] train_max ({train_max}) >= test_min ({test_min}). "
        "Chronological ordering has been violated."
    )

    # Check 2: No timestamp overlap
    train_set = set(train_df[ds_col].astype(str))
    test_set = set(test_df[ds_col].astype(str))
    overlap = train_set & test_set
    assert len(overlap) == 0, (
        f"[LEAKAGE DETECTED] {len(overlap)} timestamp(s) appear in BOTH train and test. "
        f"Examples: {list(overlap)[:5]}"
    )

    # ── Build metadata ────────────────────────────────────────────────────
    info = SplitInfo(
        split_timestamp=test_min,
        train_start=train_df[ds_col].min(),
        train_end=train_max,
        test_start=test_min,
        test_end=test_df[ds_col].max(),
        n_train=len(train_df),
        n_test=len(test_df),
        train_pct=round(100 * len(train_df) / n_total, 2),
        test_pct=round(100 * len(test_df) / n_total, 2),
    )

    logger.info("[SPLIT] ══════════════════════════════════════════════════")
    logger.info("[SPLIT] Total rows    : %d", n_total)
    logger.info("[SPLIT] Split index   : %d (%.0f%% train / %.0f%% test)",
                split_idx, 100 * train_ratio, 100 * (1 - train_ratio))
    logger.info("[SPLIT] Training set  : %d rows (%s → %s)",
                info.n_train, info.train_start.date(), info.train_end.date())
    logger.info("[SPLIT] Test set      : %d rows (%s → %s)",
                info.n_test, info.test_start.date(), info.test_end.date())
    logger.info("[SPLIT] Split boundary: %s", info.split_timestamp.date())
    logger.info("[SPLIT] LEAKAGE CHECK : PASSED ✓ (train_max < test_min, no overlap)")
    logger.info("[SPLIT] ══════════════════════════════════════════════════")

    return train_df, test_df, info
