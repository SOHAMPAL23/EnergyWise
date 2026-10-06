"""
ml/validation_models/cross_validation.py
─────────────────────────────────────────
Training-Only Time-Series Cross-Validation

Implements expanding-window cross-validation EXCLUSIVELY inside the training
partition (first 70% of the series).

CRITICAL INVARIANT:
  The test set (last 30%) MUST NEVER appear inside any CV fold.
  Any code that passes test data into a CV fold violates this contract.

CV structure (conceptual):
  TRAINING PARTITION (70%)
  ├── Fold 1: train=[0..t1], val=[t1..t1+H]
  ├── Fold 2: train=[0..t2], val=[t2..t2+H]
  └── Fold 3: train=[0..t3], val=[t3..t3+H]

  TEST PARTITION (30%) — NEVER TOUCHED DURING CV

For each fold and parameter configuration, records:
  - fold index
  - training range
  - validation range
  - parameters
  - MAE
  - RMSE
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from ml.configs.config import (
    DS_COL,
    Y_COL,
    CV_N_FOLDS,
    CV_VAL_HORIZON_DAYS,
    CV_MIN_TRAIN_DAYS,
    CV_VAL_HORIZON_WEEKS,
    CV_MIN_TRAIN_WEEKS,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    HOLIDAY_COUNTRY,
)
from ml.evaluation.metrics import mae, rmse
from ml.prophet.configuration import build_prophet_params
from ml.prophet.holidays import add_holidays

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def expanding_window_cv(
    train_df: pd.DataFrame,
    params: Dict[str, Any],
    n_folds: int = CV_N_FOLDS,
    val_horizon: int = None,
    min_train_size: int = None,
    granularity: str = "daily",
) -> List[Dict[str, Any]]:
    """
    Expanding-window cross-validation on the training partition ONLY.

    Parameters
    ----------
    train_df       : Training DataFrame (ds, y) — FIRST 70% ONLY.
    params         : Prophet hyperparameters to evaluate.
    n_folds        : Number of expanding-window folds.
    val_horizon    : Validation horizon in rows (days or weeks).
    min_train_size : Minimum training rows before first fold.
    granularity    : "daily" or "weekly" — used for logging.

    Returns
    -------
    List of fold result dicts, each with:
      fold_index, train_start, train_end, val_start, val_end, MAE, RMSE, params
    """
    if min_train_size is None:
        min_train_size = CV_MIN_TRAIN_WEEKS if granularity == "weekly" else CV_MIN_TRAIN_DAYS
    if val_horizon is None or val_horizon == CV_VAL_HORIZON_DAYS and granularity == "weekly":
        val_horizon = CV_VAL_HORIZON_WEEKS if granularity == "weekly" else CV_VAL_HORIZON_DAYS

    n_total = len(train_df)
    logger.info("[CV] Starting expanding-window CV: %d folds, horizon=%d rows, "
                "total_train_rows=%d, granularity=%s.",
                n_folds, val_horizon, n_total, granularity)

    # Validate there is enough training data for CV
    required_min = min_train_size + n_folds * val_horizon
    if n_total < required_min:
        logger.warning(
            "[CV] Training set has %d rows but CV requires ~%d rows. "
            "Reducing val_horizon to %d.",
            n_total, required_min, max(1, (n_total - min_train_size) // n_folds),
        )
        val_horizon = max(1, (n_total - min_train_size) // n_folds)

    # Suppress Prophet internal logging during CV
    import logging as _log
    _log.getLogger("prophet").setLevel(_log.ERROR)
    _log.getLogger("cmdstanpy").setLevel(_log.ERROR)

    fold_results: List[Dict[str, Any]] = []

    for fold_idx in range(n_folds):
        # Expanding window: each fold adds `val_horizon` more training rows
        val_end_idx = n_total - (n_folds - fold_idx - 1) * val_horizon
        val_start_idx = val_end_idx - val_horizon
        train_end_idx = val_start_idx  # training ends just before validation

        if train_end_idx < min_train_size:
            logger.warning("[CV] Fold %d: insufficient training rows (%d < %d). Skipping.",
                           fold_idx + 1, train_end_idx, min_train_size)
            continue

        fold_train = train_df.iloc[:train_end_idx].copy()
        fold_val = train_df.iloc[val_start_idx:val_end_idx].copy()

        # CRITICAL CHECK: Validate no overlap and that val is entirely within training partition
        fold_train_max = fold_train[DS_COL].max()
        fold_val_min = fold_val[DS_COL].min()
        assert fold_train_max < fold_val_min, (
            f"[CV LEAKAGE] Fold {fold_idx + 1}: fold_train_max ({fold_train_max}) "
            f">= fold_val_min ({fold_val_min}). CV is leaking!"
        )

        # Build and train model
        param_dict = build_prophet_params(params)
        n_years_fold = (fold_train[DS_COL].max() - fold_train[DS_COL].min()).days / 365.25
        yearly = n_years_fold >= 2.0

        model = Prophet(
            changepoint_prior_scale=param_dict["changepoint_prior_scale"],
            seasonality_prior_scale=param_dict["seasonality_prior_scale"],
            holidays_prior_scale=param_dict["holidays_prior_scale"],
            seasonality_mode=param_dict["seasonality_mode"],
            yearly_seasonality=yearly,
            weekly_seasonality=True if granularity == "daily" else False,
            daily_seasonality=False,
            interval_width=param_dict["interval_width"],
            n_changepoints=param_dict["n_changepoints"],
            changepoint_range=param_dict["changepoint_range"],
        )
        add_holidays(model)
        if granularity == "daily":
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        model.fit(fold_train[[DS_COL, Y_COL]])

        # Predict on validation fold
        val_forecast = model.predict(fold_val[[DS_COL]])
        y_true = fold_val[Y_COL].values
        y_pred = val_forecast["yhat"].values

        fold_mae = mae(y_true, y_pred)
        fold_rmse = rmse(y_true, y_pred)

        fold_result = {
            "fold_index": fold_idx + 1,
            "train_start": str(fold_train[DS_COL].min().date()),
            "train_end": str(fold_train[DS_COL].max().date()),
            "val_start": str(fold_val[DS_COL].min().date()),
            "val_end": str(fold_val[DS_COL].max().date()),
            "n_train": len(fold_train),
            "n_val": len(fold_val),
            "MAE": round(fold_mae, 4),
            "RMSE": round(fold_rmse, 4),
            "params": {k: v for k, v in params.items()},
            "granularity": granularity,
        }
        fold_results.append(fold_result)

        logger.info(
            "[CV] Fold %d/%d: train=%d rows (%s→%s) | val=%d rows (%s→%s) | "
            "MAE=%.2f | RMSE=%.2f",
            fold_idx + 1, n_folds,
            len(fold_train), fold_train[DS_COL].min().date(), fold_train[DS_COL].max().date(),
            len(fold_val), fold_val[DS_COL].min().date(), fold_val[DS_COL].max().date(),
            fold_mae, fold_rmse,
        )

    return fold_results
