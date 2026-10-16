"""
ml/validation_models/cross_validation.py
─────────────────────────────────────────
Training-only rolling-origin (expanding-window) cross-validation.

    TRAINING PARTITION (first 70%)                      TEST (last 30%)
    |------------------------------------------------|  |----------------| <- never seen
    fold 1: [=== train ===][ val ]
    fold 2: [====== train =====][ val ]
    fold k: [============ train ============][ val ]

* The training window ALWAYS starts at row 0 and grows with the origin
  (expanding window).
* Origins are anchored backwards from the END of the training partition and are
  `step` rows apart; validation windows may overlap when step < horizon.
* The function receives ONLY the training partition. If `holdout_start` is given it
  additionally verifies that no row at/after the boundary is present, and an
  AccessAuditor can be wrapped around the call to observe every dataframe that
  reaches Prophet.

Each fold result: fold_index, train/val ranges, n_train, n_val, MAE, RMSE,
interval_coverage, interval_ratio_q95 (see below), params, granularity.

`interval_ratio_q95` is the 95th percentile of |y - yhat| / (distance from yhat to
the corresponding interval edge). A value of 1.0 means the nominal interval
captured 95% of validation points; >1 means the interval is too narrow. It is
training-only calibration evidence for the anomaly-detection sensitivity knob.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.configs.config import (
    DS_COL,
    Y_COL,
    CV_N_FOLDS,
    CV_VAL_HORIZON_DAYS,
    CV_MIN_TRAIN_DAYS,
    CV_STEP_DAYS,
    CV_VAL_HORIZON_WEEKS,
    CV_MIN_TRAIN_WEEKS,
    CV_STEP_WEEKS,
    RANDOM_SEED,
)
from ml.evaluation.metrics import mae, rmse
from ml.prophet.builder import build_prophet_model, fit_model, predict_seeded
from ml.validation_models.guard import assert_partition_before

logger = logging.getLogger(__name__)


def cv_defaults(granularity: str) -> Dict[str, int]:
    if granularity == "weekly":
        return dict(min_train=CV_MIN_TRAIN_WEEKS, horizon=CV_VAL_HORIZON_WEEKS, step=CV_STEP_WEEKS)
    return dict(min_train=CV_MIN_TRAIN_DAYS, horizon=CV_VAL_HORIZON_DAYS, step=CV_STEP_DAYS)


def make_rolling_origin_folds(
    n_rows: int,
    min_train: int,
    horizon: int,
    step: int,
    n_folds: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Return [(origin, val_end), ...] as row indices into the training partition.

    train = rows[:origin]      (expanding window, starts at 0)
    val   = rows[origin:val_end]   with val_end = origin + horizon <= n_rows

    Origins are anchored backwards from n_rows - horizon so the most recent data
    is always validated, then returned in chronological order.
    """
    if min_train < 1 or horizon < 1 or step < 1:
        raise ValueError("min_train, horizon and step must all be >= 1")
    if n_rows < min_train + horizon:
        raise ValueError(
            f"Training partition too small for CV: {n_rows} rows < min_train({min_train}) "
            f"+ horizon({horizon})."
        )
    origins = []
    o = n_rows - horizon
    while o >= min_train:
        origins.append(o)
        o -= step
    origins.reverse()
    if n_folds is not None:
        origins = origins[-int(n_folds):]
    return [(o, o + horizon) for o in origins]


def _fold_metrics(val_df: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, float]:
    y = val_df[Y_COL].to_numpy(dtype=float)
    yhat = forecast["yhat"].to_numpy(dtype=float)
    lo = forecast["yhat_lower"].to_numpy(dtype=float)
    hi = forecast["yhat_upper"].to_numpy(dtype=float)
    cover = float(np.mean((y >= lo) & (y <= hi)))
    side = np.where(y >= yhat, hi - yhat, yhat - lo)
    ratio = np.abs(y - yhat) / np.maximum(side, 1e-9)
    return {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "interval_coverage": cover,
        "interval_ratio_q95": float(np.quantile(ratio, 0.95)),
    }


def run_cv(
    train_df: pd.DataFrame,
    params: Dict[str, Any],
    granularity: str = "daily",
    folds: Optional[List[Tuple[int, int]]] = None,
    holdout_start: Optional[pd.Timestamp] = None,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """
    Evaluate ONE parameter configuration across rolling-origin folds.

    train_df       : TRAINING partition only (ds, y), chronologically sorted.
    holdout_start  : first timestamp of the hold-out test set. If supplied, the
                     function raises LeakageError if train_df reaches it.
    folds          : precomputed [(origin, val_end)]; default derives from config.
    """
    if holdout_start is not None:
        assert_partition_before(train_df, holdout_start, "CV input")
    if not train_df[DS_COL].is_monotonic_increasing:
        raise ValueError("train_df must be sorted by ds ascending")

    if folds is None:
        folds = make_rolling_origin_folds(len(train_df), n_folds=CV_N_FOLDS, **cv_defaults(granularity))

    import logging as _log
    _log.getLogger("prophet").setLevel(_log.ERROR)
    _log.getLogger("cmdstanpy").setLevel(_log.ERROR)

    results: List[Dict[str, Any]] = []
    for i, (origin, val_end) in enumerate(folds, start=1):
        fold_train = train_df.iloc[:origin]
        fold_val = train_df.iloc[origin:val_end]
        assert fold_train[DS_COL].max() < fold_val[DS_COL].min(), (
            f"[CV LEAKAGE] fold {i}: train max {fold_train[DS_COL].max()} >= val min "
            f"{fold_val[DS_COL].min()}"
        )
        assert val_end <= len(train_df)

        model, _ = build_prophet_model(params, granularity, fold_train)
        fit_model(model, fold_train, seed=seed)
        forecast = predict_seeded(model, fold_val, seed=seed)
        m = _fold_metrics(fold_val, forecast)

        results.append({
            "fold_index": i,
            "train_start": str(fold_train[DS_COL].min().date()),
            "train_end": str(fold_train[DS_COL].max().date()),
            "val_start": str(fold_val[DS_COL].min().date()),
            "val_end": str(fold_val[DS_COL].max().date()),
            "n_train": int(len(fold_train)),
            "n_val": int(len(fold_val)),
            "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "interval_coverage": round(m["interval_coverage"], 4),
            "interval_ratio_q95": round(m["interval_ratio_q95"], 4),
            "params": dict(params),
            "granularity": granularity,
        })
    return results


def expanding_window_cv(
    train_df: pd.DataFrame,
    params: Dict[str, Any],
    n_folds: Optional[int] = None,
    val_horizon: Optional[int] = None,
    min_train_size: Optional[int] = None,
    granularity: str = "daily",
    step: Optional[int] = None,
    holdout_start: Optional[pd.Timestamp] = None,
) -> List[Dict[str, Any]]:
    """Backwards-compatible wrapper around make_rolling_origin_folds + run_cv."""
    d = cv_defaults(granularity)
    horizon = val_horizon or d["horizon"]
    folds = make_rolling_origin_folds(
        len(train_df),
        min_train=min_train_size or d["min_train"],
        horizon=horizon,
        step=step or horizon,       # non-overlapping by default for this legacy entry point
        n_folds=n_folds,
    )
    return run_cv(train_df, params, granularity, folds=folds, holdout_start=holdout_start)
