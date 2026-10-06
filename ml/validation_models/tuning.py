"""
ml/validation_models/tuning.py
────────────────────────────────
Training-Only Hyperparameter Tuning

Runs a grid search over the configured parameter space using training-only
expanding-window cross-validation.

CRITICAL:
  - Training data ONLY is passed to this function.
  - The test set is NEVER touched during tuning.
  - Model selection is based solely on CV MAE (average across folds).
  - The best configuration is returned WITHOUT any exposure to test performance.

Tracks all parameter configurations and their CV results.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from ml.configs.config import (
    PARAM_GRID,
    CV_N_FOLDS,
    CV_VAL_HORIZON_DAYS,
    CV_VAL_HORIZON_WEEKS,
    EXPERIMENTS_REPORT_DIR,
    REGION,
    DATASET_VERSION,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)
from ml.validation_models.cross_validation import expanding_window_cv
from ml.experiments.tracker import log_experiment

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def run_hyperparameter_tuning(
    train_df: pd.DataFrame,
    param_grid: List[Dict[str, Any]] = None,
    n_folds: int = CV_N_FOLDS,
    val_horizon: int = None,
    granularity: str = "daily",
    save: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run hyperparameter grid search over the training partition ONLY.

    Parameters
    ----------
    train_df    : Training DataFrame (FIRST 70% only — test must NOT be included).
    param_grid  : List of parameter dicts to evaluate (default: configs.PARAM_GRID).
    n_folds     : Number of expanding-window CV folds.
    val_horizon : Validation horizon in rows per fold (auto-selected by granularity if None).
    granularity : "daily" or "weekly".
    save        : Save tuning results to reports/experiments/.

    Returns
    -------
    (best_params, all_results)
      best_params — The parameter dict with the lowest average CV MAE.
      all_results — List of all trial results (one per parameter config).
    """
    make_output_dirs()
    if param_grid is None:
        param_grid = PARAM_GRID
    if val_horizon is None:
        val_horizon = CV_VAL_HORIZON_WEEKS if granularity == "weekly" else CV_VAL_HORIZON_DAYS

    logger.info("[TUNING] ══════════════════════════════════════════════════════")
    logger.info("[TUNING] Starting hyperparameter tuning on TRAINING set only.")
    logger.info("[TUNING] Granularity: %s | Folds: %d | Horizon: %d | Configs: %d",
                granularity, n_folds, val_horizon, len(param_grid))
    logger.info("[TUNING] IMPORTANT: Test set is NOT visible during tuning.")
    logger.info("[TUNING] ══════════════════════════════════════════════════════")

    all_results: List[Dict[str, Any]] = []

    for trial_idx, params in enumerate(param_grid):
        logger.info("[TUNING] Trial %d/%d: %s", trial_idx + 1, len(param_grid), params)

        # Run expanding-window CV on training data only
        fold_results = expanding_window_cv(
            train_df=train_df,
            params=params,
            n_folds=n_folds,
            val_horizon=val_horizon,
            granularity=granularity,
        )

        if not fold_results:
            logger.warning("[TUNING] Trial %d produced no valid fold results. Skipping.", trial_idx + 1)
            continue

        avg_mae = round(sum(f["MAE"] for f in fold_results) / len(fold_results), 4)
        avg_rmse = round(sum(f["RMSE"] for f in fold_results) / len(fold_results), 4)

        trial_result = {
            "trial_index": trial_idx + 1,
            "granularity": granularity,
            "params": params,
            "n_folds": len(fold_results),
            "avg_cv_mae": avg_mae,
            "avg_cv_rmse": avg_rmse,
            "fold_details": fold_results,
        }
        all_results.append(trial_result)

        # Log experiment
        log_experiment(
            experiment_id=f"tuning_{granularity}_trial_{trial_idx+1}",
            model="prophet",
            granularity=granularity,
            dataset_version=DATASET_VERSION,
            region=REGION,
            training_range=f"{train_df[train_df.columns[0]].min()} → {train_df[train_df.columns[0]].max()}",
            validation_range=f"CV {n_folds} folds, horizon={val_horizon}",
            hyperparameters=params,
            mae_val=avg_mae,
            rmse_val=avg_rmse,
        )

        logger.info("[TUNING] Trial %d/%d: avg_CV_MAE=%.4f | avg_CV_RMSE=%.4f",
                    trial_idx + 1, len(param_grid), avg_mae, avg_rmse)

    if not all_results:
        raise RuntimeError("[TUNING] No successful tuning trials. Check training data size.")

    # Select best based on avg CV MAE (NOT test MAE)
    best_trial = min(all_results, key=lambda x: x["avg_cv_mae"])
    best_params = best_trial["params"]

    logger.info("[TUNING] ══════════════════════════════════════════════════════")
    logger.info("[TUNING] BEST CONFIGURATION (by training-only CV MAE):")
    logger.info("[TUNING]   Params       : %s", best_params)
    logger.info("[TUNING]   Avg CV MAE   : %.4f MW", best_trial["avg_cv_mae"])
    logger.info("[TUNING]   Avg CV RMSE  : %.4f MW", best_trial["avg_cv_rmse"])
    logger.info("[TUNING] ══════════════════════════════════════════════════════")

    if save:
        _save_tuning_results(all_results, granularity)

    return best_params, all_results


def _save_tuning_results(all_results: List[Dict], granularity: str):
    """Save tuning results to CSV."""
    rows = []
    for trial in all_results:
        row = {
            "trial_index": trial["trial_index"],
            "granularity": trial["granularity"],
            "avg_cv_mae": trial["avg_cv_mae"],
            "avg_cv_rmse": trial["avg_cv_rmse"],
            "n_folds": trial["n_folds"],
        }
        row.update({f"param_{k}": v for k, v in trial["params"].items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = EXPERIMENTS_REPORT_DIR / f"tuning_results_{granularity}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("[TUNING] Results saved: %s", csv_path)

    # Save full JSON for complete fold details
    json_path = EXPERIMENTS_REPORT_DIR / f"tuning_results_{granularity}.json"
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info("[TUNING] Detailed results saved: %s", json_path)
