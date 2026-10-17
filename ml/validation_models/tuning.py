"""
ml/validation_models/tuning.py
────────────────────────────────
Training-only hyperparameter tuning (grid search over rolling-origin CV).

GUARANTEES
  * The only data argument is the TRAINING partition. The hold-out test set is not an
    argument, is not importable from here, and (when `holdout_start` is supplied) an
    AccessAuditor raises the moment a row at/after the boundary reaches Prophet.
  * Selection uses mean validation MAE across folds (tie -> mean RMSE -> config id).
    No test metric exists in this module.
  * EVERY fold of EVERY configuration is persisted by the ExperimentTracker, plus one
    aggregate row per configuration.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.configs.config import (
    PARAM_GRID,
    PARAM_SEARCH_SPACE,
    SELECTION_METRIC,
    CV_N_FOLDS,
    ARTIFACT_METADATA_DIR,
    DATASET_VERSION,
    REGION,
    RANDOM_SEED,
    MODEL_VERSION_DAILY,
    MODEL_VERSION_WEEKLY,
    DS_COL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)
from ml.experiments.tracker import ExperimentTracker, config_id, new_run_id
from ml.prophet.configuration import build_prophet_params
from ml.validation_models.cross_validation import cv_defaults, make_rolling_origin_folds, run_cv
from ml.validation_models.guard import AccessAuditor, assert_partition_before

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

DEFAULT_PARAMS_FOR_REFERENCE = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "n_changepoints": 25,
    "changepoint_range": 0.8,
}


def _range(df: pd.DataFrame) -> str:
    return f"{df[DS_COL].min().date()}..{df[DS_COL].max().date()}"


def run_tuning(
    train_df: pd.DataFrame,
    granularity: str = "daily",
    param_grid: Optional[List[Dict[str, Any]]] = None,
    holdout_start: Optional[pd.Timestamp] = None,
    tracker: Optional[ExperimentTracker] = None,
    cv_overrides: Optional[Dict[str, int]] = None,
    save: bool = True,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Grid search on the TRAINING partition only. Returns a dict with
    best_params, best_trial, trials (all configs), cv_spec, access_audit, run_id.
    """
    make_output_dirs()
    param_grid = param_grid if param_grid is not None else PARAM_GRID
    train_df = train_df.sort_values(DS_COL).reset_index(drop=True)
    if holdout_start is not None:
        assert_partition_before(train_df, holdout_start, "tuning input")

    spec = cv_defaults(granularity)
    if cv_overrides:
        spec.update(cv_overrides)
    folds = make_rolling_origin_folds(len(train_df), n_folds=CV_N_FOLDS, **spec)
    version = MODEL_VERSION_WEEKLY if granularity == "weekly" else MODEL_VERSION_DAILY
    run_id = new_run_id(granularity)
    tracker = tracker or ExperimentTracker()

    fold_ranges = [
        {"fold": i + 1,
         "train": f"{train_df[DS_COL].iloc[0].date()}..{train_df[DS_COL].iloc[o - 1].date()}",
         "validation": f"{train_df[DS_COL].iloc[o].date()}..{train_df[DS_COL].iloc[e - 1].date()}"}
        for i, (o, e) in enumerate(folds)
    ]
    logger.info("[TUNING] run=%s granularity=%s configs=%d folds=%d (min_train=%d horizon=%d step=%d)",
                run_id, granularity, len(param_grid), len(folds),
                spec["min_train"], spec["horizon"], spec["step"])
    logger.info("[TUNING] Training range %s | hold-out boundary: %s | test data NOT visible.",
                _range(train_df), holdout_start)

    trials: List[Dict[str, Any]] = []
    t_start = time.time()
    with AccessAuditor(holdout_start=holdout_start, strict=holdout_start is not None,
                       label=f"tuning-{granularity}") as auditor:
        for idx, params in enumerate(param_grid, start=1):
            cid = config_id(params)
            folds_res = run_cv(train_df, params, granularity, folds=folds,
                               holdout_start=holdout_start, seed=seed)
            for fr in folds_res:
                tracker.log(
                    experiment_id=f"{run_id}/{cid}/fold{fr['fold_index']}",
                    run_id=run_id, stage="cv_fold", config_id=cid, fold_index=fr["fold_index"],
                    model_version=version, granularity=granularity, parameters=params,
                    training_range=f"{fr['train_start']}..{fr['train_end']}",
                    validation_range=f"{fr['val_start']}..{fr['val_end']}",
                    n_train=fr["n_train"], n_val=fr["n_val"],
                    MAE=fr["MAE"], RMSE=fr["RMSE"], interval_coverage=fr["interval_coverage"],
                    dataset_version=DATASET_VERSION, region=REGION, seed=seed,
                )
            maes = np.array([f["MAE"] for f in folds_res])
            rmses = np.array([f["RMSE"] for f in folds_res])
            trial = {
                "trial_index": idx,
                "config_id": cid,
                "granularity": granularity,
                "params": dict(params),
                "n_folds": len(folds_res),
                "avg_cv_mae": float(round(maes.mean(), 4)),
                "std_cv_mae": float(round(maes.std(ddof=1), 4)) if len(maes) > 1 else 0.0,
                "avg_cv_rmse": float(round(rmses.mean(), 4)),
                "avg_interval_coverage": float(round(np.mean([f["interval_coverage"] for f in folds_res]), 4)),
                "avg_interval_ratio_q95": float(round(np.mean([f["interval_ratio_q95"] for f in folds_res]), 4)),
                "fold_details": folds_res,
            }
            trials.append(trial)
            tracker.log(
                experiment_id=f"{run_id}/{cid}/aggregate",
                run_id=run_id, stage="cv_aggregate", config_id=cid, fold_index=None,
                model_version=version, granularity=granularity, parameters=params,
                training_range=f"{folds_res[0]['train_start']}..{folds_res[-1]['train_end']}",
                validation_range=f"{folds_res[0]['val_start']}..{folds_res[-1]['val_end']}",
                n_train=None, n_val=sum(f["n_val"] for f in folds_res),
                MAE=trial["avg_cv_mae"], RMSE=trial["avg_cv_rmse"],
                interval_coverage=trial["avg_interval_coverage"],
                dataset_version=DATASET_VERSION, region=REGION, seed=seed,
            )
            if idx % 10 == 0 or idx == len(param_grid):
                logger.info("[TUNING] %d/%d configs done (%.0fs elapsed)",
                            idx, len(param_grid), time.time() - t_start)
    tracker.flush()

    # ── Selection: validation performance ONLY ───────────────────────────
    ranked = sorted(trials, key=lambda t: (t["avg_cv_mae"], t["avg_cv_rmse"], t["config_id"]))
    best = ranked[0]
    default_trial = next((t for t in trials if t["params"] == DEFAULT_PARAMS_FOR_REFERENCE), None)

    edge_flags = []
    for key in ("changepoint_prior_scale", "seasonality_prior_scale", "holidays_prior_scale"):
        vals = PARAM_SEARCH_SPACE[key]
        if best["params"].get(key) in (min(vals), max(vals)) and len(set(vals)) > 2:
            edge_flags.append(f"{key}={best['params'][key]} is at the edge of the searched range {vals}")

    summary = {
        "run_id": run_id,
        "granularity": granularity,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": f"mean validation {SELECTION_METRIC} across rolling-origin folds "
                            "(tie-break: mean RMSE, then config_id)",
        "test_data_used": False,
        "training_range": _range(train_df),
        "n_train_rows": int(len(train_df)),
        "holdout_start": str(holdout_start) if holdout_start is not None else None,
        "cv_spec": {**spec, "scheme": "expanding window / rolling origin", "n_folds": len(folds),
                    "folds": fold_ranges},
        "search_space": {k: [list(v) if isinstance(v, tuple) else v for v in vs]
                         for k, vs in PARAM_SEARCH_SPACE.items()},
        "n_configs": len(param_grid),
        "best_config_id": best["config_id"],
        "best_params": best["params"],
        "best_cv": {k: best[k] for k in ("avg_cv_mae", "std_cv_mae", "avg_cv_rmse",
                                          "avg_interval_coverage", "avg_interval_ratio_q95")},
        "default_params_cv": ({k: default_trial[k] for k in ("avg_cv_mae", "avg_cv_rmse")}
                              if default_trial else None),
        "recommended_sensitivity_from_cv": best["avg_interval_ratio_q95"],
        "recommended_sensitivity_note": (
            "Multiplier on the interval half-widths that would have given ~95% coverage on the "
            "training-only validation folds. Informational only; the shipped default stays 1.0."),
        "search_boundary_warnings": edge_flags,
        "top10": [{"rank": r + 1, "config_id": t["config_id"], "params": t["params"],
                   "avg_cv_mae": t["avg_cv_mae"], "std_cv_mae": t["std_cv_mae"],
                   "avg_cv_rmse": t["avg_cv_rmse"]} for r, t in enumerate(ranked[:10])],
        "access_audit": auditor.summary(),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    logger.info("[TUNING] BEST %s  params=%s  CV MAE=%.2f  RMSE=%.2f",
                best["config_id"], best["params"], best["avg_cv_mae"], best["avg_cv_rmse"])
    logger.info("[TUNING] Access audit: %s", summary["access_audit"])

    if save:
        path = ARTIFACT_METADATA_DIR / f"tuning_{granularity}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("[TUNING] Tuning summary saved: %s", path)

    return {
        "best_params": best["params"],
        "best_trial": best,
        "trials": trials,
        "ranked": ranked,
        "summary": summary,
        "run_id": run_id,
        "access_audit": auditor.summary(),
    }


def run_hyperparameter_tuning(
    train_df: pd.DataFrame,
    param_grid: Optional[List[Dict[str, Any]]] = None,
    granularity: str = "daily",
    save: bool = True,
    holdout_start: Optional[pd.Timestamp] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Backwards-compatible (best_params, all_trials) wrapper around run_tuning()."""
    res = run_tuning(train_df, granularity=granularity, param_grid=param_grid,
                     holdout_start=holdout_start, save=save, **kwargs)
    return res["best_params"], res["trials"]
