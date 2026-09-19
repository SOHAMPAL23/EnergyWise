"""
ml/baselines/evaluate.py
─────────────────────────
Baseline Model Evaluation

Trains, predicts, evaluates, and reports Naive and Seasonal-Naive forecasters
using the SAME test period that will be used for Prophet.

All metrics (MAE, RMSE) are computed from ACTUAL values — never fabricated.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ml.configs.config import (
    DS_COL,
    Y_COL,
    SEASONAL_PERIOD_DAILY,
    SEASONAL_PERIOD_WEEKLY,
    FIGURES_DIR,
    BASELINES_REPORT_DIR,
    METRICS_DIR,
    FORECASTS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)
from ml.baselines.naive import NaiveForecaster
from ml.baselines.seasonal_naive import SeasonalNaiveForecaster
from ml.evaluation.metrics import mae, rmse

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

PALETTE = {
    "actual": "#0f172a",
    "naive": "#f59e0b",
    "snaive": "#3b82f6",
    "grid": "#e2e8f0",
}


def run_baseline_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    granularity: str = "daily",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Run Naive and Seasonal-Naive baselines on the holdout test set.

    Parameters
    ----------
    train_df    : Training DataFrame (ds, y).
    test_df     : Test DataFrame (ds, y) — the SAME test window as Prophet.
    granularity : "daily" or "weekly" — determines seasonal period.
    save        : If True, save metrics, predictions, and plots.

    Returns
    -------
    dict with keys:
      naive_mae, naive_rmse, snaive_mae, snaive_rmse,
      naive_preds (DataFrame), snaive_preds (DataFrame)
    """
    make_output_dirs()

    period = SEASONAL_PERIOD_DAILY if granularity == "daily" else SEASONAL_PERIOD_WEEKLY
    y_true = test_df[Y_COL].values

    logger.info("[BASELINES] Evaluating %s baselines (period=%d) on %d test records ...",
                granularity, period, len(test_df))

    # ── Naive ─────────────────────────────────────────────────────────────
    naive_model = NaiveForecaster()
    naive_model.fit(train_df)
    naive_preds = naive_model.predict(test_df)
    naive_mae_val = mae(y_true, naive_preds["yhat"].values)
    naive_rmse_val = rmse(y_true, naive_preds["yhat"].values)

    # ── Seasonal-Naive ────────────────────────────────────────────────────
    snaive_model = SeasonalNaiveForecaster(period=period)
    snaive_model.fit(train_df)
    snaive_preds = snaive_model.predict(test_df)
    snaive_mae_val = mae(y_true, snaive_preds["yhat"].values)
    snaive_rmse_val = rmse(y_true, snaive_preds["yhat"].values)

    logger.info("[BASELINES] %s Naive         -> MAE: %.2f MW | RMSE: %.2f MW",
                granularity.upper(), naive_mae_val, naive_rmse_val)
    logger.info("[BASELINES] %s Seasonal-Naive -> MAE: %.2f MW | RMSE: %.2f MW",
                granularity.upper(), snaive_mae_val, snaive_rmse_val)

    results = {
        "granularity": granularity,
        "period": period,
        "n_test": len(test_df),
        "naive_mae": round(naive_mae_val, 4),
        "naive_rmse": round(naive_rmse_val, 4),
        "snaive_mae": round(snaive_mae_val, 4),
        "snaive_rmse": round(snaive_rmse_val, 4),
        "naive_preds": naive_preds,
        "snaive_preds": snaive_preds,
    }

    if save:
        _save_baseline_results(results, test_df, granularity)
        _plot_baselines(train_df, test_df, naive_preds, snaive_preds, granularity)

    return results


def _save_baseline_results(results: Dict, test_df: pd.DataFrame, granularity: str):
    """Save baseline metrics and predictions."""
    # Save baseline comparison CSV
    comparison = pd.DataFrame([
        {"Model": "Naive Baseline", "MAE_MW": results["naive_mae"],
         "RMSE_MW": results["naive_rmse"], "Granularity": granularity},
        {"Model": f"Seasonal-Naive (period={results['period']})", "MAE_MW": results["snaive_mae"],
         "RMSE_MW": results["snaive_rmse"], "Granularity": granularity},
    ])
    csv_path = BASELINES_REPORT_DIR / f"baseline_comparison_{granularity}.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("[BASELINES] Comparison saved: %s", csv_path)

    # Save naive predictions
    naive_out = results["naive_preds"].copy()
    naive_out["actual"] = test_df[Y_COL].values
    naive_out.to_csv(FORECASTS_DIR / f"naive_predictions_{granularity}.csv", index=False)

    # Save seasonal-naive predictions
    snaive_out = results["snaive_preds"].copy()
    snaive_out["actual"] = test_df[Y_COL].values
    snaive_out.to_csv(FORECASTS_DIR / f"seasonal_naive_predictions_{granularity}.csv", index=False)

    # Save metrics JSON
    metrics_data = {k: v for k, v in results.items() if not isinstance(v, pd.DataFrame)}
    with open(METRICS_DIR / f"baseline_metrics_{granularity}.json", "w") as fh:
        json.dump(metrics_data, fh, indent=2, default=str)


def _plot_baselines(train_df, test_df, naive_preds, snaive_preds, granularity):
    """Generate actual vs baseline forecast plot."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(test_df[DS_COL], test_df[Y_COL], color=PALETTE["actual"],
            linewidth=1.5, label="Actual Demand", zorder=3)
    ax.plot(naive_preds[DS_COL], naive_preds["yhat"], color=PALETTE["naive"],
            linewidth=1.5, linestyle="--", label="Naive Forecast", alpha=0.9)
    ax.plot(snaive_preds[DS_COL], snaive_preds["yhat"], color=PALETTE["snaive"],
            linewidth=1.5, linestyle="-.", label="Seasonal-Naive Forecast", alpha=0.9)

    ax.set_title(f"Baseline Forecasters vs Actual Demand — {granularity.capitalize()}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Load (MW)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()

    fpath = FIGURES_DIR / f"baseline_forecast_{granularity}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[BASELINES] Plot saved: %s", fpath)
