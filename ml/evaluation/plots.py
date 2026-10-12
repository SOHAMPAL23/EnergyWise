"""
ml/evaluation/plots.py
───────────────────────
Final Evaluation Visualization

Generates publication-quality plots for the final evaluation:
  1. Actual vs Prophet forecast with uncertainty bounds (train/test boundary)
  2. Forecast uncertainty bands zoom
  3. Weekly aggregated comparison
  4. Largest error periods
  5. Error distribution histogram
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from ml.configs.config import DS_COL, Y_COL, FIGURES_DIR, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL, make_output_dirs

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

PALETTE = {
    "actual": "#0f172a",
    "prophet": "#0d9488",
    "interval": "#14b8a6",
    "naive": "#f59e0b",
    "snaive": "#3b82f6",
    "error_pos": "#ef4444",
    "error_neg": "#3b82f6",
    "train_region": "#f0fdf4",
    "test_region": "#fef9f0",
    "grid": "#e2e8f0",
    "boundary": "#94a3b8",
}
FIG_DPI = 150


def plot_final_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    final_preds: pd.DataFrame,
    split_info,
    granularity: str = "daily",
    save: bool = True,
) -> None:
    """
    Generate all final evaluation plots.

    Parameters
    ----------
    train_df    : Training DataFrame (ds, y).
    test_df     : Test DataFrame (ds, y).
    final_preds : Output from evaluation.run_final_evaluation (merged predictions).
    split_info  : SplitInfo from splitting.py.
    granularity : "daily" or "weekly".
    save        : If True, save to FIGURES_DIR.
    """
    make_output_dirs()

    _plot_actual_vs_forecast(train_df, test_df, final_preds, split_info, granularity, save)
    _plot_uncertainty_zoom(final_preds, granularity, save)
    _plot_weekly_aggregated(test_df, final_preds, granularity, save)
    _plot_largest_errors(final_preds, granularity, save)
    _plot_error_distribution(final_preds, granularity, save)
    _plot_error_periods(final_preds, granularity, save)


def _plot_actual_vs_forecast(train_df, test_df, final_preds, split_info, granularity, save):
    """Full actual vs forecast plot with train/test boundary."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Training history (last 365 days / 52 weeks for context)
    train_tail = train_df.tail(365) if granularity == "daily" else train_df.tail(52)
    ax.plot(train_tail[DS_COL], train_tail[Y_COL], color=PALETTE["actual"],
            linewidth=1.0, alpha=0.5, label="Training Demand")

    # Test actuals
    ax.plot(final_preds[DS_COL], final_preds["actual"], color=PALETTE["actual"],
            linewidth=1.5, label="Actual Demand (Test)", zorder=3)

    # Prophet forecast
    ax.plot(final_preds[DS_COL], final_preds["prophet_yhat"], color=PALETTE["prophet"],
            linewidth=2.0, label="Prophet Forecast", zorder=4)

    # Uncertainty interval
    ax.fill_between(final_preds[DS_COL],
                    final_preds["prophet_yhat_lower"],
                    final_preds["prophet_yhat_upper"],
                    color=PALETTE["interval"], alpha=0.25, label="95% Prediction Interval")

    # Train/Test boundary line
    boundary = pd.to_datetime(split_info.split_timestamp)
    ax.axvline(boundary, color=PALETTE["boundary"], linewidth=2, linestyle="--", alpha=0.8)
    ax.text(boundary, ax.get_ylim()[0], " ← Train | Test →",
            color=PALETTE["boundary"], fontsize=9, va="bottom")

    ax.set_title(f"Actual vs Prophet Forecast — {granularity.capitalize()} Energy Demand (Germany)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Load (MW)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    _save_fig(fig, f"eval_01_actual_vs_forecast_{granularity}.png", save)


def _plot_uncertainty_zoom(final_preds, granularity, save):
    """Zoom plot: forecast uncertainty intervals."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(final_preds[DS_COL], final_preds["actual"], color=PALETTE["actual"],
            linewidth=1.5, label="Actual", zorder=3)
    ax.plot(final_preds[DS_COL], final_preds["prophet_yhat"], color=PALETTE["prophet"],
            linewidth=2.0, label="Prophet Forecast", zorder=4)
    ax.fill_between(final_preds[DS_COL],
                    final_preds["prophet_yhat_lower"],
                    final_preds["prophet_yhat_upper"],
                    color=PALETTE["interval"], alpha=0.35, label="95% PI")
    ax.set_title(f"Forecast Uncertainty Intervals — {granularity.capitalize()} (95% PI)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Load (MW)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    _save_fig(fig, f"eval_02_uncertainty_{granularity}.png", save)


def _plot_weekly_aggregated(test_df, final_preds, granularity, save):
    """Weekly-aggregated (complete Mon–Sun weeks) actual vs all three models."""
    from ml.evaluation.evaluation import weekly_aggregate
    if granularity == "weekly":
        wk = final_preds[[DS_COL, "actual", "prophet_yhat", "naive_yhat", "snaive_yhat"]].copy()
        fname = "eval_03_weekly_comparison.png"
    else:
        wk = weekly_aggregate(final_preds)
        fname = f"eval_03_weekly_aggregated_{granularity}.png"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wk[DS_COL], wk["actual"], color=PALETTE["actual"], linewidth=1.8, label="Actual (weekly mean)")
    ax.plot(wk[DS_COL], wk["prophet_yhat"], color=PALETTE["prophet"], linewidth=1.6, label="Prophet")
    ax.plot(wk[DS_COL], wk["snaive_yhat"], color=PALETTE["snaive"], linewidth=1.0, linestyle="-.",
            alpha=0.8, label="Seasonal-Naive")
    ax.plot(wk[DS_COL], wk["naive_yhat"], color=PALETTE["naive"], linewidth=1.0, linestyle="--",
            alpha=0.8, label="Naive")
    ax.set_title("Weekly-aggregated comparison on hold-out test (complete Mon–Sun weeks)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Week start")
    ax.set_ylabel("Mean load (MW)")
    ax.legend(fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    _save_fig(fig, fname, save)

def _plot_largest_errors(final_preds, granularity, save):
    """Highlight the 20 largest absolute errors."""
    temp = final_preds.copy()
    temp["abs_error"] = temp["prophet_error"].abs()
    top20 = temp.nlargest(20, "abs_error")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(temp[DS_COL], temp["actual"], color=PALETTE["actual"],
            linewidth=1.0, alpha=0.6, label="Actual")
    ax.plot(temp[DS_COL], temp["prophet_yhat"], color=PALETTE["prophet"],
            linewidth=1.0, alpha=0.8, label="Prophet")
    pos_err = top20[top20["prophet_error"] > 0]
    neg_err = top20[top20["prophet_error"] < 0]
    if not pos_err.empty:
        ax.scatter(pos_err[DS_COL], pos_err["actual"], color=PALETTE["error_pos"],
                   s=50, zorder=5, label="Underestimate (actual > forecast)")
    if not neg_err.empty:
        ax.scatter(neg_err[DS_COL], neg_err["actual"], color=PALETTE["error_neg"],
                   s=50, zorder=5, label="Overestimate (actual < forecast)")
    ax.set_title(f"Top 20 Largest Forecast Errors — {granularity.capitalize()}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.legend(fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    _save_fig(fig, f"eval_04_largest_errors_{granularity}.png", save)


def _plot_error_distribution(final_preds, granularity, save):
    """Error distribution histogram."""
    errors = final_preds["prophet_error"].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=40, color=PALETTE["prophet"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=PALETTE["actual"], linewidth=2, linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color=PALETTE["error_pos"], linewidth=1.5, linestyle="-.",
               label=f"Mean error: {errors.mean():+.0f} MW")
    ax.set_title(f"Forecast Error Distribution (Actual − Forecast) — {granularity.capitalize()}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Error (MW)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5)
    plt.tight_layout()
    _save_fig(fig, f"eval_05_error_distribution_{granularity}.png", save)


def _save_fig(fig, filename, save):
    plt.tight_layout()
    if save:
        fpath = FIGURES_DIR / filename
        fig.savefig(fpath, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("[EVAL PLOTS] Saved: %s", fpath)
    plt.close(fig)


def _plot_error_periods(final_preds, granularity, save):
    """Prophet vs Seasonal-Naive MAE per calendar month — shows WHEN the model fails."""
    t = final_preds.copy()
    t[DS_COL] = pd.to_datetime(t[DS_COL])
    t["month"] = t[DS_COL].dt.to_period("M").astype(str)
    g = t.groupby("month").agg(
        prophet=("prophet_error", lambda e: float(np.mean(np.abs(e)))),
        snaive=("snaive_error", lambda e: float(np.mean(np.abs(e)))),
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["prophet"], width=0.4, color=PALETTE["prophet"], label="Prophet MAE")
    ax.bar(x + 0.2, g["snaive"], width=0.4, color=PALETTE["snaive"], label="Seasonal-Naive MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("MAE (MW)")
    ax.set_title(f"Error by period — {granularity.capitalize()} (hold-out test)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend()
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5)
    plt.tight_layout()
    _save_fig(fig, f"eval_06_error_periods_{granularity}.png", save)