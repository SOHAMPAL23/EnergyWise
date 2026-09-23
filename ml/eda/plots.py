"""
ml/eda/plots.py
────────────────
EDA Visualization

Creates and saves all EDA figures to reports/figures/.
All plots use the ACTUAL dataset — no fabricated values.

Figures generated:
  1. eda_01_demand_over_time.png      — Full time series with 30-day rolling mean
  2. eda_02_rolling_mean.png          — 30-day rolling mean & std
  3. eda_03_day_of_week.png           — Day-of-week average demand
  4. eda_04_monthly_seasonality.png   — Monthly average demand
  5. eda_05_yearly_analysis.png       — Annual mean demand (if ≥2 years)
  6. eda_06_demand_distribution.png   — Demand histogram + KDE
  7. eda_07_extreme_demand.png        — High / low demand period highlights

EDA does NOT modify any dataset.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ml.configs.config import (
    DS_COL,
    Y_COL,
    FIGURES_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#0d9488",      # Teal
    "secondary": "#14b8a6",    # Light teal
    "accent": "#0f172a",       # Dark navy
    "muted": "#94a3b8",        # Slate gray
    "highlight_high": "#ef4444",  # Red
    "highlight_low": "#3b82f6",   # Blue
    "grid": "#e2e8f0",         # Light grey
}

FIG_DPI = 150
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (10, 6)


def plot_all_eda(eda_results: Dict[str, Any], save: bool = True) -> None:
    """
    Generate all EDA figures from the analysis results dict.

    Parameters
    ----------
    eda_results : Output from eda.analysis.run_eda_analysis().
    save        : If True, save figures to FIGURES_DIR.
    """
    make_output_dirs()
    eda_df = eda_results.get("_eda_df")
    if eda_df is None:
        logger.error("[EDA PLOTS] No _eda_df found in results — cannot generate plots.")
        return

    _plot_demand_over_time(eda_df, eda_results, save)
    _plot_rolling_stats(eda_df, eda_results, save)
    _plot_day_of_week(eda_results, save)
    _plot_monthly_seasonality(eda_results, save)
    if eda_results.get("yearly_seasonality_sufficient"):
        _plot_yearly_analysis(eda_results, save)
    else:
        logger.info("[EDA PLOTS] Skipping yearly plot — insufficient data.")
    _plot_demand_distribution(eda_df, eda_results, save)
    _plot_extreme_demand(eda_df, eda_results, save)

    logger.info("[EDA PLOTS] All EDA figures saved to %s", FIGURES_DIR)


# ─── Individual plot functions ────────────────────────────────────────────────

def _plot_demand_over_time(eda_df, results, save):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(eda_df[DS_COL], eda_df[Y_COL], color=PALETTE["muted"], alpha=0.5,
            linewidth=0.8, label="Daily Demand")
    ax.plot(eda_df[DS_COL], eda_df["rolling_mean"], color=PALETTE["primary"],
            linewidth=2.0, label=f"{results['rolling_window']}-day Rolling Mean")
    ax.set_title("Germany Electricity Demand — Daily Load with Rolling Mean",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Load (MW)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _finalize(fig, "eda_01_demand_over_time.png", save)


def _plot_rolling_stats(eda_df, results, save):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    w = results["rolling_window"]

    axes[0].plot(eda_df[DS_COL], eda_df["rolling_mean"], color=PALETTE["primary"],
                 linewidth=1.8, label=f"{w}-day Rolling Mean")
    axes[0].set_title(f"{w}-Day Rolling Mean Load (MW)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Mean Load (MW)")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, color=PALETTE["grid"], linewidth=0.5)

    axes[1].plot(eda_df[DS_COL], eda_df["rolling_std"], color=PALETTE["highlight_high"],
                 linewidth=1.8, label=f"{w}-day Rolling Std")
    axes[1].set_title(f"{w}-Day Rolling Standard Deviation (MW)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].set_ylabel("Std Dev (MW)")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, color=PALETTE["grid"], linewidth=0.5)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    _finalize(fig, "eda_02_rolling_stats.png", save)


def _plot_day_of_week(results, save):
    dow_data = results.get("day_of_week", {})
    if not dow_data:
        return
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    means = [dow_data.get(d, {}).get("mean_mw", 0) for d in days]
    stds = [dow_data.get(d, {}).get("std_mw", 0) for d in days]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    bars = ax.bar(days, means, color=[PALETTE["primary"] if i < 5 else PALETTE["muted"] for i in range(7)],
                  edgecolor="white", linewidth=1)
    ax.errorbar(days, means, yerr=stds, fmt="none", color=PALETTE["accent"],
                capsize=4, linewidth=1.5)
    ax.set_title("Mean Electricity Demand by Day of Week", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Mean Load (MW)", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    _finalize(fig, "eda_03_day_of_week.png", save)


def _plot_monthly_seasonality(results, save):
    monthly = results.get("monthly", {})
    if not monthly:
        return
    months = sorted(monthly.keys())
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    means = [monthly[m]["mean_mw"] for m in months]
    stds = [monthly[m]["std_mw"] for m in months]
    month_labels = [labels[m - 1] for m in months]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.plot(month_labels, means, color=PALETTE["primary"], marker="o",
            linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(month_labels,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=PALETTE["secondary"], alpha=0.2, label="±1 Std Dev")
    ax.set_title("Mean Electricity Demand by Month (Annual Seasonality)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Mean Load (MW)", fontsize=11)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.legend(fontsize=10)
    _finalize(fig, "eda_04_monthly_seasonality.png", save)


def _plot_yearly_analysis(results, save):
    yearly = results.get("yearly", {})
    if not yearly:
        return
    years = sorted(yearly.keys())
    means = [yearly[y]["mean_mw"] for y in years]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    bars = ax.bar([str(y) for y in years], means, color=PALETTE["primary"],
                  edgecolor="white", linewidth=1)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Mean Annual Electricity Demand by Year", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Mean Load (MW)", fontsize=11)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    _finalize(fig, "eda_05_yearly_analysis.png", save)


def _plot_demand_distribution(eda_df, results, save):
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    data = eda_df[Y_COL].dropna()
    ax.hist(data, bins=50, color=PALETTE["secondary"], edgecolor=PALETTE["accent"],
            alpha=0.8, density=False)
    mean_val = data.mean()
    ax.axvline(mean_val, color=PALETTE["highlight_high"], linewidth=2, linestyle="--",
               label=f"Mean: {mean_val:,.0f} MW")
    ax.axvline(data.median(), color=PALETTE["primary"], linewidth=2, linestyle="-.",
               label=f"Median: {data.median():,.0f} MW")
    ax.set_title("Distribution of Daily Electricity Demand (MW)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Daily Mean Load (MW)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    _finalize(fig, "eda_06_demand_distribution.png", save)


def _plot_extreme_demand(eda_df, results, save):
    extremes = results.get("extremes", {})
    if not extremes:
        return

    high_thresh = extremes["threshold_high_mw"]
    low_thresh = extremes["threshold_low_mw"]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(eda_df[DS_COL], eda_df[Y_COL], color=PALETTE["muted"], alpha=0.5,
            linewidth=0.8, label="Daily Demand")
    ax.axhline(high_thresh, color=PALETTE["highlight_high"], linewidth=1.5, linestyle="--",
               label=f"95th Pct ({high_thresh:,.0f} MW)")
    ax.axhline(low_thresh, color=PALETTE["highlight_low"], linewidth=1.5, linestyle="--",
               label=f"5th Pct ({low_thresh:,.0f} MW)")

    high_mask = eda_df[Y_COL] > high_thresh
    low_mask = eda_df[Y_COL] < low_thresh
    ax.scatter(eda_df.loc[high_mask, DS_COL], eda_df.loc[high_mask, Y_COL],
               color=PALETTE["highlight_high"], s=10, zorder=5, alpha=0.7,
               label=f"High-demand days (n={high_mask.sum()})")
    ax.scatter(eda_df.loc[low_mask, DS_COL], eda_df.loc[low_mask, Y_COL],
               color=PALETTE["highlight_low"], s=10, zorder=5, alpha=0.7,
               label=f"Low-demand days (n={low_mask.sum()})")

    ax.set_title("Extreme Demand Days — 5th and 95th Percentile Thresholds", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Load (MW)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _finalize(fig, "eda_07_extreme_demand.png", save)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _finalize(fig, filename: str, save: bool):
    plt.tight_layout()
    if save:
        fpath = FIGURES_DIR / filename
        fig.savefig(fpath, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("[EDA PLOTS] Saved: %s", fpath)
    plt.close(fig)
