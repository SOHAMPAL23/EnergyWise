"""
ml/eda/analysis.py
───────────────────
Exploratory Data Analysis — Statistical Computations

All computations are derived from the ACTUAL dataset.
Nothing is fabricated. If a statistic cannot be computed, it is explicitly
marked as "NOT AVAILABLE" in the output.

Analyzes:
  - Overall demand trend (rolling mean/std)
  - Weekly seasonality (day-of-week pattern)
  - Yearly seasonality (monthly pattern — only if ≥2 years of data)
  - Annual demand by year
  - Demand distribution
  - Extreme values (top/bottom percentiles)
  - Missingness assessment

Returns structured result dict suitable for report generation.
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from ml.configs.config import DS_COL, Y_COL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# Minimum years of data required to report yearly seasonality
MIN_YEARS_FOR_YEARLY = 2
# Rolling window for trend analysis
ROLLING_WINDOW_DAYS = 30
EXTREME_PCT_HIGH = 95
EXTREME_PCT_LOW = 5


def run_eda_analysis(
    df: pd.DataFrame,
    ds_col: str = DS_COL,
    y_col: str = Y_COL,
) -> Dict[str, Any]:
    """
    Perform full EDA analysis on the daily time series.

    Parameters
    ----------
    df     : Daily Prophet DataFrame with [ds, y].
    ds_col : Timestamp column.
    y_col  : Demand column.

    Returns
    -------
    dict — Structured EDA findings (all values from actual data).
    """
    logger.info("[EDA] Starting exploratory data analysis on %d daily records.", len(df))
    eda = df.copy()
    eda[ds_col] = pd.to_datetime(eda[ds_col])

    results: Dict[str, Any] = {}

    # ── 1. Basic statistics ───────────────────────────────────────────────
    desc = eda[y_col].describe()
    results["basic_stats"] = {
        "n_rows": int(desc["count"]),
        "mean_mw": round(float(desc["mean"]), 2),
        "std_mw": round(float(desc["std"]), 2),
        "min_mw": round(float(desc["min"]), 2),
        "q5_mw": round(float(np.nanpercentile(eda[y_col].dropna(), EXTREME_PCT_LOW)), 2),
        "q25_mw": round(float(desc["25%"]), 2),
        "median_mw": round(float(desc["50%"]), 2),
        "q75_mw": round(float(desc["75%"]), 2),
        "q95_mw": round(float(np.nanpercentile(eda[y_col].dropna(), EXTREME_PCT_HIGH)), 2),
        "max_mw": round(float(desc["max"]), 2),
    }
    logger.info("[EDA] Basic stats: mean=%.1f MW, std=%.1f MW, min=%.1f MW, max=%.1f MW",
                results["basic_stats"]["mean_mw"],
                results["basic_stats"]["std_mw"],
                results["basic_stats"]["min_mw"],
                results["basic_stats"]["max_mw"])

    # ── 2. Date range and data coverage ──────────────────────────────────
    date_min = eda[ds_col].min()
    date_max = eda[ds_col].max()
    n_years = (date_max - date_min).days / 365.25
    results["date_range"] = {
        "start": str(date_min.date()),
        "end": str(date_max.date()),
        "n_days": int((date_max - date_min).days),
        "n_years": round(n_years, 2),
        "n_records": len(eda),
    }

    # ── 3. Rolling mean and std ───────────────────────────────────────────
    eda["rolling_mean"] = eda[y_col].rolling(window=ROLLING_WINDOW_DAYS, min_periods=1).mean()
    eda["rolling_std"] = eda[y_col].rolling(window=ROLLING_WINDOW_DAYS, min_periods=1).std()
    results["rolling_window"] = ROLLING_WINDOW_DAYS

    # ── 4. Day-of-week seasonality ────────────────────────────────────────
    eda["dayofweek"] = eda[ds_col].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = (
        eda.groupby("dayofweek")[y_col]
        .agg(["mean", "std"])
        .reindex(dow_order)
        .rename(columns={"mean": "mean_mw", "std": "std_mw"})
        .round(2)
    )
    results["day_of_week"] = dow_stats.to_dict(orient="index")
    logger.info("[EDA] Day-of-week seasonality computed.")

    # ── 5. Monthly seasonality ────────────────────────────────────────────
    eda["month"] = eda[ds_col].dt.month
    eda["month_name"] = eda[ds_col].dt.strftime("%b")
    month_stats = (
        eda.groupby("month")[y_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_mw", "std": "std_mw"})
        .round(2)
    )
    results["monthly"] = month_stats.to_dict(orient="index")
    logger.info("[EDA] Monthly seasonality computed.")

    # ── 6. Yearly seasonality — only if ≥ 2 years of data ────────────────
    eda["year"] = eda[ds_col].dt.year
    if n_years >= MIN_YEARS_FOR_YEARLY:
        yearly_stats = (
            eda.groupby("year")[y_col]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_mw", "std": "std_mw", "count": "n_days"})
            .round(2)
        )
        results["yearly"] = yearly_stats.to_dict(orient="index")
        results["yearly_seasonality_sufficient"] = True
        logger.info("[EDA] Yearly analysis computed (%.1f years of data).", n_years)
    else:
        results["yearly"] = {}
        results["yearly_seasonality_sufficient"] = False
        logger.warning(
            "[EDA] Insufficient data for yearly seasonality (%.1f years < %d required).",
            n_years, MIN_YEARS_FOR_YEARLY,
        )

    # ── 7. Extreme demand analysis ────────────────────────────────────────
    threshold_high = float(np.nanpercentile(eda[y_col].dropna(), EXTREME_PCT_HIGH))
    threshold_low = float(np.nanpercentile(eda[y_col].dropna(), EXTREME_PCT_LOW))

    high_demand = eda[eda[y_col] > threshold_high][[ds_col, y_col]].copy()
    low_demand = eda[eda[y_col] < threshold_low][[ds_col, y_col]].copy()

    results["extremes"] = {
        "threshold_high_mw": round(threshold_high, 2),
        "threshold_low_mw": round(threshold_low, 2),
        "n_high_demand_days": len(high_demand),
        "n_low_demand_days": len(low_demand),
        "top5_high": high_demand.nlargest(5, y_col).assign(
            **{ds_col: high_demand.nlargest(5, y_col)[ds_col].astype(str)}
        ).values.tolist(),
        "top5_low": low_demand.nsmallest(5, y_col).assign(
            **{ds_col: low_demand.nsmallest(5, y_col)[ds_col].astype(str)}
        ).values.tolist(),
    }
    logger.info(
        "[EDA] Extremes: %d high-demand days (>%.0f MW), %d low-demand days (<%.0f MW).",
        len(high_demand), threshold_high, len(low_demand), threshold_low,
    )

    # ── 8. Missingness in daily series ───────────────────────────────────
    n_null_y = int(eda[y_col].isna().sum())
    results["missingness"] = {
        "n_null_daily": n_null_y,
        "null_pct_daily": round(100 * n_null_y / len(eda), 4) if len(eda) > 0 else 0.0,
    }

    # Attach processed eda df for plotting
    results["_eda_df"] = eda

    logger.info("[EDA] Analysis complete.")
    return results
