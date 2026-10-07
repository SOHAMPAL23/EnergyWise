"""
ml/anomaly/detector.py
───────────────────────
Prediction-interval anomaly detection.

    actual <  lower_bound                    -> DROP_ANOMALY
    actual >  upper_bound                    -> SPIKE_ANOMALY
    lower_bound <= actual <= upper_bound     -> NORMAL

Bounds come from the forecast's prediction interval, scaled by `sensitivity`:

    lower_bound = yhat - sensitivity * (yhat - yhat_lower)
    upper_bound = yhat + sensitivity * (yhat_upper - yhat)

    sensitivity = 1.0  -> raw interval (default)
    sensitivity > 1.0  -> wider bounds, fewer alerts
    sensitivity < 1.0  -> narrower bounds, more alerts

IMPORTANT
  * Anomalies are INVESTIGATION SIGNALS, not ground truth. No labelled anomalies exist
    in this dataset, so NO precision/recall/F1 is reported (an earlier version scored the
    detector against a 2.5-sigma label derived from the same test residuals, which is
    circular and has been removed).
  * `sensitivity` is a user setting. It must not be tuned on the hold-out test set.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ml.configs.config import (
    DS_COL,
    Y_COL,
    ANOMALY_SENSITIVITY,
    FORECASTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Anomaly classifications are investigation signals only. They are NOT ground truth and "
    "do not confirm abnormal grid behaviour. They flag observations outside a model-based "
    "prediction interval; model drift, regime change (e.g. COVID-19 in 2020) or interval "
    "mis-calibration will also trigger them. Review flagged periods with a domain expert."
)

PALETTE = {
    "actual": "#0f172a", "forecast": "#0d9488", "interval": "#14b8a6",
    "spike": "#ef4444", "drop": "#3b82f6", "grid": "#e2e8f0",
}

OUTPUT_COLUMNS = ["timestamp", "ds", "actual", "yhat", "lower_bound", "upper_bound",
                  "deviation", "deviation_magnitude", "breach_magnitude", "classification"]


def classify_anomalies(
    actual: pd.Series, lower: pd.Series, upper: pd.Series
) -> pd.Series:
    """Vectorised classification; bounds are inclusive for NORMAL."""
    out = pd.Series("NORMAL", index=actual.index, dtype=object)
    out[actual < lower] = "DROP_ANOMALY"
    out[actual > upper] = "SPIKE_ANOMALY"
    return out


def detect_anomalies(
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    sensitivity: float = ANOMALY_SENSITIVITY,
    granularity: str = "daily",
    save: bool = True,
) -> pd.DataFrame:
    """
    Classify each observation against the (sensitivity-scaled) prediction interval.

    Returns a DataFrame with:
      timestamp / ds         observation time
      actual                 actual demand
      yhat                   point forecast
      lower_bound/upper_bound  sensitivity-scaled interval
      deviation              actual - yhat (signed)
      deviation_magnitude    |actual - yhat|
      breach_magnitude       distance OUTSIDE the nearest bound (0 when NORMAL)
      classification         NORMAL | DROP_ANOMALY | SPIKE_ANOMALY
    """
    if not (sensitivity > 0):
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
    make_output_dirs()

    merged = pd.merge(
        test_df[[DS_COL, Y_COL]].rename(columns={Y_COL: "actual"}),
        forecast_df[[DS_COL, "yhat", "yhat_lower", "yhat_upper"]],
        on=DS_COL, how="inner",
    )
    if len(merged) != len(test_df):
        raise ValueError(f"Forecast covers {len(merged)} of {len(test_df)} observations.")

    merged["lower_bound"] = merged["yhat"] - sensitivity * (merged["yhat"] - merged["yhat_lower"])
    merged["upper_bound"] = merged["yhat"] + sensitivity * (merged["yhat_upper"] - merged["yhat"])
    merged["classification"] = classify_anomalies(
        merged["actual"], merged["lower_bound"], merged["upper_bound"])
    merged["deviation"] = merged["actual"] - merged["yhat"]
    merged["deviation_magnitude"] = merged["deviation"].abs()
    merged["breach_magnitude"] = (
        (merged["lower_bound"] - merged["actual"]).clip(lower=0)
        + (merged["actual"] - merged["upper_bound"]).clip(lower=0)
    )
    merged["timestamp"] = merged[DS_COL]
    merged = merged[OUTPUT_COLUMNS + [c for c in ("yhat_lower", "yhat_upper") if c in merged.columns]]

    counts = merged["classification"].value_counts()
    n_total = len(merged)
    n_spike = int(counts.get("SPIKE_ANOMALY", 0))
    n_drop = int(counts.get("DROP_ANOMALY", 0))
    n_normal = int(counts.get("NORMAL", 0))
    logger.info("[ANOMALY] sensitivity=%.2f: %d NORMAL / %d DROP / %d SPIKE (of %d)",
                sensitivity, n_normal, n_drop, n_spike, n_total)

    if save:
        _save_anomaly_results(merged, sensitivity, granularity, n_spike, n_drop, n_normal)
        _plot_anomalies(merged, granularity)
    return merged


def _save_anomaly_results(merged: pd.DataFrame, sensitivity: float, granularity: str,
                          n_spike: int, n_drop: int, n_normal: int) -> None:
    csv_path = FORECASTS_DIR / f"anomaly_detections_{granularity}.csv"
    merged.to_csv(csv_path, index=False)

    flagged = merged[merged["classification"] != "NORMAL"].copy()
    records = [
        {
            "timestamp": str(r["timestamp"]),
            "actual_demand": float(r["actual"]),
            "lower_bound": float(r["lower_bound"]),
            "upper_bound": float(r["upper_bound"]),
            "deviation_magnitude": float(r["deviation_magnitude"]),
            "breach_magnitude": float(r["breach_magnitude"]),
            "classification": r["classification"],
        }
        for _, r in flagged.iterrows()
    ]
    n_total = len(merged)
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "granularity": granularity,
        "sensitivity_multiplier": sensitivity,
        "rule": "actual < lower_bound -> DROP_ANOMALY; actual > upper_bound -> SPIKE_ANOMALY; else NORMAL",
        "bounds_definition": "lower = yhat - s*(yhat - yhat_lower); upper = yhat + s*(yhat_upper - yhat)",
        "n_total": n_total,
        "n_normal": n_normal,
        "n_drop_anomaly": n_drop,
        "n_spike_anomaly": n_spike,
        "alert_rate_pct": round(100 * (n_drop + n_spike) / n_total, 2) if n_total else 0.0,
        "empirical_interval_coverage_pct": round(100 * n_normal / n_total, 2) if n_total else 0.0,
        "labelled_ground_truth_available": False,
        "classification_metrics": None,
        "disclaimer": DISCLAIMER,
    }
    with open(METRICS_DIR / f"anomaly_summary_{granularity}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(FORECASTS_DIR / f"anomalies_{granularity}.json", "w", encoding="utf-8") as fh:
        json.dump({"sensitivity_multiplier": sensitivity, "disclaimer": DISCLAIMER,
                   "anomalies": records}, fh, indent=2)
    logger.info("[ANOMALY] Saved %s (+ summary, %d flagged records)", csv_path, len(records))


def _plot_anomalies(merged: pd.DataFrame, granularity: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(merged[DS_COL], merged["actual"], color=PALETTE["actual"], linewidth=1.2,
            label="Actual Demand", zorder=3)
    ax.plot(merged[DS_COL], merged["yhat"], color=PALETTE["forecast"], linewidth=1.6,
            label="Prophet Forecast", zorder=4)
    ax.fill_between(merged[DS_COL], merged["lower_bound"], merged["upper_bound"],
                    color=PALETTE["interval"], alpha=0.25, label="Anomaly bounds")
    drops = merged[merged["classification"] == "DROP_ANOMALY"]
    spikes = merged[merged["classification"] == "SPIKE_ANOMALY"]
    if not drops.empty:
        ax.scatter(drops[DS_COL], drops["actual"], color=PALETTE["drop"], s=28, zorder=6,
                   label=f"Drop (n={len(drops)})")
    if not spikes.empty:
        ax.scatter(spikes[DS_COL], spikes["actual"], color=PALETTE["spike"], s=28, zorder=6,
                   label=f"Spike (n={len(spikes)})")
    rate = 100 * (len(drops) + len(spikes)) / max(len(merged), 1)
    ax.set_title(f"Anomaly signals — {granularity.capitalize()} (alert rate {rate:.1f}%) — "
                 "investigation signals, not ground truth", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.legend(fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"anomaly_detection_{granularity}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
