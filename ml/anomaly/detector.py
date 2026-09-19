
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
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
from ml.evaluation.metrics import (
    classification_accuracy,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix_anomalies,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

PALETTE = {
    "actual": "#0f172a",
    "forecast": "#0d9488",
    "interval": "#14b8a6",
    "spike": "#ef4444",
    "drop": "#3b82f6",
    "normal": "#94a3b8",
    "grid": "#e2e8f0",
}


def detect_anomalies(
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    sensitivity: float = ANOMALY_SENSITIVITY,
    granularity: str = "daily",
    save: bool = True,
) -> pd.DataFrame:
    """
    Classify observations as anomalies using forecast uncertainty intervals.

    Parameters
    ----------
    test_df      : Test DataFrame with (ds, y) columns.
    forecast_df  : Prophet forecast with (ds, yhat, yhat_lower, yhat_upper).
    sensitivity  : Multiplier for PI half-width. 1.0 = use raw bounds.
                   > 1.0 = wider bounds (fewer anomalies).
                   < 1.0 = narrower bounds (more anomalies).
    granularity  : "daily" or "weekly" — used in artifact naming.
    save         : If True, save anomaly results to disk.

    Returns
    -------
    DataFrame with columns:
      ds, actual, yhat, lower_bound, upper_bound, deviation, classification

    Anomaly classification:
      NORMAL        — actual is within [lower_bound, upper_bound]
      DROP_ANOMALY  — actual < lower_bound
      SPIKE_ANOMALY — actual > upper_bound
    """
    make_output_dirs()

    logger.info("[ANOMALY] Classifying anomalies with sensitivity=%.2f ...", sensitivity)

    # ── Merge test actuals with forecast ──────────────────────────────────
    merged = pd.merge(
        test_df[[DS_COL, Y_COL]].rename(columns={Y_COL: "actual"}),
        forecast_df[[DS_COL, "yhat", "yhat_lower", "yhat_upper"]],
        on=DS_COL,
        how="inner",
    )

    # ── Apply sensitivity multiplier to PI half-width ─────────────────────
    if sensitivity != 1.0:
        half_width = (merged["yhat_upper"] - merged["yhat_lower"]) / 2.0
        adjusted_half_width = half_width * sensitivity
        merged["lower_bound"] = merged["yhat"] - adjusted_half_width
        merged["upper_bound"] = merged["yhat"] + adjusted_half_width
        logger.info(
            "[ANOMALY] Sensitivity=%.2f applied. Bounds widened by factor %.2f.",
            sensitivity, sensitivity,
        )
    else:
        merged["lower_bound"] = merged["yhat_lower"]
        merged["upper_bound"] = merged["yhat_upper"]

    # ── Classify each observation ─────────────────────────────────────────
    def classify(row):
        if row["actual"] < row["lower_bound"]:
            return "DROP_ANOMALY"
        elif row["actual"] > row["upper_bound"]:
            return "SPIKE_ANOMALY"
        else:
            return "NORMAL"

    merged["classification"] = merged.apply(classify, axis=1)
    merged["deviation"] = merged["actual"] - merged["yhat"]
    merged["deviation_magnitude"] = merged["deviation"].abs()

    # ── Summary statistics ────────────────────────────────────────────────
    n_total = len(merged)
    n_spike = int((merged["classification"] == "SPIKE_ANOMALY").sum())
    n_drop = int((merged["classification"] == "DROP_ANOMALY").sum())
    n_normal = int((merged["classification"] == "NORMAL").sum())
    alert_rate = round(100 * (n_spike + n_drop) / n_total, 2) if n_total > 0 else 0.0

    logger.info("[ANOMALY] Results:")
    logger.info("[ANOMALY]   Total observations : %d", n_total)
    logger.info("[ANOMALY]   NORMAL             : %d (%.1f%%)", n_normal, 100 * n_normal / n_total)
    logger.info("[ANOMALY]   SPIKE_ANOMALY      : %d (%.1f%%)", n_spike, 100 * n_spike / n_total)
    logger.info("[ANOMALY]   DROP_ANOMALY       : %d (%.1f%%)", n_drop, 100 * n_drop / n_total)
    # ── Classification performance vs statistical reference benchmark ───────
    dev = merged["deviation"].values
    dev_std = float(np.std(dev)) if len(dev) > 1 else 1.0
    # True statistical anomaly threshold: 2.5 std devs from forecast
    ref_anomaly = np.abs(dev) >= (2.5 * dev_std)
    y_true_binary = np.where(ref_anomaly, "ANOMALY", "NORMAL")
    y_pred_binary = np.where(merged["classification"] != "NORMAL", "ANOMALY", "NORMAL")

    accuracy_val = classification_accuracy(y_true_binary, y_pred_binary)
    precision_val = precision_score(y_true_binary, y_pred_binary, pos_label="ANOMALY")
    recall_val = recall_score(y_true_binary, y_pred_binary, pos_label="ANOMALY")
    f1_val = f1_score(y_true_binary, y_pred_binary, pos_label="ANOMALY")
    cm_val = confusion_matrix_anomalies(y_true_binary, y_pred_binary, pos_label="ANOMALY")

    classification_metrics = {
        "accuracy_pct": round(accuracy_val, 2),
        "precision": round(precision_val, 4),
        "recall": round(recall_val, 4),
        "f1_score": round(f1_val, 4),
        "confusion_matrix": cm_val,
        "n_reference_anomalies": int(np.sum(ref_anomaly)),
    }

    logger.info("[ANOMALY] Classification Performance (vs. 2.5-sigma benchmark):")
    logger.info("[ANOMALY]   Accuracy           : %.2f%%", accuracy_val)
    logger.info("[ANOMALY]   Precision          : %.4f", precision_val)
    logger.info("[ANOMALY]   Recall             : %.4f", recall_val)
    logger.info("[ANOMALY]   F1-Score           : %.4f", f1_val)
    logger.info("[ANOMALY]   Confusion Matrix   : TP=%d, FP=%d, TN=%d, FN=%d",
                cm_val["tp"], cm_val["fp"], cm_val["tn"], cm_val["fn"])
    logger.info("[ANOMALY] NOTE: Anomaly flags are investigation signals only.")
    logger.info("[ANOMALY]       False positives are expected at 95%% PI width.")

    if save:
        _save_anomaly_results(
            merged, alert_rate, sensitivity, granularity,
            n_spike, n_drop, n_normal, classification_metrics
        )
        _plot_anomalies(merged, granularity)

    return merged


def _save_anomaly_results(merged, alert_rate, sensitivity, granularity,
                          n_spike, n_drop, n_normal, classification_metrics=None):
    """Save anomaly classifications and summary metadata."""
    # CSV
    csv_path = FORECASTS_DIR / f"anomaly_detections_{granularity}.csv"
    merged.to_csv(csv_path, index=False)
    logger.info("[ANOMALY] Classifications saved: %s", csv_path)

    # Summary JSON
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "granularity": granularity,
        "sensitivity_multiplier": sensitivity,
        "anomaly_threshold_rationale": (
            "95% prediction interval from Prophet. "
            f"Sensitivity={sensitivity} (1.0 = raw bounds; >1 = wider). "
            "Chosen as a conservative baseline for investigation signalling."
        ),
        "n_total": len(merged),
        "n_normal": n_normal,
        "n_spike_anomaly": n_spike,
        "n_drop_anomaly": n_drop,
        "alert_rate_pct": alert_rate,
        "classification_metrics": classification_metrics or {},
        "disclaimer": (
            "Anomaly classifications are investigation signals only. "
            "They are NOT confirmation of abnormal grid behaviour. "
            "All flagged events should be reviewed by a domain expert."
        ),
    }
    summary_path = METRICS_DIR / f"anomaly_summary_{granularity}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("[ANOMALY] Summary saved: %s", summary_path)


def _plot_anomalies(merged: pd.DataFrame, granularity: str):
    """Generate anomaly detection visualization."""
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(merged[DS_COL], merged["actual"], color=PALETTE["actual"],
            linewidth=1.2, label="Actual Demand", zorder=3)
    ax.plot(merged[DS_COL], merged["yhat"], color=PALETTE["forecast"],
            linewidth=1.8, label="Prophet Forecast", zorder=4)
    ax.fill_between(merged[DS_COL], merged["lower_bound"], merged["upper_bound"],
                    color=PALETTE["interval"], alpha=0.25, label="Prediction Interval")

    # Drop anomalies
    drops = merged[merged["classification"] == "DROP_ANOMALY"]
    if not drops.empty:
        ax.scatter(drops[DS_COL], drops["actual"], color=PALETTE["drop"],
                   s=60, zorder=6, label=f"Drop Anomalies (n={len(drops)})")

    # Spike anomalies
    spikes = merged[merged["classification"] == "SPIKE_ANOMALY"]
    if not spikes.empty:
        ax.scatter(spikes[DS_COL], spikes["actual"], color=PALETTE["spike"],
                   s=60, zorder=6, label=f"Spike Anomalies (n={len(spikes)})")

    n_total = len(merged)
    n_anomalies = len(drops) + len(spikes)
    alert_rate = 100 * n_anomalies / n_total if n_total > 0 else 0.0

    ax.set_title(
        f"Anomaly Detection — {granularity.capitalize()} Energy Demand\n"
        f"Alert rate: {alert_rate:.1f}% | Intervals are 95% PI | "
        "Classifications are investigation signals only",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Load (MW)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()

    fpath = FIGURES_DIR / f"anomaly_detection_{granularity}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[ANOMALY] Plot saved: %s", fpath)
