"""
ml/prophet/components.py
─────────────────────────
Prophet Component Extractor

Extracts and saves all additive model components from a fitted Prophet forecast:
  - trend
  - weekly seasonality
  - yearly seasonality
  - holiday effects
  - changepoints

These are saved as machine-readable JSON/CSV artifacts suitable for consumption
by a FastAPI backend and React frontend.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prophet import Prophet

from ml.configs.config import (
    DS_COL,
    FIGURES_DIR,
    FORECASTS_DIR,
    ARTIFACT_METADATA_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def extract_components(
    model: Prophet,
    forecast: pd.DataFrame,
    granularity: str = "daily",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Extract interpretable components from a fitted Prophet forecast.

    Parameters
    ----------
    model      : Fitted Prophet model.
    forecast   : Forecast DataFrame from model.predict().
    granularity: "daily" or "weekly" — used in file naming.
    save       : If True, save components to disk.

    Returns
    -------
    dict with keys: trend, weekly, yearly, holidays, changepoints, component_df
    """
    make_output_dirs()
    components: Dict[str, Any] = {}

    # ── Columns available in forecast ─────────────────────────────────────
    avail = set(forecast.columns)

    # ── Trend ─────────────────────────────────────────────────────────────
    if "trend" in avail:
        trend_df = forecast[[DS_COL, "trend"]].copy()
        components["trend"] = trend_df.to_dict(orient="records")
        logger.info("[COMPONENTS] Trend extracted (%d points).", len(trend_df))
    else:
        components["trend"] = []
        logger.warning("[COMPONENTS] 'trend' column not found in forecast.")

    # ── Weekly seasonality ────────────────────────────────────────────────
    if "weekly" in avail:
        weekly_df = forecast[[DS_COL, "weekly"]].copy()
        components["weekly_seasonality"] = weekly_df.to_dict(orient="records")
        logger.info("[COMPONENTS] Weekly seasonality extracted.")
    else:
        components["weekly_seasonality"] = []
        logger.info("[COMPONENTS] Weekly seasonality not in forecast (disabled for weekly model).")

    # ── Yearly seasonality ────────────────────────────────────────────────
    if "yearly" in avail:
        yearly_df = forecast[[DS_COL, "yearly"]].copy()
        components["yearly_seasonality"] = yearly_df.to_dict(orient="records")
        logger.info("[COMPONENTS] Yearly seasonality extracted.")
    else:
        components["yearly_seasonality"] = []
        logger.info("[COMPONENTS] Yearly seasonality not in forecast.")

    # ── Holiday effects ───────────────────────────────────────────────────
    if "holidays" in avail:
        holiday_df = forecast[[DS_COL, "holidays"]].copy()
        components["holidays"] = holiday_df.to_dict(orient="records")
        logger.info("[COMPONENTS] Holiday effects extracted.")
    else:
        components["holidays"] = []
        logger.info("[COMPONENTS] No holiday effects in forecast.")

    # ── Changepoints ──────────────────────────────────────────────────────
    try:
        changepoints = [str(cp.date()) for cp in model.changepoints]
        deltas = model.params.get("delta", np.array([[]]))[0]
        cp_records = [
            {
                "date": cp,
                "delta": round(float(deltas[i]), 6) if i < len(deltas) else 0.0,
                "magnitude": round(abs(float(deltas[i])), 6) if i < len(deltas) else 0.0,
            }
            for i, cp in enumerate(changepoints)
        ]
        # Sort by magnitude (largest impact first)
        cp_records.sort(key=lambda x: x["magnitude"], reverse=True)
        components["changepoints"] = cp_records
        logger.info("[COMPONENTS] %d changepoint(s) extracted.", len(cp_records))
    except Exception as exc:
        components["changepoints"] = []
        logger.warning("[COMPONENTS] Could not extract changepoints: %s", exc)

    # ── Store full forecast for downstream use ────────────────────────────
    components["component_df"] = forecast

    if save:
        _save_components(components, granularity)
        _save_component_plots(model, forecast, granularity)

    return components


def _save_components(components: Dict, granularity: str):
    """Save component data as JSON and CSV artifacts."""
    # Save serializable components as JSON (exclude DataFrame)
    serializable = {k: v for k, v in components.items() if k != "component_df"}
    json_path = ARTIFACT_METADATA_DIR / f"prophet_{granularity}_components.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, default=str)
    logger.info("[COMPONENTS] Saved: %s", json_path)

    # Save full forecast CSV
    if "component_df" in components:
        csv_path = FORECASTS_DIR / f"prophet_{granularity}_forecast_components.csv"
        components["component_df"].to_csv(csv_path, index=False)
        logger.info("[COMPONENTS] Forecast CSV saved: %s", csv_path)


def _save_component_plots(model: Prophet, forecast: pd.DataFrame, granularity: str):
    """Save Prophet component decomposition plot."""
    try:
        fig = model.plot_components(forecast)
        fig.suptitle(f"Prophet Model Components — {granularity.capitalize()}",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fpath = FIGURES_DIR / f"prophet_{granularity}_components.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[COMPONENTS] Component plot saved: %s", fpath)
    except Exception as exc:
        logger.warning("[COMPONENTS] Could not save component plot: %s", exc)
