"""
ml/configs/config.py
────────────────────
Centralized configuration for the Energy Demand Forecasting ML pipeline.

All paths, column names, model hyperparameters, and operational settings live here.
Do NOT hardcode these values in individual modules — import from this file instead.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────
# Root of the ml/ package (two levels up from this file)
ML_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ML_ROOT.parent

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────
# Primary OPSD time-series file (60-minute resolution, singleindex)
OPSD_RAW_DIR = ML_ROOT / "data" / "opsd-time_series-2020-10-06" / "opsd-time_series-2020-10-06"
OPSD_60MIN_CSV = OPSD_RAW_DIR / "time_series_60min_singleindex.csv"
OPSD_DATAPACKAGE_JSON = OPSD_RAW_DIR / "datapackage.json"

# Processed data output directories
PROCESSED_DIR = ML_ROOT / "data" / "processed"
METADATA_DIR = ML_ROOT / "data" / "metadata"

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET CONFIGURATION
#    Discovered from actual OPSD dataset inspection — do NOT guess these.
# ─────────────────────────────────────────────────────────────────────────────
DATASET_VERSION = "2020-10-06"
DATASET_TITLE = "OPSD Time Series — Load, Wind, Solar, Prices (hourly)"

# Timestamp columns in the raw CSV
TIMESTAMP_COL_UTC = "utc_timestamp"
TIMESTAMP_COL_LOCAL = "cet_cest_timestamp"
TIMESTAMP_COL = TIMESTAMP_COL_UTC          # Primary timestamp column used

# Region / country selection
REGION = "DE"                              # Germany national grid (ENTSO-E Transparency)

# Target energy demand column (actual hourly load, MW)
TARGET_COL_RAW = "DE_load_actual_entsoe_transparency"
TARGET_UNIT_RAW = "MW"                    # Hourly average load in MW

# Prophet input columns (after preprocessing)
DS_COL = "ds"
Y_COL = "y"

# Forecast input frequency
RAW_FREQUENCY = "h"                        # Hourly raw data
DAILY_FREQUENCY = "D"                      # Daily resampled target
WEEKLY_FREQUENCY = "W-MON"                 # Weekly resampled target (week starts Monday)

# Resampling aggregation for MW -> daily/weekly
RESAMPLE_AGG = "mean"                      # Mean hourly load per day/week

# Timezone handling
TIMEZONE = "UTC"                           # Raw data is in UTC; normalized to UTC naive

# Completeness rules for resampled periods. A partial period would silently mix
# a different averaging window into the target, so it is dropped (never imputed).
MIN_HOURS_PER_DAY = 24                     # hourly observations required for a daily row
DAYS_PER_WEEK = 7                          # daily observations required for a weekly row

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70        # First 70% of chronological series → Training
TEST_RATIO = 0.30         # Last 30% of chronological series  → Holdout Test

# ─────────────────────────────────────────────────────────────────────────────
# 5. BASELINE MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Seasonal period for Seasonal-Naive baseline
# Daily granularity → 7-day weekly cycle
# Weekly granularity → 52-week annual cycle
SEASONAL_PERIOD_DAILY = 7       # 7-day weekly seasonality
SEASONAL_PERIOD_WEEKLY = 52     # 52-week annual seasonality

# ─────────────────────────────────────────────────────────────────────────────
# 6. PROPHET HYPERPARAMETER DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
PROPHET_DEFAULT_PARAMS = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "seasonality_mode": "additive",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.95,          # 95% prediction uncertainty interval
    "n_changepoints": 25,
    "changepoint_range": 0.8,        # Changepoints in first 80% of training series
}

# Country holiday calendar
HOLIDAY_COUNTRY = "DE"               # Germany — must match selected REGION

# ─────────────────────────────────────────────────────────────────────────────
# 7. HYPERPARAMETER SEARCH SPACE (for training-only grid search)
#    FROZEN BEFORE ANY TUNING RUN. It must never be edited after looking at
#    hold-out test results (that would be tuning against the test set).
# ─────────────────────────────────────────────────────────────────────────────
import itertools as _itertools

PARAM_SEARCH_SPACE = {
    "changepoint_prior_scale": [0.005, 0.01, 0.05, 0.1, 0.5],
    "seasonality_prior_scale": [1.0, 10.0],
    "holidays_prior_scale": [1.0, 10.0],
    # Changepoint configuration: (n_changepoints, changepoint_range)
    "changepoint_config": [(25, 0.8), (25, 0.9), (10, 0.8), (40, 0.8)],
}


def build_param_grid(space=None):
    """Full cartesian grid (list of param dicts) from a search-space dict."""
    space = space or PARAM_SEARCH_SPACE
    grid = []
    for cps, sps, hps, (n_cp, cp_range) in _itertools.product(
        space["changepoint_prior_scale"],
        space["seasonality_prior_scale"],
        space["holidays_prior_scale"],
        space["changepoint_config"],
    ):
        grid.append({
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
            "holidays_prior_scale": hps,
            "n_changepoints": n_cp,
            "changepoint_range": cp_range,
        })
    return grid


PARAM_GRID = build_param_grid()

# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSS-VALIDATION SETTINGS  (rolling-origin / expanding-window, TRAIN ONLY)
#    The training window always starts at row 0 and grows with each origin
#    (expanding window). Origins are anchored backwards from the END of the
#    training partition, spaced `step` rows apart. Validation windows may
#    overlap when step < horizon (standard rolling-origin evaluation).
# ─────────────────────────────────────────────────────────────────────────────
CV_MIN_TRAIN_DAYS = 730    # >= 2 years so yearly seasonality is identifiable in every fold
CV_VAL_HORIZON_DAYS = 180  # forecast horizon scored per fold (~6 months)
CV_STEP_DAYS = 90          # distance between consecutive forecast origins
CV_MIN_TRAIN_WEEKS = 104
CV_VAL_HORIZON_WEEKS = 26
CV_STEP_WEEKS = 13
CV_N_FOLDS = None          # None = as many folds as fit; int = keep the most recent N

# Primary model-selection metric (validation only). Ties -> lower RMSE.
SELECTION_METRIC = "MAE"

# ─────────────────────────────────────────────────────────────────────────────
# 9. ANOMALY DETECTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Base sensitivity: 1.0 = use raw 95% PI bounds; >1 = wider (fewer anomalies)
ANOMALY_SENSITIVITY = 1.0  # multiplier applied to the PI half-widths
# Rationale: 1.0 == the raw 95% Prophet interval, exactly the rule in the spec
# (actual < lower -> DROP, actual > upper -> SPIKE). The value is a user-facing knob;
# it must NOT be tuned on the hold-out test set. Training-only calibration evidence
# (CV coverage) is written to the tuning artifact for reference.

# ─────────────────────────────────────────────────────────────────────────────
# 10. ARTIFACT & REPORT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = ML_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FORECASTS_DIR = ARTIFACTS_DIR / "forecasts"
ARTIFACT_METADATA_DIR = ARTIFACTS_DIR / "metadata"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_REPORT_DIR = REPORTS_DIR / "eda"
BASELINES_REPORT_DIR = REPORTS_DIR / "baselines"
EXPERIMENTS_REPORT_DIR = REPORTS_DIR / "experiments"
EVALUATION_REPORT_DIR = REPORTS_DIR / "evaluation"
AUDIT_REPORT_DIR = REPORTS_DIR / "audit"

# ─────────────────────────────────────────────────────────────────────────────
# 11. LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────────────────────────────────────────
# 12. REPRODUCIBILITY SEED
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# 13. MODEL VERSIONING
# ─────────────────────────────────────────────────────────────────────────────
MODEL_VERSION_DAILY = "v2"     # v1 artifacts predate leakage-safe CV/preprocessing fixes
MODEL_VERSION_WEEKLY = "v2"
MODEL_ID_DAILY = f"prophet-daily-{MODEL_VERSION_DAILY}"
MODEL_ID_WEEKLY = f"prophet-weekly-{MODEL_VERSION_WEEKLY}"


def make_output_dirs():
    """Create all required output directories if they do not exist."""
    dirs = [
        PROCESSED_DIR,
        METADATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        FORECASTS_DIR,
        ARTIFACT_METADATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        EDA_REPORT_DIR,
        BASELINES_REPORT_DIR,
        EXPERIMENTS_REPORT_DIR,
        EVALUATION_REPORT_DIR,
        AUDIT_REPORT_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
