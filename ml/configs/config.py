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
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = [
    {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 1.0,
        "holidays_prior_scale": 1.0,
    },
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 5.0,
        "holidays_prior_scale": 10.0,
    },
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
    },
    {
        "changepoint_prior_scale": 0.10,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
    },
    {
        "changepoint_prior_scale": 0.15,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
    },
    {
        "changepoint_prior_scale": 0.20,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSS-VALIDATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CV_N_FOLDS = 3             # Number of expanding-window folds
CV_VAL_HORIZON_DAYS = 60   # Validation horizon per fold (60 days forward)
CV_MIN_TRAIN_DAYS = 365    # Minimum training days before first fold
CV_VAL_HORIZON_WEEKS = 12  # Validation horizon per fold for weekly granularity (~3 months)
CV_MIN_TRAIN_WEEKS = 52    # Minimum training weeks before first fold (1 year)

# ─────────────────────────────────────────────────────────────────────────────
# 9. ANOMALY DETECTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Base sensitivity: 1.0 = use raw 95% PI bounds; >1 = wider (fewer anomalies)
ANOMALY_SENSITIVITY = 1.0  # multiplier applied to the PI half-width
# Documented rationale: 95% PI from Prophet is already broad; sensitivity=1.0
# gives a baseline alert rate without artificially suppressing or inflating anomalies.

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
MODEL_VERSION_DAILY = "v1"
MODEL_VERSION_WEEKLY = "v1"
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
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
