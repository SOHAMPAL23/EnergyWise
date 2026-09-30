"""
ml/ingestion/profiler.py
────────────────────────
OPSD Dataset Profiler

Generates a comprehensive machine-readable dataset profile from the raw DataFrame
returned by loader.load_opsd_timeseries().

Profile includes:
  - File metadata (path, checksum, size)
  - Row and column counts
  - Column names and data types
  - Date range and detected frequency
  - Timezone info
  - Missing-value statistics
  - Duplicate row and timestamp statistics
  - Time-gap analysis
  - Descriptive statistics of the target column

The profile is saved as JSON under reports/dataset_profile.json and
ml/data/metadata/dataset_profile.json.

IMPORTANT: All values are derived from the actual dataset — nothing is fabricated.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np

from ml.configs.config import (
    OPSD_60MIN_CSV,
    OPSD_DATAPACKAGE_JSON,
    TIMESTAMP_COL,
    TARGET_COL_RAW,
    METADATA_DIR,
    REPORTS_DIR,
    DATASET_VERSION,
    REGION,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


# ─── Public API ──────────────────────────────────────────────────────────────

def profile_dataset(
    df: pd.DataFrame,
    checksum: str,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL_RAW,
    data_path: Optional[Path] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Generate a comprehensive dataset profile from the raw OPSD DataFrame.

    Parameters
    ----------
    df            : Raw OPSD DataFrame (must contain timestamp_col and target_col).
    checksum      : SHA-256 hex digest computed during loading.
    timestamp_col : Name of the UTC timestamp column.
    target_col    : Name of the energy-demand column.
    data_path     : Path to source CSV (for metadata).
    save          : If True, write profile to JSON files.

    Returns
    -------
    dict — Full dataset profile as a serialisable dictionary.
    """
    make_output_dirs()
    path = Path(data_path) if data_path else OPSD_60MIN_CSV
    logger.info("[PROFILER] Generating dataset profile ...")

    # ── Dataset-level metadata ────────────────────────────────────────────
    datapackage_meta = _read_datapackage()

    ts = df[timestamp_col]
    demand = df[target_col]

    # ── Date range ────────────────────────────────────────────────────────
    ts_min = ts.min()
    ts_max = ts.max()

    # ── Detected frequency ────────────────────────────────────────────────
    freq_detected = _detect_frequency(ts)

    # ── Missing values ────────────────────────────────────────────────────
    n_rows = len(df)
    n_null_target = int(demand.isna().sum())
    null_pct = round(100 * n_null_target / n_rows, 4) if n_rows > 0 else 0.0

    # ── Duplicate analysis ────────────────────────────────────────────────
    n_dup_rows = int(df.duplicated().sum())
    n_dup_ts = int(ts.duplicated().sum())

    # ── Time gaps ─────────────────────────────────────────────────────────
    gap_info = _detect_gaps(ts, freq_detected)

    # ── Descriptive statistics of target ─────────────────────────────────
    desc = demand.describe()
    target_stats = {
        "count": int(desc["count"]),
        "mean_mw": round(float(desc["mean"]), 2),
        "std_mw": round(float(desc["std"]), 2),
        "min_mw": round(float(desc["min"]), 2),
        "q25_mw": round(float(desc["25%"]), 2),
        "median_mw": round(float(desc["50%"]), 2),
        "q75_mw": round(float(desc["75%"]), 2),
        "max_mw": round(float(desc["max"]), 2),
    }

    # ── Timezone ──────────────────────────────────────────────────────────
    tz_info = str(ts.dt.tz) if hasattr(ts, "dt") and ts.dt.tz is not None else "UTC (inferred)"

    # ── Build profile dict ────────────────────────────────────────────────
    profile: Dict[str, Any] = {
        "profile_generated_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "file": str(path),
            "sha256": checksum,
            "file_size_bytes": path.stat().st_size if path.exists() else "unknown",
            "dataset_version": datapackage_meta.get("version", DATASET_VERSION),
            "dataset_title": datapackage_meta.get("title", "OPSD Time Series"),
            "dataset_description": datapackage_meta.get("description", "NOT AVAILABLE"),
        },
        "dimensions": {
            "total_rows": n_rows,
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "column_dtypes": {c: str(df[c].dtype) for c in df.columns},
        },
        "region": REGION,
        "timestamp_column": timestamp_col,
        "target_column": target_col,
        "target_unit": "MW (average hourly load)",
        "date_range": {
            "start_utc": str(ts_min),
            "end_utc": str(ts_max),
            "total_days": (ts_max - ts_min).days if pd.notna(ts_min) and pd.notna(ts_max) else "unknown",
        },
        "timezone": tz_info,
        "detected_frequency": freq_detected,
        "missing_values": {
            "target_null_count": n_null_target,
            "target_null_pct": null_pct,
            "rows_with_any_null": int(df.isnull().any(axis=1).sum()),
        },
        "duplicates": {
            "duplicate_rows": n_dup_rows,
            "duplicate_timestamps": n_dup_ts,
        },
        "time_gaps": gap_info,
        "target_statistics": target_stats,
    }

    logger.info("[PROFILER] Rows: %d | Nulls: %d | DupRows: %d | DupTS: %d | Gaps: %d",
                n_rows, n_null_target, n_dup_rows, n_dup_ts, gap_info["n_gaps"])

    if save:
        _save_profile(profile)

    return profile


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_datapackage() -> Dict[str, str]:
    """Read datapackage.json for version and title if available."""
    try:
        with open(OPSD_DATAPACKAGE_JSON, encoding="utf-8") as fh:
            pkg = json.load(fh)
        return {
            "version": pkg.get("version", "NOT AVAILABLE"),
            "title": pkg.get("title", "NOT AVAILABLE"),
            "description": pkg.get("description", "NOT AVAILABLE"),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("[PROFILER] datapackage.json not found or unreadable.")
        return {}


def _detect_frequency(ts: pd.Series) -> str:
    """Infer the dominant frequency from timestamp differences."""
    try:
        diffs = ts.sort_values().diff().dropna()
        if diffs.empty:
            return "UNKNOWN"
        mode_diff = diffs.mode()[0]
        total_seconds = int(mode_diff.total_seconds())
        if total_seconds == 3600:
            return "Hourly (1h)"
        elif total_seconds == 1800:
            return "30-minute (30min)"
        elif total_seconds == 900:
            return "15-minute (15min)"
        elif total_seconds == 86400:
            return "Daily (1D)"
        else:
            return f"Unknown ({total_seconds}s)"
    except Exception as exc:
        logger.warning("[PROFILER] Could not detect frequency: %s", exc)
        return "NOT AVAILABLE"


def _detect_gaps(ts: pd.Series, freq_str: str) -> Dict[str, Any]:
    """
    Identify time gaps larger than the detected step size.
    A 'gap' is defined as any interval more than 1.5x the expected step.
    """
    try:
        sorted_ts = ts.sort_values().reset_index(drop=True)
        diffs = sorted_ts.diff().dropna()
        expected_step = diffs.mode()[0]
        threshold = expected_step * 1.5
        gaps = diffs[diffs > threshold]
        gap_records = []
        for idx in gaps.index:
            gap_records.append({
                "gap_start": str(sorted_ts.iloc[idx - 1]),
                "gap_end": str(sorted_ts.iloc[idx]),
                "gap_duration_hours": round(gaps[idx].total_seconds() / 3600, 2),
            })
        return {
            "n_gaps": len(gap_records),
            "expected_step": str(expected_step),
            "threshold": str(threshold),
            "gaps": gap_records[:20],  # cap at 20 for readability
        }
    except Exception as exc:
        logger.warning("[PROFILER] Gap detection failed: %s", exc)
        return {"n_gaps": "NOT AVAILABLE", "gaps": []}


def _save_profile(profile: Dict[str, Any]) -> None:
    """Save profile JSON to both the metadata dir and reports dir."""
    targets = [
        METADATA_DIR / "dataset_profile.json",
        REPORTS_DIR / "dataset_profile.json",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=str)
        logger.info("[PROFILER] Profile saved → %s", target)
