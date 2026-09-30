"""
ml/ingestion/loader.py
──────────────────────
OPSD Dataset Loader

Provides a reusable, configurable loader for the OPSD time-series dataset.
Performs initial validation (file existence, column presence, type checking)
and computes a SHA-256 checksum for reproducibility tracking.

Responsibilities:
  - Load raw OPSD CSV (60min singleindex)
  - Validate schema at point of ingestion
  - Return a clean raw DataFrame with the essential columns
  - Log all key dataset facts
  - Never fabricate any metadata values
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ml.configs.config import (
    OPSD_60MIN_CSV,
    TIMESTAMP_COL,
    TARGET_COL_RAW,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)

# ─── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


# ─── Public API ──────────────────────────────────────────────────────────────

def load_opsd_timeseries(
    data_path: Optional[Path] = None,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL_RAW,
) -> Tuple[pd.DataFrame, str]:
    """
    Load the OPSD 60-minute time-series CSV and return a validated raw DataFrame
    along with the file SHA-256 checksum.

    Parameters
    ----------
    data_path     : Path to the OPSD CSV file.  Defaults to config value.
    timestamp_col : Name of the UTC timestamp column.
    target_col    : Name of the energy-demand target column.

    Returns
    -------
    (df, checksum) : tuple
        df       — raw DataFrame containing [timestamp_col, target_col]
        checksum — SHA-256 hex digest of the raw CSV for reproducibility

    Raises
    ------
    FileNotFoundError  if the CSV file does not exist.
    ValueError         if required columns are missing or types are invalid.
    """
    path = Path(data_path) if data_path else OPSD_60MIN_CSV

    # ── Gate 1: File must exist ───────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"[INGESTION ERROR] Dataset not found at: {path}\n"
            "Please verify the OPSD data is present in ml/data/."
        )

    logger.info("[INGESTION] Loading OPSD dataset from: %s", path)

    # ── Compute SHA-256 checksum for reproducibility ──────────────────────
    checksum = _sha256(path)
    logger.info("[INGESTION] SHA-256 checksum: %s", checksum)

    # ── Load only required columns for performance (file is ~130 MB) ──────
    try:
        df = pd.read_csv(
            path,
            usecols=[timestamp_col, target_col],
            dtype={target_col: "float64"},
            low_memory=False,
        )
    except ValueError as exc:
        # Attempt a full header read to report which columns exist
        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [c for c in [timestamp_col, target_col] if c not in header]
        raise ValueError(
            f"[INGESTION ERROR] Required column(s) not found: {missing}\n"
            f"Available columns (first 20): {header[:20]}"
        ) from exc

    # ── Gate 2: Dataset must not be empty ────────────────────────────────
    if df.empty:
        raise ValueError("[INGESTION ERROR] Dataset loaded but is empty (0 rows).")

    # ── Gate 3: Timestamp column must parse ──────────────────────────────
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    unparseable = df[timestamp_col].isna().sum()
    if unparseable > 0:
        raise ValueError(
            f"[INGESTION ERROR] {unparseable} timestamp(s) could not be parsed. "
            "Check the timestamp column format."
        )

    # ── Gate 4: Demand values must be numeric ────────────────────────────
    if not pd.api.types.is_float_dtype(df[target_col]):
        raise ValueError(
            f"[INGESTION ERROR] Target column '{target_col}' must be numeric float. "
            f"Got: {df[target_col].dtype}"
        )

    # ── Log ingestion summary ─────────────────────────────────────────────
    n_rows = len(df)
    n_nulls = df[target_col].isna().sum()
    ts_min = df[timestamp_col].min()
    ts_max = df[timestamp_col].max()

    logger.info("[INGESTION] Rows loaded      : %d", n_rows)
    logger.info("[INGESTION] Date range       : %s → %s", ts_min, ts_max)
    logger.info("[INGESTION] Target nulls     : %d / %d (%.2f%%)",
                n_nulls, n_rows, 100 * n_nulls / n_rows)

    return df, checksum


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
