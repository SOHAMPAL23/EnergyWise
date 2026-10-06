"""
ml/tests/test_data.py
──────────────────────
Data Validation Tests

Tests:
  - Dataset loading
  - Schema validation
  - Timestamp validation
  - Duplicate detection
  - Missingness checks
  - Chronological ordering
  - Gap detection
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path

from ml.validation.schema import validate_schema
from ml.validation.timestamps import validate_timestamps
from ml.validation.missingness import validate_missingness
from ml.validation.duplicates import validate_duplicates
from ml.validation.gaps import validate_gaps
from ml.configs.config import TIMESTAMP_COL, TARGET_COL_RAW, DS_COL, Y_COL


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_hourly_df():
    """A valid hourly DataFrame with no issues."""
    dates = pd.date_range("2015-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame({
        TIMESTAMP_COL: dates,
        TARGET_COL_RAW: np.random.uniform(40000, 70000, 48),
    })


@pytest.fixture
def valid_daily_df():
    """A valid daily Prophet DataFrame."""
    dates = pd.date_range("2015-01-01", periods=100, freq="D")
    return pd.DataFrame({
        DS_COL: dates,
        Y_COL: np.random.uniform(40000, 70000, 100),
    })


# ─── Schema tests ──────────────────────────────────────────────────────────────

def test_schema_valid_df(valid_hourly_df):
    result = validate_schema(valid_hourly_df, strict=True)
    assert result["status"] == "PASS"


def test_schema_missing_timestamp_col(valid_hourly_df):
    df = valid_hourly_df.drop(columns=[TIMESTAMP_COL])
    with pytest.raises(ValueError, match="not found"):
        validate_schema(df, strict=True)


def test_schema_missing_target_col(valid_hourly_df):
    df = valid_hourly_df.drop(columns=[TARGET_COL_RAW])
    with pytest.raises(ValueError, match="not found"):
        validate_schema(df, strict=True)


def test_schema_non_numeric_target(valid_hourly_df):
    df = valid_hourly_df.copy()
    df[TARGET_COL_RAW] = "bad_value"
    with pytest.raises(ValueError, match="not numeric"):
        validate_schema(df, strict=True)


def test_schema_empty_df():
    df = pd.DataFrame({TIMESTAMP_COL: [], TARGET_COL_RAW: []})
    with pytest.raises(ValueError, match="empty"):
        validate_schema(df, strict=True)


# ─── Timestamp tests ───────────────────────────────────────────────────────────

def test_timestamps_valid_monotonic(valid_hourly_df):
    result = validate_timestamps(valid_hourly_df, strict=True)
    assert result["status"] == "PASS"


def test_timestamps_not_monotonic(valid_hourly_df):
    df = valid_hourly_df.copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    result = validate_timestamps(df, strict=False)
    # After random shuffle it may not be monotonic
    assert any(f["check"] == "monotonic_increasing" for f in result["findings"])


def test_timestamps_parseable():
    df = pd.DataFrame({
        TIMESTAMP_COL: ["2015-01-01T00:00:00Z", "2015-01-01T01:00:00Z"],
        TARGET_COL_RAW: [50000.0, 51000.0],
    })
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    result = validate_timestamps(df, strict=True)
    assert result["status"] == "PASS"


# ─── Missingness tests ─────────────────────────────────────────────────────────

def test_missingness_no_nulls(valid_hourly_df):
    result = validate_missingness(valid_hourly_df, target_col=TARGET_COL_RAW)
    assert result["n_null"] == 0


def test_missingness_with_nulls():
    df = pd.DataFrame({
        TIMESTAMP_COL: pd.date_range("2015-01-01", periods=10, freq="h", tz="UTC"),
        TARGET_COL_RAW: [50000.0, np.nan, 52000.0, 53000.0, np.nan,
                         55000.0, 56000.0, 57000.0, 58000.0, 59000.0],
    })
    result = validate_missingness(df, target_col=TARGET_COL_RAW)
    assert result["n_null"] == 2


# ─── Duplicate tests ───────────────────────────────────────────────────────────

def test_duplicates_none(valid_hourly_df):
    result = validate_duplicates(valid_hourly_df, strict=False)
    assert result["n_duplicate_rows"] == 0
    assert result["n_duplicate_timestamps"] == 0


def test_duplicate_rows_detected():
    df = pd.DataFrame({
        TIMESTAMP_COL: pd.to_datetime(["2015-01-01T00:00:00Z", "2015-01-01T00:00:00Z"], utc=True),
        TARGET_COL_RAW: [50000.0, 50000.0],
    })
    result = validate_duplicates(df, strict=False)
    assert result["n_duplicate_rows"] >= 1


def test_duplicate_timestamps_detected():
    df = pd.DataFrame({
        TIMESTAMP_COL: pd.to_datetime(["2015-01-01T00:00:00Z", "2015-01-01T00:00:00Z"], utc=True),
        TARGET_COL_RAW: [50000.0, 51000.0],
    })
    result = validate_duplicates(df, strict=False)
    assert result["n_duplicate_timestamps"] >= 1


# ─── Gap tests ─────────────────────────────────────────────────────────────────

def test_gaps_no_gaps(valid_hourly_df):
    result = validate_gaps(valid_hourly_df)
    assert result["n_gaps"] == 0


def test_gaps_with_gap():
    # Insert a 24-hour gap
    ts_before = pd.date_range("2015-01-01", periods=5, freq="h", tz="UTC")
    ts_after = pd.date_range("2015-01-02 10:00:00", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({
        TIMESTAMP_COL: ts_before.append(ts_after),
        TARGET_COL_RAW: np.random.uniform(40000, 70000, 10),
    })
    result = validate_gaps(df)
    assert result["n_gaps"] >= 1
    assert result["max_gap_hours"] > 1.0


# ─── Ordering test ─────────────────────────────────────────────────────────────

def test_chronological_ordering(valid_daily_df):
    """Sorted daily df must be chronologically ordered."""
    assert valid_daily_df[DS_COL].is_monotonic_increasing
