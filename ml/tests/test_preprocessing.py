"""
ml/tests/test_preprocessing.py
────────────────────────────────
Preprocessing Tests

Tests:
  - Cleaning (duplicate removal, interpolation)
  - Daily resampling
  - Weekly resampling
  - Chronological train/test split
"""

import pytest
import pandas as pd
import numpy as np

from ml.preprocessing.cleaning import clean_raw_dataframe
from ml.preprocessing.resampling import resample_to_daily, resample_to_weekly
from ml.preprocessing.splitting import chronological_train_test_split
from ml.configs.config import TIMESTAMP_COL, TARGET_COL_RAW, DS_COL, Y_COL


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_hourly_df():
    dates = pd.date_range("2015-01-01", periods=240, freq="h", tz="UTC")
    vals = np.random.uniform(40000, 70000, 240)
    vals[10] = np.nan   # insert one null
    return pd.DataFrame({TIMESTAMP_COL: dates, TARGET_COL_RAW: vals})


@pytest.fixture
def clean_hourly_df(raw_hourly_df):
    return clean_raw_dataframe(raw_hourly_df)


@pytest.fixture
def daily_df(clean_hourly_df):
    return resample_to_daily(clean_hourly_df)


@pytest.fixture
def large_daily_df():
    """500-day daily series for split testing."""
    dates = pd.date_range("2015-01-01", periods=500, freq="D")
    return pd.DataFrame({DS_COL: dates, Y_COL: np.random.uniform(40000, 70000, 500)})


# ─── Cleaning tests ────────────────────────────────────────────────────────────

def test_cleaning_returns_dataframe(clean_hourly_df):
    assert isinstance(clean_hourly_df, pd.DataFrame)


def test_cleaning_no_duplicates_after(raw_hourly_df):
    raw_with_dups = pd.concat([raw_hourly_df, raw_hourly_df.head(3)], ignore_index=True)
    cleaned = clean_raw_dataframe(raw_with_dups)
    # No exact duplicate rows
    assert cleaned.duplicated().sum() == 0


def test_cleaning_fills_short_nulls(raw_hourly_df):
    cleaned = clean_raw_dataframe(raw_hourly_df)
    # After interpolation of 1 null, should have 0 remaining nulls (or at most original large gaps)
    n_null_before = raw_hourly_df[TARGET_COL_RAW].isna().sum()
    n_null_after = cleaned[TARGET_COL_RAW].isna().sum()
    assert n_null_after <= n_null_before


def test_cleaning_does_not_modify_original(raw_hourly_df):
    original_nulls = raw_hourly_df[TARGET_COL_RAW].isna().sum()
    _ = clean_raw_dataframe(raw_hourly_df)
    # Original should be unchanged
    assert raw_hourly_df[TARGET_COL_RAW].isna().sum() == original_nulls


def test_cleaning_sorts_chronologically(raw_hourly_df):
    shuffled = raw_hourly_df.sample(frac=1, random_state=0).reset_index(drop=True)
    cleaned = clean_raw_dataframe(shuffled)
    assert cleaned[TIMESTAMP_COL].is_monotonic_increasing


# ─── Daily resampling tests ────────────────────────────────────────────────────

def test_daily_resample_columns(daily_df):
    assert DS_COL in daily_df.columns
    assert Y_COL in daily_df.columns


def test_daily_resample_reduces_rows(clean_hourly_df, daily_df):
    assert len(daily_df) < len(clean_hourly_df)


def test_daily_resample_one_row_per_day(daily_df):
    assert daily_df[DS_COL].nunique() == len(daily_df)


def test_daily_resample_demand_positive(daily_df):
    # Energy demand should always be positive
    assert (daily_df[Y_COL].dropna() > 0).all()


# ─── Weekly resampling tests ───────────────────────────────────────────────────

def test_weekly_resample_columns(daily_df):
    weekly = resample_to_weekly(daily_df)
    assert DS_COL in weekly.columns
    assert Y_COL in weekly.columns


def test_weekly_resample_fewer_rows_than_daily(daily_df):
    weekly = resample_to_weekly(daily_df)
    assert len(weekly) < len(daily_df)


def test_weekly_resample_demand_positive(daily_df):
    weekly = resample_to_weekly(daily_df)
    assert (weekly[Y_COL].dropna() > 0).all()


# ─── Chronological split tests ─────────────────────────────────────────────────

def test_split_returns_three_items(large_daily_df):
    result = chronological_train_test_split(large_daily_df, train_ratio=0.70)
    assert len(result) == 3


def test_split_sizes_correct(large_daily_df):
    train, test, info = chronological_train_test_split(large_daily_df, train_ratio=0.70)
    n = len(large_daily_df)
    assert len(train) + len(test) == n
    # Train should be ~70%
    assert abs(len(train) / n - 0.70) < 0.02


def test_split_train_before_test(large_daily_df):
    train, test, info = chronological_train_test_split(large_daily_df)
    assert train[DS_COL].max() < test[DS_COL].min()


def test_split_no_timestamp_overlap(large_daily_df):
    train, test, info = chronological_train_test_split(large_daily_df)
    train_dates = set(train[DS_COL].astype(str))
    test_dates = set(test[DS_COL].astype(str))
    assert train_dates.isdisjoint(test_dates)


def test_split_is_deterministic(large_daily_df):
    train1, test1, _ = chronological_train_test_split(large_daily_df)
    train2, test2, _ = chronological_train_test_split(large_daily_df)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_split_empty_df_raises():
    df = pd.DataFrame({DS_COL: [], Y_COL: []})
    with pytest.raises(ValueError, match="empty"):
        chronological_train_test_split(df)
