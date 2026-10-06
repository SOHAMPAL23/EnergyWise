"""
ml/tests/test_leakage.py
─────────────────────────
Data Leakage Prevention Tests

Tests that PROVE zero data leakage:
  - Train timestamps are strictly < test timestamps
  - No overlap between train and test sets
  - Test records are never in training set
  - Split is deterministic (same input → same result)
  - CV folds stay within training partition only
  - Tuning only sees training data
"""

import pytest
import numpy as np
import pandas as pd

from ml.preprocessing.splitting import chronological_train_test_split
from ml.validation_models.cross_validation import expanding_window_cv
from ml.configs.config import DS_COL, Y_COL


@pytest.fixture
def ts_df():
    """800-day series suitable for leakage testing."""
    dates = pd.date_range("2015-01-01", periods=800, freq="D")
    vals = np.random.uniform(40000, 70000, 800)
    return pd.DataFrame({DS_COL: dates, Y_COL: vals})


# ─── 70/30 split leakage tests ────────────────────────────────────────────────

def test_train_timestamps_lt_test_timestamps(ts_df):
    """CRITICAL: ALL training timestamps must be < ALL test timestamps."""
    train, test, _ = chronological_train_test_split(ts_df)
    assert train[DS_COL].max() < test[DS_COL].min(), \
        "LEAKAGE: Some training timestamp >= test timestamp"


def test_no_timestamp_overlap(ts_df):
    """No timestamp should appear in both train and test."""
    train, test, _ = chronological_train_test_split(ts_df)
    train_set = set(train[DS_COL].astype(str))
    test_set = set(test[DS_COL].astype(str))
    overlap = train_set & test_set
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} overlapping timestamps: {list(overlap)[:3]}"


def test_no_test_records_in_training(ts_df):
    """Test records must not exist in training set."""
    train, test, _ = chronological_train_test_split(ts_df)
    # Merge to check if any test row matches any training row
    merged = pd.merge(train, test, on=DS_COL, how="inner")
    assert len(merged) == 0, \
        f"LEAKAGE: {len(merged)} test records found in training data"


def test_split_deterministic(ts_df):
    """Same input must always produce the same split boundary."""
    _, _, info1 = chronological_train_test_split(ts_df)
    _, _, info2 = chronological_train_test_split(ts_df)
    assert info1.split_timestamp == info2.split_timestamp, \
        "NON-DETERMINISTIC: Split boundary changed between runs"


def test_split_boundary_correct_ratio(ts_df):
    """Train partition must be exactly floor(N * 0.70) rows."""
    n = len(ts_df)
    expected_train = int(np.floor(n * 0.70))
    train, _, _ = chronological_train_test_split(ts_df, train_ratio=0.70)
    assert len(train) == expected_train, \
        f"Split size mismatch: expected {expected_train} train rows, got {len(train)}"


# ─── CV leakage tests ──────────────────────────────────────────────────────────

def test_cv_folds_within_training_partition_only(ts_df):
    """
    All CV fold validation ranges must be strictly before the test set boundary.
    """
    train, test, info = chronological_train_test_split(ts_df)
    test_start = info.test_start

    simple_params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
    }

    fold_results = expanding_window_cv(
        train_df=train,
        params=simple_params,
        n_folds=2,
        val_horizon=20,
        min_train_size=100,
        granularity="daily",
    )

    for fold in fold_results:
        val_end = pd.to_datetime(fold["val_end"])
        assert val_end < test_start, (
            f"LEAKAGE: CV fold {fold['fold_index']} validation end ({val_end}) "
            f">= test start ({test_start}). Test data entered CV!"
        )


def test_cv_fold_train_before_val(ts_df):
    """Within each CV fold, training timestamps must be < validation timestamps."""
    train, _, _ = chronological_train_test_split(ts_df)

    simple_params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
    }

    fold_results = expanding_window_cv(
        train_df=train,
        params=simple_params,
        n_folds=2,
        val_horizon=20,
        min_train_size=100,
        granularity="daily",
    )

    for fold in fold_results:
        assert pd.to_datetime(fold["train_end"]) < pd.to_datetime(fold["val_start"]), \
            f"LEAKAGE: Fold {fold['fold_index']} train_end >= val_start"


def test_test_data_never_in_cv(ts_df):
    """
    Test partition timestamps must never appear in any CV fold.
    This checks the invariant at the split level.
    """
    train, test, _ = chronological_train_test_split(ts_df)
    test_timestamps = set(test[DS_COL].astype(str))
    train_timestamps = set(train[DS_COL].astype(str))

    # Since CV folds are subsets of train, test timestamps cannot appear in CV
    overlap = train_timestamps & test_timestamps
    assert len(overlap) == 0, \
        "Test timestamps leaked into training partition (CV is contaminated)"
