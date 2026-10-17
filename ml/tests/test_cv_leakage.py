"""
Proof-style leakage tests: they OBSERVE which rows reach Prophet during CV/tuning/training,
rather than only checking that the split function is chronological.
"""

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from ml.configs.config import DS_COL, Y_COL
from ml.experiments.tracker import ExperimentTracker
from ml.preprocessing.splitting import chronological_train_test_split
from ml.validation_models.cross_validation import make_rolling_origin_folds, run_cv
from ml.validation_models.guard import (
    AccessAuditor, HoldoutAccessError, HoldoutVault, LeakageError,
)
from ml.validation_models.tuning import run_tuning
from ml.tests.test_cv import synth, SMALL_GRID, SMALL_CV


@pytest.fixture
def full_df():
    return synth(900, seed=3)


def test_no_test_rows_accessed_during_tuning(full_df, tmp_path):
    """Observed Prophet.fit/predict inputs during tuning never reach the test partition."""
    train, test, info = chronological_train_test_split(full_df)
    tr = ExperimentTracker(tmp_path / "e.csv", tmp_path / "e.json")
    with AccessAuditor(holdout_start=info.test_start, label="outer") as outer:
        res = run_tuning(train, "daily", param_grid=SMALL_GRID, cv_overrides=SMALL_CV,
                         holdout_start=info.test_start, tracker=tr, save=False)
    s = outer.summary()
    assert s["n_fit_calls"] > 0 and s["n_predict_calls"] > 0       # the audit actually observed work
    assert pd.Timestamp(s["max_ds_accessed"]) <= train[DS_COL].max()
    assert pd.Timestamp(s["max_ds_accessed"]) < info.test_start
    assert s["n_calls_touching_holdout"] == 0 and s["no_holdout_rows_accessed"]
    assert res["access_audit"]["no_holdout_rows_accessed"]


def test_cv_rejects_input_that_contains_test_rows(full_df):
    _, _, info = chronological_train_test_split(full_df)
    with pytest.raises(LeakageError):
        run_cv(full_df, SMALL_GRID[0], "daily", holdout_start=info.test_start,
               folds=make_rolling_origin_folds(len(full_df), 300, 60, 60))


def test_auditor_trips_when_prophet_sees_test_rows(full_df):
    """Negative control: the auditor must FAIL LOUDLY if test rows ever reach Prophet."""
    _, _, info = chronological_train_test_split(full_df)
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    with AccessAuditor(holdout_start=info.test_start):
        with pytest.raises(LeakageError):
            m.fit(full_df[[DS_COL, Y_COL]])         # includes test rows


def test_auditor_restores_prophet_methods():
    fit, predict = Prophet.fit, Prophet.predict
    with AccessAuditor():
        assert Prophet.fit is not fit
    assert Prophet.fit is fit and Prophet.predict is predict


def test_tuning_result_is_independent_of_test_values(full_df, tmp_path):
    """
    Counterfactual replay: poison every test-partition value, redo split → tuning.
    Selected params AND every fold metric must be bit-identical.
    """
    def run(df, name):
        train, _, info = chronological_train_test_split(df)
        tr = ExperimentTracker(tmp_path / f"{name}.csv", tmp_path / f"{name}.json")
        return run_tuning(train, "daily", param_grid=SMALL_GRID, cv_overrides=SMALL_CV,
                          holdout_start=info.test_start, tracker=tr, save=False)

    n_train = int(np.floor(len(full_df) * 0.70))
    poisoned = full_df.copy()
    rng = np.random.RandomState(0)
    poisoned.loc[n_train:, Y_COL] = rng.uniform(-1e6, 1e6, len(poisoned) - n_train)

    a, b = run(full_df, "clean"), run(poisoned, "poisoned")
    assert a["best_params"] == b["best_params"]
    for ta, tb in zip(a["trials"], b["trials"]):
        assert [f["MAE"] for f in ta["fold_details"]] == [f["MAE"] for f in tb["fold_details"]]
        assert [f["RMSE"] for f in ta["fold_details"]] == [f["RMSE"] for f in tb["fold_details"]]


# ── vault ────────────────────────────────────────────────────────────────────

def test_vault_denies_development_stages_and_logs(full_df):
    _, test, _ = chronological_train_test_split(full_df)
    v = HoldoutVault(test)
    for stage in ("cv", "tuning", "training", "eda"):
        with pytest.raises(HoldoutAccessError):
            v.open(stage)
    assert v.summary()["denied_accesses"] == 4
    out = v.open("final_evaluation")
    assert len(out) == len(test)
    assert v.access_log[-1]["granted"] is True


def test_vault_detects_mutation_and_copies_are_isolated(full_df):
    _, test, _ = chronological_train_test_split(full_df)
    v = HoldoutVault(test)
    out = v.open("final_evaluation")
    out[Y_COL] = 0.0                              # caller mutates its copy
    assert v.verify_unchanged()                   # vault unaffected
    v._df.loc[0, Y_COL] += 1.0                    # tamper with the vault itself
    assert not v.verify_unchanged()
