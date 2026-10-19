"""
Rolling-origin / expanding-window CV: fold geometry and tuning bookkeeping.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ml.experiments.tracker import ExperimentTracker, load_experiments
from ml.validation_models.cross_validation import make_rolling_origin_folds, run_cv
from ml.validation_models.tuning import run_tuning
from ml.configs.config import DS_COL, Y_COL, PARAM_GRID, PARAM_SEARCH_SPACE


def synth(n=700, seed=1):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = 50000 + 3000 * np.sin(2 * np.pi * t / 7) + 5000 * np.sin(2 * np.pi * t / 365.25) \
        + 4 * t + rng.normal(0, 400, n)
    return pd.DataFrame({DS_COL: pd.date_range("2015-01-01", periods=n, freq="D"), Y_COL: y})


# ── fold geometry ────────────────────────────────────────────────────────────

def test_folds_are_expanding_and_inside_partition():
    n = 1470
    folds = make_rolling_origin_folds(n, min_train=730, horizon=180, step=90)
    origins = [o for o, _ in folds]
    assert origins == sorted(origins)                     # chronological
    assert all(b > a for a, b in zip(origins, origins[1:]))   # train window GROWS
    assert origins[0] >= 730                              # minimum history respected
    for o, e in folds:
        assert 0 < o < e <= n                             # val strictly after train, inside partition
        assert e - o == 180
    assert folds[-1][1] == n                              # most recent data is validated
    assert {b - a for a, b in zip(origins, origins[1:])} == {90}


def test_n_folds_keeps_most_recent():
    all_f = make_rolling_origin_folds(1000, 300, 100, 100)
    two = make_rolling_origin_folds(1000, 300, 100, 100, n_folds=2)
    assert two == all_f[-2:]


def test_too_small_partition_raises():
    with pytest.raises(ValueError):
        make_rolling_origin_folds(100, min_train=90, horizon=30, step=10)


def test_fold_validation_never_precedes_training_end():
    df = synth(500)
    folds = make_rolling_origin_folds(len(df), 200, 40, 40)
    res = run_cv(df, {"changepoint_prior_scale": 0.05}, "daily", folds=folds)
    assert len(res) == len(folds)
    for r in res:
        assert pd.Timestamp(r["train_end"]) < pd.Timestamp(r["val_start"])
        assert r["train_start"] == "2015-01-01"           # expanding window anchored at row 0
        assert r["MAE"] > 0 and r["RMSE"] >= r["MAE"]


# ── search space ─────────────────────────────────────────────────────────────

def test_grid_covers_all_required_hyperparameters():
    keys = set().union(*[set(p) for p in PARAM_GRID])
    assert {"changepoint_prior_scale", "seasonality_prior_scale", "holidays_prior_scale",
            "n_changepoints", "changepoint_range"} <= keys
    n = 1
    for v in PARAM_SEARCH_SPACE.values():
        n *= len(v)
    assert len(PARAM_GRID) == n == len({json.dumps(p, sort_keys=True) for p in PARAM_GRID})


# ── tuning bookkeeping ───────────────────────────────────────────────────────

SMALL_GRID = [
    {"changepoint_prior_scale": 0.01, "seasonality_prior_scale": 10.0, "holidays_prior_scale": 10.0,
     "n_changepoints": 25, "changepoint_range": 0.8},
    {"changepoint_prior_scale": 0.5, "seasonality_prior_scale": 1.0, "holidays_prior_scale": 1.0,
     "n_changepoints": 10, "changepoint_range": 0.9},
]
SMALL_CV = {"min_train": 300, "horizon": 60, "step": 60}


def test_selection_is_argmin_validation_mae_and_logged(tmp_path):
    df = synth(600)
    tr = ExperimentTracker(tmp_path / "e.csv", tmp_path / "e.json")
    res = run_tuning(df, "daily", param_grid=SMALL_GRID, cv_overrides=SMALL_CV, tracker=tr, save=False)

    best = min(res["trials"], key=lambda t: (t["avg_cv_mae"], t["avg_cv_rmse"]))
    assert res["best_params"] == best["params"]
    assert res["summary"]["test_data_used"] is False

    rows = load_experiments(tmp_path / "e.csv")
    n_folds = res["trials"][0]["n_folds"]
    assert len(rows) == len(SMALL_GRID) * (n_folds + 1)     # every fold + one aggregate per config
    required = {"experiment_id", "model_version", "granularity", "parameters", "training_range",
                "validation_range", "MAE", "RMSE", "timestamp"}
    assert required <= set(rows[0])
    for r in rows:
        assert all(r[k] not in ("", None) for k in required)
        json.loads(r["parameters"])
        assert float(r["MAE"]) > 0
    assert (tmp_path / "e.json").exists()


def test_experiment_validation_ranges_stay_inside_training(tmp_path):
    df = synth(600)
    tr = ExperimentTracker(tmp_path / "e.csv", tmp_path / "e.json")
    run_tuning(df, "daily", param_grid=SMALL_GRID[:1], cv_overrides=SMALL_CV, tracker=tr, save=False)
    train_end = df[DS_COL].max()
    for r in load_experiments(tmp_path / "e.csv"):
        assert pd.Timestamp(r["validation_range"].split("..")[1]) <= train_end
