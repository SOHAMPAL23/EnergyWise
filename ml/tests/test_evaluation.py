"""
Final-evaluation tests: metrics must be recomputed from predictions (never copied),
weekly aggregation must use complete Monday–Sunday weeks, and misaligned inputs rejected.
"""

import numpy as np
import pandas as pd
import pytest

from ml.configs.config import DS_COL, Y_COL
from ml.evaluation.evaluation import run_final_evaluation, weekly_aggregate
from ml.evaluation.metrics import mae, rmse
from ml.preprocessing.splitting import SplitInfo


def make_inputs(n=21, start="2019-01-10"):     # starts Thursday -> partial edge weeks
    ds = pd.date_range(start, periods=n, freq="D")
    rng = np.random.RandomState(0)
    y = 50000 + rng.normal(0, 1000, n)
    test = pd.DataFrame({DS_COL: ds, Y_COL: y})
    fc = pd.DataFrame({DS_COL: ds, "yhat": y + 500, "yhat_lower": y - 1500, "yhat_upper": y + 2500})
    naive = pd.DataFrame({DS_COL: ds, "yhat": np.full(n, 50000.0)})
    snaive = pd.DataFrame({DS_COL: ds, "yhat": y + 2000})
    info = SplitInfo(ds[0], ds[0] - pd.Timedelta(days=100), ds[0] - pd.Timedelta(days=1), ds[0], ds[-1], 100, n, 70, 30)
    return test, fc, naive, snaive, info


def test_metrics_are_recomputed_from_predictions():
    test, fc, naive, snaive, info = make_inputs()
    r = run_final_evaluation(test, fc, naive, snaive, {"parameters": {"interval_width": 0.95}}, info, save=False)
    m = r["final_metrics"]["metrics"]
    assert m["Prophet"]["MAE_MW"] == pytest.approx(500.0, abs=1e-3)
    assert m["Prophet"]["RMSE_MW"] == pytest.approx(500.0, abs=1e-3)
    assert m["Seasonal_Naive"]["MAE_MW"] == pytest.approx(2000.0, abs=1e-3)
    assert m["Naive"]["MAE_MW"] == pytest.approx(mae(test[Y_COL], naive["yhat"]), abs=1e-3)
    assert m["Naive"]["RMSE_MW"] == pytest.approx(rmse(test[Y_COL], naive["yhat"]), abs=1e-3)
    assert r["final_metrics"]["ranking_by_MAE"][0] == "Prophet"


def test_coverage_computed_from_interval():
    test, fc, naive, snaive, info = make_inputs()
    r = run_final_evaluation(test, fc, naive, snaive, {"parameters": {"interval_width": 0.95}}, info, save=False)
    assert r["final_metrics"]["metrics"]["Prophet"]["coverage_95_pct"] == 100.0
    fc2 = fc.copy()
    fc2["yhat_upper"] = fc2["yhat"] + 1                   # intervals now too narrow
    fc2["yhat_lower"] = fc2["yhat"] - 1
    r2 = run_final_evaluation(test, fc2, naive, snaive, {"parameters": {}}, info, save=False)
    assert r2["final_metrics"]["metrics"]["Prophet"]["coverage_95_pct"] == 0.0


def test_weekly_aggregation_uses_only_complete_weeks():
    test, fc, naive, snaive, info = make_inputs(21)       # Thu 10 Jan .. Wed 30 Jan
    r = run_final_evaluation(test, fc, naive, snaive, {"parameters": {}}, info, save=False)
    wk = r["weekly_comparison"]
    assert list(wk[DS_COL]) == [pd.Timestamp("2019-01-14"), pd.Timestamp("2019-01-21")]   # 2 complete Mon–Sun weeks
    expected = test[(test[DS_COL] >= "2019-01-14") & (test[DS_COL] <= "2019-01-20")][Y_COL].mean()
    assert wk["actual"].iloc[0] == pytest.approx(expected)
    assert "weekly_aggregated_metrics" in r["final_metrics"]


def test_largest_error_periods_identify_injected_shock():
    test, fc, naive, snaive, info = make_inputs(120, "2019-01-01")
    test.loc[75:80, Y_COL] -= 30000                        # shock in March 2019
    r = run_final_evaluation(test, fc, naive, snaive, {"parameters": {}}, info, save=False)
    assert r["error_periods"].iloc[0]["period"] == "2019-03"
    worst_day = r["largest_errors"].iloc[0][DS_COL]
    assert pd.Timestamp("2019-03-17") <= worst_day <= pd.Timestamp("2019-03-23")


def test_misaligned_predictions_rejected():
    test, fc, naive, snaive, info = make_inputs()
    with pytest.raises(ValueError):
        run_final_evaluation(test, fc.iloc[::-1].reset_index(drop=True), naive, snaive, {"parameters": {}}, info, save=False)
