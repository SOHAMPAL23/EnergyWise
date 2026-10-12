"""
ml/evaluation/evaluation.py
────────────────────────────
Final hold-out evaluation: Naive vs Seasonal-Naive vs Prophet.

Every number is recomputed from the prediction frames passed in — nothing is read
back from earlier reports. This is the ONLY place (besides baseline scoring) where the
hold-out test set is read, and it is reached exclusively through
HoldoutVault.open("final_evaluation").

Outputs
  final_metrics_{g}.json          MAE / RMSE (+ MAPE, R2, coverage) for all 3 models
  final_predictions_{g}.csv       aligned actual + 3 forecasts + interval
  weekly_comparison_daily.csv     Mon–Sun aggregated actual vs forecasts (daily only)
  largest_errors_{g}.csv          top individual errors for Prophet
  error_periods_{g}.csv           MAE/RMSE per calendar month (worst first)
  evaluation_report_{g}.md
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from ml.configs.config import (
    DS_COL,
    Y_COL,
    EVALUATION_REPORT_DIR,
    FORECASTS_DIR,
    METRICS_DIR,
    REGION,
    DATASET_VERSION,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)
from ml.evaluation.metrics import mae, rmse, mape, r2_score, coverage_score

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

MODELS = ("naive", "snaive", "prophet")
MODEL_LABELS = {"naive": "Naive", "snaive": "Seasonal_Naive", "prophet": "Prophet"}


def _aligned(test_df, forecast_df, naive_preds, snaive_preds) -> pd.DataFrame:
    ds = pd.to_datetime(test_df[DS_COL]).reset_index(drop=True)
    for name, f in (("forecast", forecast_df), ("naive", naive_preds), ("snaive", snaive_preds)):
        if len(f) != len(test_df) or not (pd.to_datetime(f[DS_COL]).reset_index(drop=True) == ds).all():
            raise ValueError(f"[EVAL] {name} predictions are not aligned with the test timestamps.")
    return pd.DataFrame({
        DS_COL: ds,
        "actual": test_df[Y_COL].to_numpy(dtype=float),
        "prophet_yhat": forecast_df["yhat"].to_numpy(dtype=float),
        "prophet_yhat_lower": forecast_df["yhat_lower"].to_numpy(dtype=float),
        "prophet_yhat_upper": forecast_df["yhat_upper"].to_numpy(dtype=float),
        "naive_yhat": naive_preds["yhat"].to_numpy(dtype=float),
        "snaive_yhat": snaive_preds["yhat"].to_numpy(dtype=float),
    })


def weekly_aggregate(preds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to complete Monday–Sunday weeks (partial edge weeks dropped)."""
    cols = ["actual", "prophet_yhat", "naive_yhat", "snaive_yhat"]
    t = preds.set_index(pd.to_datetime(preds[DS_COL]))[cols]
    wk = t.resample("W-MON", closed="left", label="left").mean()
    n = t["actual"].resample("W-MON", closed="left", label="left").count()
    wk = wk[n == 7].reset_index().rename(columns={"index": DS_COL, DS_COL: DS_COL})
    wk.columns = [DS_COL] + cols
    return wk


def _metrics_block(y, yhat, extra=None) -> Dict[str, float]:
    out = {
        "MAE_MW": round(mae(y, yhat), 4),
        "RMSE_MW": round(rmse(y, yhat), 4),
        "MAPE_pct": round(mape(y, yhat), 4),
        "R2_score": round(r2_score(y, yhat), 4),
    }
    if extra:
        out.update(extra)
    return out


def run_final_evaluation(
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    naive_preds: pd.DataFrame,
    snaive_preds: pd.DataFrame,
    training_meta: Dict[str, Any],
    split_info: Any,
    granularity: str = "daily",
    baseline_results: Dict[str, Any] = None,   # accepted for backwards compatibility; unused
    save: bool = True,
) -> Dict[str, Any]:
    make_output_dirs()
    preds = _aligned(test_df, forecast_df, naive_preds, snaive_preds)
    y = preds["actual"].to_numpy()

    interval_width = preds["prophet_yhat_upper"] - preds["prophet_yhat_lower"]
    cov = coverage_score(y, preds["prophet_yhat_lower"], preds["prophet_yhat_upper"])
    metrics = {
        "Prophet": _metrics_block(y, preds["prophet_yhat"], {
            "coverage_95_pct": round(cov, 2),
            "mean_interval_width_MW": round(float(interval_width.mean()), 2),
        }),
        "Seasonal_Naive": _metrics_block(y, preds["snaive_yhat"]),
        "Naive": _metrics_block(y, preds["naive_yhat"]),
    }

    nominal = float(training_meta.get("parameters", {}).get("interval_width", 0.95)) * 100
    ranking = sorted(metrics, key=lambda k: metrics[k]["MAE_MW"])

    final_metrics: Dict[str, Any] = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "granularity": granularity,
        "dataset_version": DATASET_VERSION,
        "region": REGION,
        "unit": "MW (mean load over the period)",
        "model_id": training_meta.get("model_id"),
        "model_version": training_meta.get("model_version"),
        "training_range": f"{training_meta.get('train_start')}..{training_meta.get('train_end')}",
        "test_period": {
            "start": str(split_info.test_start.date()),
            "end": str(split_info.test_end.date()),
            "n_records": int(split_info.n_test),
        },
        "forecast_mode": ("multi-step from a FIXED origin (end of training); the model is NOT "
                          "refit or updated with test actuals. Horizon = full test length."),
        "metrics": metrics,
        "ranking_by_MAE": ranking,
        "prophet_beats_seasonal_naive": bool(
            metrics["Prophet"]["MAE_MW"] < metrics["Seasonal_Naive"]["MAE_MW"]
            and metrics["Prophet"]["RMSE_MW"] < metrics["Seasonal_Naive"]["RMSE_MW"]),
        "prophet_vs_seasonal_naive_MAE_change_pct": round(
            100 * (metrics["Prophet"]["MAE_MW"] / metrics["Seasonal_Naive"]["MAE_MW"] - 1), 2),
        "interval_calibration": {
            "nominal_pct": nominal,
            "empirical_coverage_pct": round(cov, 2),
            "calibration_gap_pct_points": round(cov - nominal, 2),
        },
        "prophet_parameters": training_meta.get("parameters", {}),
    }

    # Weekly-aggregated comparison (daily model only; weekly model is already weekly)
    weekly_cmp = None
    if granularity == "daily":
        weekly_cmp = weekly_aggregate(preds)
        yw = weekly_cmp["actual"].to_numpy()
        final_metrics["weekly_aggregated_metrics"] = {
            "n_complete_weeks": int(len(weekly_cmp)),
            "note": "Daily forecasts averaged to complete Mon–Sun weeks, then scored.",
            "Prophet": _metrics_block(yw, weekly_cmp["prophet_yhat"]),
            "Seasonal_Naive": _metrics_block(yw, weekly_cmp["snaive_yhat"]),
            "Naive": _metrics_block(yw, weekly_cmp["naive_yhat"]),
        }

    # Error attribution
    preds["prophet_error"] = preds["actual"] - preds["prophet_yhat"]
    preds["naive_error"] = preds["actual"] - preds["naive_yhat"]
    preds["snaive_error"] = preds["actual"] - preds["snaive_yhat"]
    preds["prophet_abs_error"] = preds["prophet_error"].abs()

    largest = preds.nlargest(15, "prophet_abs_error")[
        [DS_COL, "actual", "prophet_yhat", "prophet_error", "prophet_abs_error"]].copy()
    month = preds[DS_COL].dt.to_period("M").astype(str)
    periods = (preds.assign(period=month).groupby("period")
               .agg(n=("actual", "size"),
                    prophet_MAE=("prophet_abs_error", "mean"),
                    prophet_RMSE=("prophet_error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
                    prophet_mean_error=("prophet_error", "mean"),
                    snaive_MAE=("snaive_error", lambda e: float(np.mean(np.abs(e)))))
               .reset_index().sort_values("prophet_MAE", ascending=False).round(2))
    final_metrics["largest_error_periods"] = {
        "worst_months_by_prophet_MAE": periods.head(5).to_dict(orient="records"),
        "worst_days": [
            {"ds": str(r[DS_COL].date()), "actual": round(r["actual"], 1),
             "forecast": round(r["prophet_yhat"], 1), "error": round(r["prophet_error"], 1)}
            for _, r in largest.head(10).iterrows()],
        "bias_mean_error_MW": round(float(preds["prophet_error"].mean()), 2),
    }

    logger.info("[EVAL] %s TEST  Prophet MAE=%.2f RMSE=%.2f | S-Naive MAE=%.2f RMSE=%.2f | "
                "Naive MAE=%.2f RMSE=%.2f | coverage=%.1f%%",
                granularity.upper(),
                metrics["Prophet"]["MAE_MW"], metrics["Prophet"]["RMSE_MW"],
                metrics["Seasonal_Naive"]["MAE_MW"], metrics["Seasonal_Naive"]["RMSE_MW"],
                metrics["Naive"]["MAE_MW"], metrics["Naive"]["RMSE_MW"], cov)

    if save:
        _save_evaluation(final_metrics, preds, weekly_cmp, largest, periods, split_info, granularity)

    return {
        "final_metrics": final_metrics,
        "final_preds": preds,
        "weekly_comparison": weekly_cmp,
        "largest_errors": largest,
        "error_periods": periods,
        "prophet_mae": metrics["Prophet"]["MAE_MW"],
        "prophet_rmse": metrics["Prophet"]["RMSE_MW"],
    }


def _save_evaluation(final_metrics, preds, weekly_cmp, largest, periods, split_info, granularity):
    d = EVALUATION_REPORT_DIR
    with open(d / f"final_metrics_{granularity}.json", "w", encoding="utf-8") as fh:
        json.dump(final_metrics, fh, indent=2, default=str)
    with open(METRICS_DIR / f"final_metrics_{granularity}.json", "w", encoding="utf-8") as fh:
        json.dump(final_metrics, fh, indent=2, default=str)
    preds.to_csv(d / f"final_predictions_{granularity}.csv", index=False)
    preds.to_csv(FORECASTS_DIR / f"final_predictions_{granularity}.csv", index=False)
    largest.to_csv(d / f"largest_errors_{granularity}.csv", index=False)
    periods.to_csv(d / f"error_periods_{granularity}.csv", index=False)
    if weekly_cmp is not None:
        weekly_cmp.to_csv(d / f"weekly_comparison_{granularity}.csv", index=False)
    _write_report(final_metrics, granularity, split_info, periods)


def _write_report(fm: Dict, granularity: str, split_info, periods: pd.DataFrame) -> None:
    m = fm["metrics"]
    rows = "\n".join(
        f"| {label} | {m[k]['MAE_MW']:,.2f} | {m[k]['RMSE_MW']:,.2f} | {m[k]['MAPE_pct']:.2f} | {m[k]['R2_score']:.4f} |"
        for label, k in (("Prophet", "Prophet"), ("Seasonal-Naive", "Seasonal_Naive"), ("Naive", "Naive")))
    weekly = ""
    if "weekly_aggregated_metrics" in fm:
        w = fm["weekly_aggregated_metrics"]
        weekly = (f"\n## Weekly-aggregated comparison ({w['n_complete_weeks']} complete Mon–Sun weeks)\n\n"
                  "| Model | MAE (MW) | RMSE (MW) |\n|---|---:|---:|\n"
                  + "\n".join(f"| {lbl} | {w[k]['MAE_MW']:,.2f} | {w[k]['RMSE_MW']:,.2f} |"
                              for lbl, k in (("Prophet", "Prophet"), ("Seasonal-Naive", "Seasonal_Naive"),
                                             ("Naive", "Naive"))) + "\n")
    worst = periods.head(5)
    worst_tbl = "\n".join(
        f"| {r.period} | {int(r.n)} | {r.prophet_MAE:,.0f} | {r.prophet_RMSE:,.0f} | {r.prophet_mean_error:+,.0f} |"
        for r in worst.itertuples())
    cal = fm["interval_calibration"]
    report = f"""# Final Hold-out Evaluation — {granularity.capitalize()} (Germany load, {fm['unit']})

Generated: {fm['evaluation_timestamp']}

- **Model**: {fm['model_id']} ({fm['model_version']})  — trained on {fm['training_range']}
- **Hold-out test**: {fm['test_period']['start']} → {fm['test_period']['end']} ({fm['test_period']['n_records']} records)
- **Forecast mode**: {fm['forecast_mode']}
- **Hyperparameters were selected by training-only cross-validation; the test set did not influence them.**

## Test metrics

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
{rows}

Ranking by MAE: {' < '.join(fm['ranking_by_MAE'])}.
Prophet vs Seasonal-Naive MAE change: {fm['prophet_vs_seasonal_naive_MAE_change_pct']:+.2f}%
(Prophet beats Seasonal-Naive on both MAE and RMSE: **{fm['prophet_beats_seasonal_naive']}**).

## Prediction-interval calibration
Nominal {cal['nominal_pct']:.0f}% → empirical coverage on test **{cal['empirical_coverage_pct']:.2f}%**
(gap {cal['calibration_gap_pct_points']:+.2f} pp). Mean interval width {m['Prophet']['mean_interval_width_MW']:,.0f} MW.
{weekly}
## Largest-error periods (Prophet, by calendar month)

| Month | n | MAE | RMSE | Mean error (actual − forecast) |
|---|---:|---:|---:|---:|
{worst_tbl}

Bias over whole test: {fm['largest_error_periods']['bias_mean_error_MW']:+,.0f} MW.

## Caveats
- The test window is forecast 100% out-of-sample from a single fixed origin; error grows with horizon.
- Test period contains the 2020 COVID-19 demand shock, which no model trained on 2015–2019 can anticipate.
- Federal-level German holidays only; regional holidays are not modelled.
- Anomaly flags are investigation signals, not ground truth.
"""
    with open(EVALUATION_REPORT_DIR / f"evaluation_report_{granularity}.md", "w", encoding="utf-8") as fh:
        fh.write(report)
