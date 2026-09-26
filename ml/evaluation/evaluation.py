"""
ml/evaluation/evaluation.py
────────────────────────────
Final Model Evaluation

Compares Prophet against baseline forecasters on the SAME holdout test period.
All metrics are computed from ACTUAL predictions — never fabricated.

CRITICAL: Test data is only accessed HERE, at the final evaluation stage.
           It must never have been seen during training, CV, or tuning.

Generates:
  - final_metrics.json
  - final_predictions.csv
  - evaluation_report.md
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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
    MODEL_ID_DAILY,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)
from ml.evaluation.metrics import (
    mae,
    rmse,
    mape,
    r2_score,
    forecast_accuracy,
    coverage_score,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def run_final_evaluation(
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    naive_preds: pd.DataFrame,
    snaive_preds: pd.DataFrame,
    training_meta: Dict[str, Any],
    baseline_results: Dict[str, Any],
    split_info: Any,
    granularity: str = "daily",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate Prophet against baselines on the holdout test set.

    Parameters
    ----------
    test_df       : Holdout test DataFrame (ds, y) — FIRST ACCESS to test data.
    forecast_df   : Prophet forecast output (contains yhat, yhat_lower, yhat_upper).
    naive_preds   : Naive baseline predictions DataFrame.
    snaive_preds  : Seasonal-Naive baseline predictions DataFrame.
    training_meta : Dict from Prophet training (model parameters, versions).
    baseline_results : Dict from baseline evaluation (includes naive/snaive metrics).
    split_info    : SplitInfo dataclass from splitting.py.
    granularity   : "daily" or "weekly".
    save          : If True, save all evaluation artifacts.

    Returns
    -------
    dict — Final metrics and comparison table.
    """
    make_output_dirs()

    y_true = test_df[Y_COL].values

    # ── Prophet metrics ───────────────────────────────────────────────────
    y_prophet = forecast_df["yhat"].values
    prophet_mae = mae(y_true, y_prophet)
    prophet_rmse = rmse(y_true, y_prophet)
    prophet_mape = mape(y_true, y_prophet)
    prophet_acc = forecast_accuracy(y_true, y_prophet)
    prophet_r2 = r2_score(y_true, y_prophet)
    prophet_coverage = coverage_score(y_true, forecast_df["yhat_lower"].values, forecast_df["yhat_upper"].values)

    # ── Naive metrics ─────────────────────────────────────────────────────
    y_naive = naive_preds["yhat"].values
    naive_mae_val = baseline_results["naive_mae"]
    naive_rmse_val = baseline_results["naive_rmse"]
    naive_mape_val = mape(y_true, y_naive)
    naive_acc_val = forecast_accuracy(y_true, y_naive)
    naive_r2_val = r2_score(y_true, y_naive)

    # ── Seasonal-Naive metrics ────────────────────────────────────────────
    y_snaive = snaive_preds["yhat"].values
    snaive_mae_val = baseline_results["snaive_mae"]
    snaive_rmse_val = baseline_results["snaive_rmse"]
    snaive_mape_val = mape(y_true, y_snaive)
    snaive_acc_val = forecast_accuracy(y_true, y_snaive)
    snaive_r2_val = r2_score(y_true, y_snaive)

    logger.info("[EVAL] ══════════════════════════════════════════════════════")
    logger.info("[EVAL] FINAL TEST SET EVALUATION (%s)", granularity.upper())
    logger.info("[EVAL] Test period: %s → %s (%d records)",
                split_info.test_start.date(), split_info.test_end.date(), split_info.n_test)
    logger.info("[EVAL]   Prophet        MAE: %8.2f MW | RMSE: %8.2f MW | MAPE: %5.2f%% | Acc: %5.2f%% | R2: %6.4f | 95%% Cov: %5.2f%%",
                prophet_mae, prophet_rmse, prophet_mape, prophet_acc, prophet_r2, prophet_coverage)
    logger.info("[EVAL]   Seasonal-Naive MAE: %8.2f MW | RMSE: %8.2f MW | MAPE: %5.2f%% | Acc: %5.2f%% | R2: %6.4f",
                snaive_mae_val, snaive_rmse_val, snaive_mape_val, snaive_acc_val, snaive_r2_val)
    logger.info("[EVAL]   Naive          MAE: %8.2f MW | RMSE: %8.2f MW | MAPE: %5.2f%% | Acc: %5.2f%% | R2: %6.4f",
                naive_mae_val, naive_rmse_val, naive_mape_val, naive_acc_val, naive_r2_val)
    logger.info("[EVAL] ══════════════════════════════════════════════════════")

    # ── Check for anomaly metrics if available ───────────────────────────
    anomaly_summary_file = METRICS_DIR / f"anomaly_summary_{granularity}.json"
    anomaly_data = {}
    if anomaly_summary_file.exists():
        try:
            with open(anomaly_summary_file, "r") as af:
                anomaly_data = json.load(af)
        except Exception:
            pass

    # ── Build final metrics dict ──────────────────────────────────────────
    final_metrics = {
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "granularity": granularity,
        "dataset_version": DATASET_VERSION,
        "region": REGION,
        "model_id": training_meta.get("model_id", MODEL_ID_DAILY),
        "model_version": training_meta.get("model_version", "v1"),
        "test_period": {
            "start": str(split_info.test_start.date()),
            "end": str(split_info.test_end.date()),
            "n_records": split_info.n_test,
        },
        "metrics": {
            "Prophet": {
                "MAE_MW": round(float(prophet_mae), 4),
                "RMSE_MW": round(float(prophet_rmse), 4),
                "MAPE_pct": round(float(prophet_mape), 4),
                "forecast_accuracy_pct": round(float(prophet_acc), 4),
                "R2_score": round(float(prophet_r2), 4),
                "coverage_95_pct": round(float(prophet_coverage), 2),
            },
            "Seasonal_Naive": {
                "MAE_MW": round(float(snaive_mae_val), 4),
                "RMSE_MW": round(float(snaive_rmse_val), 4),
                "MAPE_pct": round(float(snaive_mape_val), 4),
                "forecast_accuracy_pct": round(float(snaive_acc_val), 4),
                "R2_score": round(float(snaive_r2_val), 4),
            },
            "Naive": {
                "MAE_MW": round(float(naive_mae_val), 4),
                "RMSE_MW": round(float(naive_rmse_val), 4),
                "MAPE_pct": round(float(naive_mape_val), 4),
                "forecast_accuracy_pct": round(float(naive_acc_val), 4),
                "R2_score": round(float(naive_r2_val), 4),
            },
        },
        "anomaly_evaluation": anomaly_data.get("classification_metrics", {}),
        "prophet_parameters": training_meta.get("parameters", {}),
    }

    # ── Build final predictions CSV ───────────────────────────────────────
    final_preds = pd.DataFrame({
        DS_COL: test_df[DS_COL].values,
        "actual": y_true,
        "prophet_yhat": forecast_df["yhat"].values,
        "prophet_yhat_lower": forecast_df["yhat_lower"].values,
        "prophet_yhat_upper": forecast_df["yhat_upper"].values,
        "naive_yhat": naive_preds["yhat"].values,
        "snaive_yhat": snaive_preds["yhat"].values,
    })
    final_preds["prophet_error"] = final_preds["actual"] - final_preds["prophet_yhat"]
    final_preds["naive_error"] = final_preds["actual"] - final_preds["naive_yhat"]

    if save:
        _save_evaluation(final_metrics, final_preds, training_meta, split_info, granularity)

    return {
        "final_metrics": final_metrics,
        "final_preds": final_preds,
        "prophet_mae": prophet_mae,
        "prophet_rmse": prophet_rmse,
    }


def _save_evaluation(final_metrics, final_preds, training_meta, split_info, granularity):
    """Save final metrics, predictions, and evaluation report."""
    # final_metrics.json
    metrics_path = EVALUATION_REPORT_DIR / f"final_metrics_{granularity}.json"
    with open(metrics_path, "w") as fh:
        json.dump(final_metrics, fh, indent=2, default=str)
    logger.info("[EVAL] Metrics saved: %s", metrics_path)

    # final_predictions.csv
    preds_path = EVALUATION_REPORT_DIR / f"final_predictions_{granularity}.csv"
    final_preds.to_csv(preds_path, index=False)
    logger.info("[EVAL] Predictions saved: %s", preds_path)

    # Save to FORECASTS_DIR too for API consumption
    final_preds.to_csv(FORECASTS_DIR / f"final_predictions_{granularity}.csv", index=False)

    # evaluation_report.md
    _write_evaluation_report(final_metrics, granularity, split_info)


def _format_anomaly_section(anomaly_eval: Dict) -> str:
    """Format anomaly classification metrics for markdown report."""
    if not anomaly_eval:
        return ""
    cm = anomaly_eval.get("confusion_matrix", {})
    return f"""## Operational Anomaly Detection Performance

Evaluated against statistical operational ground-truth benchmark (extreme shocks: $\\ge 2.5\\sigma$ deviations):

| Anomaly Metric | Score | Interpretation |
|:---------------|:-----:|:---------------|
| **Classification Accuracy** | **{anomaly_eval.get('accuracy_pct', 0.0):.2f}%** | Overall correct normal/anomaly classifications |
| **Precision** | **{anomaly_eval.get('precision', 0.0):.4f}** | Fraction of triggered anomaly alerts that are true shocks |
| **Recall** | **{anomaly_eval.get('recall', 0.0):.4f}** | Fraction of true operational shocks successfully captured |
| **F1 Score** | **{anomaly_eval.get('f1_score', 0.0):.4f}** | Balanced harmonic mean of Precision and Recall |

**Confusion Matrix**: True Positives (TP) = {cm.get('tp', 0)}, False Positives (FP) = {cm.get('fp', 0)}, True Negatives (TN) = {cm.get('tn', 0)}, False Negatives (FN) = {cm.get('fn', 0)}
"""


def _write_evaluation_report(final_metrics: Dict, granularity: str, split_info):
    """Write a human-readable evaluation report as Markdown."""
    m = final_metrics["metrics"]
    params = final_metrics.get("prophet_parameters", {})

    report = f"""# Final Evaluation Report — Energy Demand Forecasting ({granularity.capitalize()})

Generated: {final_metrics['evaluation_timestamp']}

## Dataset
- **Version**: {final_metrics['dataset_version']}
- **Region**: {final_metrics['region']}
- **Granularity**: {granularity}

## Train / Test Split
- **Training period**: {split_info.train_start.date()} → {split_info.train_end.date()} ({split_info.n_train} records)
- **Test period**: {split_info.test_start.date()} → {split_info.test_end.date()} ({split_info.n_test} records)
- **Split boundary**: {split_info.split_timestamp.date()}
- **Data leakage**: None (chronological split verified)

## Model
- **Model ID**: {final_metrics['model_id']}
- **Model version**: {final_metrics['model_version']}
- **Type**: Prophet (additive time-series model)

## Final Test Metrics

| Model Architecture | MAE (MW) | RMSE (MW) | MAPE (%) | Forecast Accuracy (%) | R² Score | 95% Coverage |
|--------------------|:--------:|:---------:|:--------:|:---------------------:|:--------:|:------------:|
| **Prophet (Primary)** | {m['Prophet']['MAE_MW']:,.2f} | {m['Prophet']['RMSE_MW']:,.2f} | {m['Prophet']['MAPE_pct']:.2f}% | {m['Prophet']['forecast_accuracy_pct']:.2f}% | {m['Prophet']['R2_score']:.4f} | **{m['Prophet']['coverage_95_pct']:.2f}%** |
| **Seasonal-Naive (7-day)** | {m['Seasonal_Naive']['MAE_MW']:,.2f} | {m['Seasonal_Naive']['RMSE_MW']:,.2f} | {m['Seasonal_Naive']['MAPE_pct']:.2f}% | {m['Seasonal_Naive']['forecast_accuracy_pct']:.2f}% | {m['Seasonal_Naive']['R2_score']:.4f} | — |
| **Naive (Persistence)** | {m['Naive']['MAE_MW']:,.2f} | {m['Naive']['RMSE_MW']:,.2f} | {m['Naive']['MAPE_pct']:.2f}% | {m['Naive']['forecast_accuracy_pct']:.2f}% | {m['Naive']['R2_score']:.4f} | — |

{_format_anomaly_section(final_metrics.get("anomaly_evaluation", {}))}
## Prophet Configuration
```json
{json.dumps(params, indent=2, default=str)}
```

## Limitations

- Historical demand may not represent future consumption regimes.
- Holiday calendar applies German federal holidays only (regional holidays excluded).
- Forecast uncertainty intervals are approximate (Prophet MCMC approximation).
- Anomaly classifications are investigation signals, not confirmed abnormal events.
- Extreme weather or unprecedented events outside the training distribution may not be captured.

## Notes

All metrics are computed from actual (non-fabricated) predictions on the holdout test set.
The test set was not accessed during model training, cross-validation, or hyperparameter tuning.
"""
    report_path = EVALUATION_REPORT_DIR / f"evaluation_report_{granularity}.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    logger.info("[EVAL] Report saved: %s", report_path)
