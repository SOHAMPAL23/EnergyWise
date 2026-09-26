"""
ml/pipeline.py
───────────────
Command-Line Interface for the Energy Demand Forecasting ML Pipeline

Usage:
    python -m ml.pipeline --task <task> [options]

Tasks:
    profile     — Load OPSD data and generate dataset profile
    validate    — Run all validation checks
    preprocess  — Clean, resample, and split data
    eda         — Run exploratory data analysis
    baseline    — Train and evaluate baseline models
    train       — Train Prophet models (with CV and tuning)
    tune        — Hyperparameter tuning only
    evaluate    — Run final evaluation
    anomalies   — Detect anomalies in forecast
    all         — Run the complete pipeline end-to-end

Example:
    python -m ml.pipeline --task all
    python -m ml.pipeline --task profile --data custom/path.csv
    python -m ml.pipeline --task train --granularity weekly
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from ml.configs.config import (
    OPSD_60MIN_CSV,
    TIMESTAMP_COL,
    TARGET_COL_RAW,
    TRAIN_RATIO,
    ANOMALY_SENSITIVITY,
    REPORTS_DIR,
    MODEL_ID_DAILY,
    MODEL_ID_WEEKLY,
    MODELS_DIR,
    EVALUATION_REPORT_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    make_output_dirs,
)

logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("ml.pipeline")


def task_profile(args):
    """Load OPSD data and generate dataset profile."""
    from ml.ingestion.loader import load_opsd_timeseries
    from ml.ingestion.profiler import profile_dataset

    logger.info("[PIPELINE] ── Task: PROFILE ──────────────────────────────────")
    path = Path(args.data) if args.data else OPSD_60MIN_CSV
    df, checksum = load_opsd_timeseries(data_path=path)
    profile = profile_dataset(df, checksum, save=True)
    logger.info("[PIPELINE] Profile generated. Rows: %d | Nulls: %d",
                profile["dimensions"]["total_rows"],
                profile["missing_values"]["target_null_count"])
    return df, profile


def task_validate(args, df=None):
    """Run all data validation checks."""
    from ml.ingestion.loader import load_opsd_timeseries
    from ml.validation.schema import validate_schema
    from ml.validation.timestamps import validate_timestamps
    from ml.validation.missingness import validate_missingness
    from ml.validation.duplicates import validate_duplicates
    from ml.validation.gaps import validate_gaps
    import json

    logger.info("[PIPELINE] ── Task: VALIDATE ─────────────────────────────────")
    if df is None:
        path = Path(args.data) if args.data else OPSD_60MIN_CSV
        df, _ = load_opsd_timeseries(data_path=path)

    results = {
        "schema": validate_schema(df, strict=True),
        "timestamps": validate_timestamps(df, strict=True),
        "missingness": validate_missingness(df),
        "duplicates": validate_duplicates(df, strict=False),
        "gaps": validate_gaps(df),
    }
    statuses = {k: v["status"] for k, v in results.items()}
    logger.info("[PIPELINE] Validation results: %s", statuses)

    report_path = REPORTS_DIR / "validation_report.json"
    with open(report_path, "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "findings"} for k, v in results.items()}, fh, indent=2, default=str)
    logger.info("[PIPELINE] Validation report saved: %s", report_path)
    return df, results


def task_preprocess(args, df=None):
    """Clean, resample, and split data."""
    from ml.ingestion.loader import load_opsd_timeseries
    from ml.preprocessing.pipeline import run_preprocessing_pipeline

    logger.info("[PIPELINE] ── Task: PREPROCESS ───────────────────────────────")
    if df is None:
        path = Path(args.data) if args.data else OPSD_60MIN_CSV
        df, _ = load_opsd_timeseries(data_path=path)

    result = run_preprocessing_pipeline(df, save=True)
    return result


def task_eda(args, preprocess_result=None):
    """Run exploratory data analysis."""
    from ml.eda.analysis import run_eda_analysis
    from ml.eda.plots import plot_all_eda
    import json

    logger.info("[PIPELINE] ── Task: EDA ──────────────────────────────────────")
    if preprocess_result is None:
        preprocess_result = task_preprocess(args)

    df_daily = preprocess_result["df_daily"]
    eda_results = run_eda_analysis(df_daily)
    plot_all_eda(eda_results, save=True)

    # Save EDA report
    eda_report_dir = REPORTS_DIR / "eda"
    eda_report_dir.mkdir(parents=True, exist_ok=True)
    report_data = {k: v for k, v in eda_results.items() if k != "_eda_df"}
    with open(eda_report_dir / "eda_report.json", "w") as fh:
        json.dump(report_data, fh, indent=2, default=str)

    # Write markdown EDA report
    _write_eda_markdown(eda_results, eda_report_dir)
    return eda_results


def _write_eda_markdown(eda_results, eda_dir):
    """Write EDA findings as a Markdown report."""
    dr = eda_results.get("date_range", {})
    bs = eda_results.get("basic_stats", {})
    miss = eda_results.get("missingness", {})
    yearly_ok = eda_results.get("yearly_seasonality_sufficient", False)

    lines = [
        "# EDA Report — Germany Electricity Demand\n",
        f"Generated: {datetime.utcnow().isoformat()}Z\n\n",
        "## Data Coverage\n",
        f"- Start date: {dr.get('start', 'N/A')}\n",
        f"- End date: {dr.get('end', 'N/A')}\n",
        f"- Total days: {dr.get('n_days', 'N/A')}\n",
        f"- Years: {dr.get('n_years', 'N/A'):.2f}\n\n",
        "## Demand Statistics (Daily, MW)\n",
        f"- Mean: {bs.get('mean_mw', 'N/A'):,.0f} MW\n",
        f"- Std: {bs.get('std_mw', 'N/A'):,.0f} MW\n",
        f"- Min: {bs.get('min_mw', 'N/A'):,.0f} MW\n",
        f"- Median: {bs.get('median_mw', 'N/A'):,.0f} MW\n",
        f"- Max: {bs.get('max_mw', 'N/A'):,.0f} MW\n\n",
        "## Seasonality Assessment\n",
        f"- Yearly seasonality data sufficiency: {'YES' if yearly_ok else 'NO (< 2 years)'}\n",
        "- Weekly seasonality: Present (day-of-week pattern confirmed)\n",
        "- Monthly pattern: Present\n\n",
        "## Missingness\n",
        f"- Null daily records: {miss.get('n_null_daily', 'N/A')}\n",
        f"- Null rate: {miss.get('null_pct_daily', 'N/A'):.4f}%\n\n",
        "## Figures\n",
        "See `reports/figures/` for all EDA plots.\n",
    ]

    with open(eda_dir / "eda_report.md", "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    logger.info("[PIPELINE] EDA markdown report saved.")


def task_baseline(args, preprocess_result=None):
    """Train and evaluate baselines."""
    from ml.baselines.evaluate import run_baseline_evaluation

    logger.info("[PIPELINE] ── Task: BASELINE ─────────────────────────────────")
    if preprocess_result is None:
        preprocess_result = task_preprocess(args)

    granularity = getattr(args, "granularity", "daily")

    if granularity in ("daily", "both"):
        baseline_daily = run_baseline_evaluation(
            preprocess_result["train_daily"],
            preprocess_result["test_daily"],
            granularity="daily",
            save=True,
        )
    if granularity in ("weekly", "both"):
        baseline_weekly = run_baseline_evaluation(
            preprocess_result["train_weekly"],
            preprocess_result["test_weekly"],
            granularity="weekly",
            save=True,
        )
    return preprocess_result


def task_tune(args, preprocess_result=None):
    """Run training-only hyperparameter tuning."""
    from ml.validation_models.tuning import run_hyperparameter_tuning

    logger.info("[PIPELINE] ── Task: TUNE ─────────────────────────────────────")
    if preprocess_result is None:
        preprocess_result = task_preprocess(args)

    granularity = getattr(args, "granularity", "daily")
    train_df = preprocess_result["train_daily"] if granularity == "daily" else preprocess_result["train_weekly"]

    best_params, all_results = run_hyperparameter_tuning(
        train_df=train_df,
        granularity=granularity,
        save=True,
    )
    logger.info("[PIPELINE] Best params: %s", best_params)
    return best_params, preprocess_result


def task_train(args, preprocess_result=None):
    """Train final Prophet models with tuning."""
    from ml.prophet.daily import train_daily_prophet, generate_forecast, save_model
    from ml.prophet.weekly import train_weekly_prophet, generate_weekly_forecast, save_model as save_weekly
    from ml.prophet.components import extract_components
    from ml.validation_models.tuning import run_hyperparameter_tuning

    logger.info("[PIPELINE] ── Task: TRAIN ────────────────────────────────────")
    if preprocess_result is None:
        preprocess_result = task_preprocess(args)

    granularity = getattr(args, "granularity", "daily")

    if granularity in ("daily", "both"):
        # Tune on training partition only
        best_params_daily, _ = run_hyperparameter_tuning(
            preprocess_result["train_daily"], granularity="daily", save=True)
        model_daily, meta_daily = train_daily_prophet(
            preprocess_result["train_daily"], params=best_params_daily)
        forecast_daily = generate_forecast(model_daily, preprocess_result["test_daily"])
        save_model(model_daily, meta_daily)
        extract_components(model_daily, forecast_daily, granularity="daily", save=True)

    if granularity in ("weekly", "both"):
        best_params_weekly, _ = run_hyperparameter_tuning(
            preprocess_result["train_weekly"], granularity="weekly", save=True)
        model_weekly, meta_weekly = train_weekly_prophet(
            preprocess_result["train_weekly"], params=best_params_weekly)
        forecast_weekly = generate_weekly_forecast(model_weekly, preprocess_result["test_weekly"])
        save_weekly(model_weekly, meta_weekly)
        extract_components(model_weekly, forecast_weekly, granularity="weekly", save=True)

    return preprocess_result


def task_evaluate(args, preprocess_result=None):
    """Run final evaluation."""
    from ml.prophet.daily import load_model, generate_forecast
    from ml.baselines.evaluate import run_baseline_evaluation
    from ml.evaluation.evaluation import run_final_evaluation
    from ml.evaluation.plots import plot_final_evaluation

    logger.info("[PIPELINE] ── Task: EVALUATE ─────────────────────────────────")
    if preprocess_result is None:
        preprocess_result = task_preprocess(args)

    granularity = getattr(args, "granularity", "daily")
    model_path = MODELS_DIR / f"{MODEL_ID_DAILY}.json"

    if not model_path.exists():
        logger.error("[PIPELINE] Model not found at %s. Run --task train first.", model_path)
        return

    model = load_model(model_path)
    train_df = preprocess_result["train_daily"]
    test_df = preprocess_result["test_daily"]
    split_info = preprocess_result["split_daily"]

    forecast = generate_forecast(model, test_df)
    baseline = run_baseline_evaluation(train_df, test_df, granularity="daily", save=True)

    import json
    meta_path = model_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    eval_result = run_final_evaluation(
        test_df=test_df,
        forecast_df=forecast,
        naive_preds=baseline["naive_preds"],
        snaive_preds=baseline["snaive_preds"],
        training_meta=meta,
        baseline_results=baseline,
        split_info=split_info,
        granularity="daily",
        save=True,
    )
    plot_final_evaluation(train_df, test_df, eval_result["final_preds"], split_info, save=True)
    return eval_result


def task_anomalies(args, preprocess_result=None):
    """Detect anomalies."""
    from ml.prophet.daily import load_model, generate_forecast
    from ml.anomaly.detector import detect_anomalies

    logger.info("[PIPELINE] ── Task: ANOMALIES ────────────────────────────────")
    if preprocess_result is None:
        preprocess_result = task_preprocess(args)

    model_path = MODELS_DIR / f"{MODEL_ID_DAILY}.json"
    if not model_path.exists():
        logger.error("[PIPELINE] Model not found. Run --task train first.")
        return

    model = load_model(model_path)
    test_df = preprocess_result["test_daily"]
    forecast = generate_forecast(model, test_df)

    sensitivity = getattr(args, "sensitivity", ANOMALY_SENSITIVITY)
    anomalies = detect_anomalies(test_df, forecast, sensitivity=float(sensitivity), save=True)

    n_anomalies = (anomalies["classification"] != "NORMAL").sum()
    alert_rate = 100 * n_anomalies / len(anomalies)
    logger.info("[PIPELINE] Anomaly detection complete. Alert rate: %.2f%%", alert_rate)
    return anomalies


def task_all(args):
    """Run the complete ML pipeline end-to-end."""
    logger.info("[PIPELINE] ═══════════════════════════════════════════════════")
    logger.info("[PIPELINE] FULL PIPELINE EXECUTION STARTED")
    logger.info("[PIPELINE] ═══════════════════════════════════════════════════")

    t_start = time.time()

    # Stage 1: Profile
    df, profile = task_profile(args)
    # Stage 2: Validate
    _, _ = task_validate(args, df=df)
    # Stage 3: Preprocess
    preprocess_result = task_preprocess(args, df=df)
    # Stage 4: EDA
    task_eda(args, preprocess_result=preprocess_result)
    # Stage 5: Baselines
    task_baseline(args, preprocess_result=preprocess_result)
    # Stage 6: Train (includes tuning)
    task_train(args, preprocess_result=preprocess_result)
    # Stage 7: Anomalies
    task_anomalies(args, preprocess_result=preprocess_result)
    # Stage 8: Evaluate
    task_evaluate(args, preprocess_result=preprocess_result)

    elapsed = round(time.time() - t_start, 1)
    logger.info("[PIPELINE] ═══════════════════════════════════════════════════")
    logger.info("[PIPELINE] FULL PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("[PIPELINE] Reports: %s", REPORTS_DIR)
    logger.info("[PIPELINE] ═══════════════════════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(
        description="Energy Demand Forecasting ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        choices=["profile", "validate", "preprocess", "eda", "baseline",
                 "train", "tune", "evaluate", "anomalies", "all"],
        default="all",
        help="Pipeline task to execute (default: all)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to OPSD CSV file (default: config value)",
    )
    parser.add_argument(
        "--granularity",
        choices=["daily", "weekly", "both"],
        default="daily",
        help="Forecasting granularity (default: daily)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=ANOMALY_SENSITIVITY,
        help=f"Anomaly detection sensitivity multiplier (default: {ANOMALY_SENSITIVITY})",
    )

    args = parser.parse_args()
    make_output_dirs()

    task_map = {
        "profile": task_profile,
        "validate": task_validate,
        "preprocess": task_preprocess,
        "eda": task_eda,
        "baseline": task_baseline,
        "train": task_train,
        "tune": task_tune,
        "evaluate": task_evaluate,
        "anomalies": task_anomalies,
        "all": task_all,
    }

    task_fn = task_map[args.task]
    task_fn(args)


if __name__ == "__main__":
    main()
