"""
ml/pipeline.py
───────────────
Command-line interface for the Energy Demand Forecasting ML pipeline.

    python -m ml.pipeline --task all                       # everything, both granularities
    python -m ml.pipeline --task tune --granularity daily
    python -m ml.pipeline --task audit                     # evidence-based PRD audit

Tasks
    profile     load OPSD, profile the dataset
    validate    run the five data-quality validators
    preprocess  clean, resample, 70/30 chronological split
    eda         EDA on the TRAINING partition only
    tune        rolling-origin CV + grid search (training partition only)
    train       fit the final Prophet with the tuned params (training partition only)
    evaluate    hold-out evaluation vs Naive / Seasonal-Naive + anomaly detection + artifacts
    baseline    baseline scores on the hold-out (baselines are fit on train only)
    anomalies   (re)run anomaly detection with --sensitivity
    audit       run the programmatic ML audit
    all         profile → validate → preprocess → eda → (tune → train → evaluate)* → manifest

LEAKAGE DESIGN
    The 30% test partition is wrapped in a HoldoutVault. tune/train/eda only ever receive
    the training frames. The vault opens only for the `evaluate`/`baseline` stages, logs
    every access, and its SHA-256 fingerprint is re-verified at the end.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ml.configs.config import (
    OPSD_60MIN_CSV,
    TRAIN_RATIO,
    ANOMALY_SENSITIVITY,
    REPORTS_DIR,
    AUDIT_REPORT_DIR,
    MODELS_DIR,
    ARTIFACT_METADATA_DIR,
    MODEL_VERSION_DAILY,
    MODEL_VERSION_WEEKLY,
    DATASET_VERSION,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    make_output_dirs,
)
from ml.validation_models.guard import AccessAuditor, HoldoutVault

logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("ml.pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Context: training frames are free to use; test frames live in a locked vault.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Context:
    train: Dict[str, pd.DataFrame]
    vaults: Dict[str, HoldoutVault]
    splits: Dict[str, Any]
    df_daily_train: pd.DataFrame
    data_info: Dict[str, Any]
    cleaning_log: Dict[str, Any]
    tuning: Dict[str, Any] = field(default_factory=dict)
    trained: Dict[str, Any] = field(default_factory=dict)


def _grans(args) -> List[str]:
    g = getattr(args, "granularity", "both")
    return ["daily", "weekly"] if g == "both" else [g]


def _version(g: str) -> str:
    return MODEL_VERSION_WEEKLY if g == "weekly" else MODEL_VERSION_DAILY


def _model_id(g: str) -> str:
    return f"prophet-{g}-{_version(g)}"


def _data_path(args) -> Path:
    return Path(args.data) if getattr(args, "data", None) else OPSD_60MIN_CSV


def _update_evidence(g: str, key: str, value: Any) -> None:
    """Merge one evidence block into reports/audit/leakage_evidence_{g}.json."""
    AUDIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_REPORT_DIR / f"leakage_evidence_{g}.json"
    cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cur[key] = value
    cur["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(cur, indent=2, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────
def task_profile(args):
    from ml.ingestion.loader import load_opsd_timeseries
    from ml.ingestion.profiler import profile_dataset

    logger.info("[PIPELINE] ── PROFILE")
    df, checksum = load_opsd_timeseries(data_path=_data_path(args))
    profile = profile_dataset(df, checksum, save=True)
    return df, checksum, profile


def task_validate(args, df=None):
    from ml.ingestion.loader import load_opsd_timeseries
    from ml.validation.schema import validate_schema
    from ml.validation.timestamps import validate_timestamps
    from ml.validation.missingness import validate_missingness
    from ml.validation.duplicates import validate_duplicates
    from ml.validation.gaps import validate_gaps

    logger.info("[PIPELINE] ── VALIDATE")
    if df is None:
        df, _ = load_opsd_timeseries(data_path=_data_path(args))
    results = {
        "schema": validate_schema(df, strict=True),
        "timestamps": validate_timestamps(df, strict=True),
        "missingness": validate_missingness(df),
        "duplicates": validate_duplicates(df, strict=False),
        "gaps": validate_gaps(df),
    }
    statuses = {k: v["status"] for k, v in results.items()}
    logger.info("[PIPELINE] Validation: %s", statuses)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "validation_report.json", "w", encoding="utf-8") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "findings"} for k, v in results.items()},
                  fh, indent=2, default=str)
    failed = [k for k, s in statuses.items() if s == "FAIL"]
    if failed:
        raise RuntimeError(f"[PIPELINE] Data validation FAILED for: {failed}. Halting before preprocessing.")
    return df, results


def task_preprocess(args, df=None, checksum=None) -> Context:
    from ml.ingestion.loader import load_opsd_timeseries
    from ml.preprocessing.pipeline import run_preprocessing_pipeline

    logger.info("[PIPELINE] ── PREPROCESS")
    if df is None or checksum is None:
        df, checksum = load_opsd_timeseries(data_path=_data_path(args))
    res = run_preprocessing_pipeline(df, save=True)

    ctx = Context(
        train={"daily": res["train_daily"], "weekly": res["train_weekly"]},
        vaults={"daily": HoldoutVault(res["test_daily"], "test_daily"),
                "weekly": HoldoutVault(res["test_weekly"], "test_weekly")},
        splits={"daily": res["split_daily"], "weekly": res["split_weekly"]},
        df_daily_train=res["train_daily"],
        data_info={
            "raw_file": _data_path(args).name,
            "raw_sha256": checksum,
            "raw_rows": int(len(df)),
            "raw_start_utc": str(df["utc_timestamp"].min()),
            "raw_end_utc": str(df["utc_timestamp"].max()),
            "n_daily_rows": int(len(res["df_daily"])),
            "n_weekly_rows": int(len(res["df_weekly"])),
            "dataset_version": DATASET_VERSION,
        },
        cleaning_log=res["cleaning_log"],
    )
    for g in ("daily", "weekly"):
        _update_evidence(g, "partition", {
            "train_range": f"{ctx.splits[g].train_start.date()}..{ctx.splits[g].train_end.date()}",
            "test_range": f"{ctx.splits[g].test_start.date()}..{ctx.splits[g].test_end.date()}",
            "n_train": int(ctx.splits[g].n_train),
            "n_test": int(ctx.splits[g].n_test),
            "test_fingerprint_sha256_at_creation": ctx.vaults[g].fingerprint_at_creation,
            "imputed_hours": ctx.cleaning_log.get("imputed_timestamps", []),
        })
    return ctx


def task_eda(args, ctx: Context):
    """EDA on the TRAINING partition only (EDA-driven modelling choices must not see the test set)."""
    from ml.eda.analysis import run_eda_analysis
    from ml.eda.plots import plot_all_eda

    logger.info("[PIPELINE] ── EDA (training partition only)")
    eda = run_eda_analysis(ctx.df_daily_train)
    plot_all_eda(eda, save=True)
    out_dir = REPORTS_DIR / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in eda.items() if k != "_eda_df"}
    report["computed_on"] = "training partition only"
    with open(out_dir / "eda_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    _write_eda_markdown(report, out_dir)
    return eda


def _write_eda_markdown(r: Dict, out_dir: Path) -> None:
    dr, bs, miss = r.get("date_range", {}), r.get("basic_stats", {}), r.get("missingness", {})
    lines = [
        "# EDA Report — Germany electricity demand (TRAINING partition only)\n\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n",
        "## Coverage\n",
        f"- {dr.get('start')} → {dr.get('end')} ({dr.get('n_records')} daily records, {dr.get('n_years', 0):.2f} years)\n\n",
        "## Daily demand (MW, mean load)\n",
        f"- mean {bs.get('mean_mw', 0):,.0f} | std {bs.get('std_mw', 0):,.0f} | min {bs.get('min_mw', 0):,.0f} | "
        f"median {bs.get('median_mw', 0):,.0f} | max {bs.get('max_mw', 0):,.0f}\n\n",
        "## Seasonality\n",
        f"- yearly seasonality identifiable (>=2 years): {r.get('yearly_seasonality_sufficient')}\n",
        "- weekly (day-of-week) and monthly patterns: see `day_of_week` / `monthly` in eda_report.json\n\n",
        f"## Missingness\n- null daily records: {miss.get('n_null_daily')} ({miss.get('null_pct_daily')}%)\n\n",
        "Figures: `reports/figures/eda_*.png`. The hold-out test partition was not read.\n",
    ]
    (out_dir / "eda_report.md").write_text("".join(lines), encoding="utf-8")


def task_tune(args, ctx: Context):
    from ml.validation_models.tuning import run_tuning

    for g in _grans(args):
        logger.info("[PIPELINE] ── TUNE (%s) — training partition only", g)
        res = run_tuning(
            ctx.train[g], granularity=g,
            holdout_start=ctx.splits[g].test_start,    # enforced by AccessAuditor
            save=True,
        )
        ctx.tuning[g] = res
        _update_evidence(g, "tuning_access_audit", res["access_audit"])
    return ctx


def _load_tuning_summary(g: str) -> Optional[Dict[str, Any]]:
    p = ARTIFACT_METADATA_DIR / f"tuning_{g}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def task_train(args, ctx: Context):
    from ml.prophet.daily import train_daily_prophet, save_model as save_daily
    from ml.prophet.weekly import train_weekly_prophet, save_model as save_weekly
    from ml.prophet.artifacts import enrich_training_meta

    for g in _grans(args):
        logger.info("[PIPELINE] ── TRAIN FINAL (%s) — training partition only", g)
        if g in ctx.tuning:
            summary = ctx.tuning[g]["summary"]
        else:
            summary = _load_tuning_summary(g)
        if summary is None:
            raise RuntimeError(f"No tuning result for '{g}'. Run --task tune first.")
        params = summary["best_params"]
        trainer, saver = ((train_weekly_prophet, save_weekly) if g == "weekly"
                          else (train_daily_prophet, save_daily))
        with AccessAuditor(holdout_start=ctx.splits[g].test_start, label=f"final-train-{g}") as aud:
            model, meta = trainer(ctx.train[g], params=params)
        meta = enrich_training_meta(meta, model, summary, ctx.data_info, ctx.splits[g])
        meta["model_id"] = _model_id(g)
        meta["model_version"] = _version(g)
        path = saver(model, meta, model_id=_model_id(g))
        ctx.trained[g] = {"model": model, "meta": meta, "path": path}
        _update_evidence(g, "final_training_access_audit", aud.summary())
    return ctx


def _evaluate_one(args, ctx: Context, g: str, stage: str = "final_evaluation"):
    from ml.prophet.daily import load_model as load_daily, generate_forecast
    from ml.prophet.weekly import load_model as load_weekly, generate_weekly_forecast
    from ml.prophet.components import extract_components
    from ml.baselines.evaluate import run_baseline_evaluation
    from ml.evaluation.evaluation import run_final_evaluation
    from ml.evaluation.plots import plot_final_evaluation
    from ml.anomaly.detector import detect_anomalies

    model_path = MODELS_DIR / f"{_model_id(g)}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run --task train first.")
    meta = json.loads(model_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    model = (load_weekly if g == "weekly" else load_daily)(model_path)   # evaluate the SAVED artifact

    # ── the one and only place the hold-out is opened ─────────────────────
    test_df = ctx.vaults[g].open(stage)
    train_df = ctx.train[g]
    split = ctx.splits[g]

    forecast = (generate_weekly_forecast if g == "weekly" else generate_forecast)(model, test_df)
    baselines = run_baseline_evaluation(train_df, test_df, granularity=g, save=True)
    result = run_final_evaluation(
        test_df=test_df, forecast_df=forecast,
        naive_preds=baselines["naive_preds"], snaive_preds=baselines["snaive_preds"],
        training_meta=meta, split_info=split, granularity=g, save=True,
    )
    plot_final_evaluation(train_df, test_df, result["final_preds"], split, granularity=g, save=True)
    extract_components(model, forecast, granularity=g, save=True)

    sens = float(getattr(args, "sensitivity", ANOMALY_SENSITIVITY))
    anomalies = detect_anomalies(test_df, forecast, sensitivity=sens, granularity=g, save=True)

    vault_summary = ctx.vaults[g].summary()
    _update_evidence(g, "holdout_vault", vault_summary)
    if not vault_summary["fingerprint_unchanged"]:
        raise RuntimeError("[LEAKAGE] Hold-out fingerprint changed during the run!")
    return result, anomalies


def task_evaluate(args, ctx: Context):
    from ml.prophet.artifacts import write_manifest

    out = {}
    for g in _grans(args):
        logger.info("[PIPELINE] ── EVALUATE (%s) — hold-out opened once", g)
        out[g] = _evaluate_one(args, ctx, g)
    available = [g for g in ("daily", "weekly") if (MODELS_DIR / f"{_model_id(g)}.json").exists()]
    write_manifest(available)
    return out


def task_baseline(args, ctx: Context):
    from ml.baselines.evaluate import run_baseline_evaluation
    for g in _grans(args):
        test_df = ctx.vaults[g].open("baseline_evaluation")
        run_baseline_evaluation(ctx.train[g], test_df, granularity=g, save=True)
    return ctx


def task_anomalies(args, ctx: Context):
    out = {}
    for g in _grans(args):
        _, an = _evaluate_one(args, ctx, g)
        out[g] = an
    return out


def task_audit(args, ctx: Optional[Context] = None):
    from ml.audit import run_audit
    return run_audit(replay=not getattr(args, "no_replay", False))


def task_all(args):
    t0 = time.time()
    logger.info("[PIPELINE] FULL PIPELINE STARTED")
    df, checksum, _ = task_profile(args)
    task_validate(args, df=df)
    ctx = task_preprocess(args, df=df, checksum=checksum)
    task_eda(args, ctx)
    task_tune(args, ctx)
    task_train(args, ctx)
    task_evaluate(args, ctx)
    logger.info("[PIPELINE] COMPLETE in %.1fs. Reports: %s", time.time() - t0, REPORTS_DIR)
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description="Energy Demand Forecasting ML Pipeline",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--task", default="all",
                   choices=["profile", "validate", "preprocess", "eda", "tune", "train", "evaluate",
                            "baseline", "anomalies", "audit", "all"])
    p.add_argument("--data", default=None, help="Path to OPSD 60-min CSV (default: config)")
    p.add_argument("--granularity", choices=["daily", "weekly", "both"], default="both")
    p.add_argument("--sensitivity", type=float, default=ANOMALY_SENSITIVITY,
                   help=f"Anomaly bound multiplier (default {ANOMALY_SENSITIVITY} = raw 95%% interval)")
    p.add_argument("--no-replay", action="store_true", help="audit: skip the counterfactual replay")
    args = p.parse_args(argv)
    make_output_dirs()

    if args.task == "all":
        task_all(args)
    elif args.task == "profile":
        task_profile(args)
    elif args.task == "validate":
        task_validate(args)
    elif args.task == "audit":
        task_audit(args)
    else:
        # Later stages need the (deterministic) preprocessing context
        ctx = task_preprocess(args)
        {"preprocess": lambda: ctx,
         "eda": lambda: task_eda(args, ctx),
         "tune": lambda: task_tune(args, ctx),
         "train": lambda: task_train(args, ctx),
         "evaluate": lambda: task_evaluate(args, ctx),
         "baseline": lambda: task_baseline(args, ctx),
         "anomalies": lambda: task_anomalies(args, ctx)}[args.task]()


if __name__ == "__main__":
    main()
