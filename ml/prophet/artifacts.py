"""
ml/prophet/artifacts.py
────────────────────────
Model metadata enrichment and the artifact manifest consumed by the backend.

The model's `.meta.json` contains ONLY information available at training time (no
hold-out metrics), so it can never be an avenue of test-set leakage. Hold-out metrics
live in final_metrics_{granularity}.json and are referenced from manifest.json.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ml.configs.config import (
    ARTIFACTS_DIR,
    DATASET_VERSION,
    MODELS_DIR,
    METRICS_DIR,
    FORECASTS_DIR,
    ARTIFACT_METADATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    REGION,
    TARGET_COL_RAW,
    TARGET_UNIT_RAW,
    make_output_dirs,
)

MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
SCHEMA_VERSION = "1.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def environment_info() -> Dict[str, Any]:
    import cmdstanpy, holidays, numpy, prophet
    status = _git("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "prophet": prophet.__version__,
        "cmdstanpy": cmdstanpy.__version__,
        "holidays": holidays.__version__,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_working_tree_dirty": bool(status) if status is not None else None,
    }


def enrich_training_meta(
    meta: Dict[str, Any],
    model,
    tuning_summary: Optional[Dict[str, Any]],
    data_info: Dict[str, Any],
    split_info,
) -> Dict[str, Any]:
    """Return a copy of `meta` with everything a consumer / auditor needs."""
    out = dict(meta)
    cps = [str(c.date()) for c in model.changepoints]
    holiday_names = sorted(map(str, model.train_holiday_names)) if model.train_holiday_names is not None else []
    out.update({
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_SEED,
        "dataset": {
            "dataset_version": DATASET_VERSION,
            "region": REGION,
            "target_column": TARGET_COL_RAW,
            "unit": TARGET_UNIT_RAW + " (mean load over the period)",
            **data_info,
        },
        "split": {
            "train_ratio": 0.70,
            "train_start": str(split_info.train_start.date()),
            "train_end": str(split_info.train_end.date()),
            "test_start": str(split_info.test_start.date()),
            "test_end": str(split_info.test_end.date()),
            "n_train": int(split_info.n_train),
            "n_test": int(split_info.n_test),
        },
        "holidays": {
            "country": "DE",
            "source": "prophet add_country_holidays (python-holidays), national-level only",
            "n_holiday_names": len(holiday_names),
            "names": holiday_names,
        },
        "changepoints": {
            "n_potential": len(cps),
            "first": cps[0] if cps else None,
            "last": cps[-1] if cps else None,
            "all_within_training_range": all(
                split_info.train_start <= pd.Timestamp(c) <= split_info.train_end for c in cps),
        },
        "uncertainty": {
            "interval_width": out.get("parameters", {}).get("interval_width"),
            "columns": ["yhat_lower", "yhat_upper"],
            "method": "Prophet MAP fit; trend-uncertainty + observation noise via Monte-Carlo samples",
        },
        "tuning": ({
            "run_id": tuning_summary["run_id"],
            "selection_metric": tuning_summary["selection_metric"],
            "best_config_id": tuning_summary["best_config_id"],
            "best_cv": tuning_summary["best_cv"],
            "n_configs": tuning_summary["n_configs"],
            "n_folds": tuning_summary["cv_spec"]["n_folds"],
            "test_data_used": tuning_summary["test_data_used"],
            "summary_file": f"metadata/tuning_{meta['granularity']}.json",
        } if tuning_summary else None),
        "hold_out_used_in_development": False,
        "environment": environment_info(),
    })
    out.pop("holiday_metadata", None)   # replaced by the verified `holidays` block
    return out


def write_manifest(granularities: List[str], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Hash every artifact the backend may consume and write manifest.json."""
    make_output_dirs()
    entries: Dict[str, Any] = {}

    def add(kind: str, path: Path):
        if path.exists():
            entries.setdefault(kind, []).append({
                "path": str(path.relative_to(ARTIFACTS_DIR)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })

    from ml.configs.config import MODEL_VERSION_DAILY, MODEL_VERSION_WEEKLY
    summary: Dict[str, Any] = {}
    for g in granularities:
        v = MODEL_VERSION_WEEKLY if g == "weekly" else MODEL_VERSION_DAILY
        mid = f"prophet-{g}-{v}"
        add("models", MODELS_DIR / f"{mid}.json")
        add("model_metadata", MODELS_DIR / f"{mid}.meta.json")
        add("tuning", ARTIFACT_METADATA_DIR / f"tuning_{g}.json")
        add("metrics", METRICS_DIR / f"final_metrics_{g}.json")
        add("metrics", METRICS_DIR / f"anomaly_summary_{g}.json")
        add("forecasts", FORECASTS_DIR / f"final_predictions_{g}.csv")
        add("forecasts", FORECASTS_DIR / f"anomaly_detections_{g}.csv")
        add("forecasts", FORECASTS_DIR / f"anomalies_{g}.json")
        fm = METRICS_DIR / f"final_metrics_{g}.json"
        if fm.exists():
            m = json.loads(fm.read_text(encoding="utf-8"))
            summary[g] = {
                "model_id": mid,
                "model_version": v,
                "test_MAE_MW": {k: m["metrics"][k]["MAE_MW"] for k in m["metrics"]},
                "test_RMSE_MW": {k: m["metrics"][k]["RMSE_MW"] for k in m["metrics"]},
            }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "granularities": granularities,
        "contract": {
            "timestamp_column": "ds (UTC, tz-naive, daily=00:00 day start, weekly=Monday start)",
            "target_unit": "MW (mean load over the period)",
            "forecast_columns": ["yhat", "yhat_lower", "yhat_upper"],
            "interval_width": 0.95,
            "load_model": "prophet.serialize.model_from_json (prophet==" + environment_info()["prophet"] + ")",
            "anomaly_classes": ["NORMAL", "DROP_ANOMALY", "SPIKE_ANOMALY"],
            "anomaly_semantics": "investigation signals, NOT ground truth",
            "ignore_versions": "any artifact whose model_version is not listed under `summary` is stale",
        },
        "summary": summary,
        "artifacts": entries,
        "environment": environment_info(),
    }
    if extra:
        manifest.update(extra)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
