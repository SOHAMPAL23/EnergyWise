"""
ml/experiments/tracker.py
──────────────────────────
Experiment Tracking — one record per CV fold AND one per configuration summary.

Required fields (per spec):
    experiment_id, model_version, granularity, parameters,
    training_range, validation_range, MAE, RMSE, timestamp

Additional context fields: run_id, stage, config_id, fold_index, n_train, n_val,
interval_coverage, dataset_version, region, seed.

The log is APPEND-ONLY across runs (run_id separates them). Only CV/tuning
experiments are written here; hold-out test metrics are NEVER stored in this
file, so it can be audited as "validation-only".
"""

import csv
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.configs.config import (
    EXPERIMENTS_REPORT_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

EXPERIMENT_LOG_CSV = EXPERIMENTS_REPORT_DIR / "experiments.csv"
EXPERIMENT_LOG_JSON = EXPERIMENTS_REPORT_DIR / "experiments.json"

FIELDS = [
    "experiment_id", "run_id", "stage", "config_id", "fold_index",
    "model", "model_version", "granularity", "parameters",
    "training_range", "validation_range", "n_train", "n_val",
    "MAE", "RMSE", "interval_coverage",
    "dataset_version", "region", "seed", "timestamp",
]


def new_run_id(granularity: str) -> str:
    return f"{granularity}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def config_id(params: Dict[str, Any]) -> str:
    """Stable short hash of a parameter dict (order independent)."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return "cfg-" + hashlib.sha1(blob).hexdigest()[:8]


class ExperimentTracker:
    """Buffers experiment records and flushes them to CSV + JSON."""

    def __init__(self, csv_path: Path = None, json_path: Path = None):
        self.csv_path = Path(csv_path or EXPERIMENT_LOG_CSV)
        self.json_path = Path(json_path or EXPERIMENT_LOG_JSON)
        self.records: List[Dict[str, Any]] = []

    def log(self, **kw) -> Dict[str, Any]:
        missing = [k for k in ("experiment_id", "model_version", "granularity", "parameters",
                               "training_range", "validation_range", "MAE", "RMSE") if k not in kw]
        if missing:
            raise ValueError(f"Experiment record is missing required fields: {missing}")
        rec = {k: kw.get(k) for k in FIELDS}
        rec["model"] = kw.get("model", "prophet")
        rec["parameters"] = json.dumps(kw["parameters"], sort_keys=True, default=str)
        rec["MAE"] = round(float(kw["MAE"]), 4)
        rec["RMSE"] = round(float(kw["RMSE"]), 4)
        rec["timestamp"] = kw.get("timestamp") or datetime.now(timezone.utc).isoformat()
        self.records.append(rec)
        return rec

    def flush(self) -> None:
        """Append buffered records to CSV, then rewrite the JSON mirror from the CSV."""
        make_output_dirs()
        if self.records:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if not exists:
                    w.writeheader()
                w.writerows(self.records)
            logger.info("[TRACKER] Appended %d experiment record(s) -> %s",
                        len(self.records), self.csv_path)
            self.records = []
        if self.csv_path.exists():
            with open(self.csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            with open(self.json_path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1)


def load_experiments(csv_path: Path = None) -> List[Dict[str, str]]:
    path = Path(csv_path or EXPERIMENT_LOG_CSV)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
