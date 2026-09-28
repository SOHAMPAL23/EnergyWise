"""
ml/experiments/tracker.py
──────────────────────────
Experiment Tracking

Logs every experiment (CV fold, tuning trial, final model) with full metadata.
All experiments are persisted to CSV and JSON for reproducibility.

Each experiment record contains:
  experiment_id, model, granularity, dataset_version, region,
  training_range, validation_range, hyperparameters, MAE, RMSE, timestamp
"""

import csv
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ml.configs.config import (
    EXPERIMENTS_REPORT_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    make_output_dirs,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

EXPERIMENT_LOG_CSV = EXPERIMENTS_REPORT_DIR / "experiment_results.csv"
EXPERIMENT_LOG_JSON = EXPERIMENTS_REPORT_DIR / "experiment_results.json"

_EXPERIMENTS: list = []  # In-memory store for the current session


def log_experiment(
    experiment_id: str,
    model: str,
    granularity: str,
    dataset_version: str,
    region: str,
    training_range: str,
    validation_range: str,
    hyperparameters: Dict[str, Any],
    mae_val: float,
    rmse_val: float,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log an experiment and persist to disk.

    Parameters
    ----------
    experiment_id    : Unique experiment identifier.
    model            : Model name (e.g., "prophet", "naive").
    granularity      : "daily" or "weekly".
    dataset_version  : OPSD dataset version string.
    region           : Country/region code (e.g., "DE").
    training_range   : Human-readable training range string.
    validation_range : Human-readable validation range string.
    hyperparameters  : Dict of hyperparameter key-value pairs.
    mae_val          : Mean Absolute Error (from actual predictions).
    rmse_val         : Root Mean Squared Error.
    notes            : Optional notes string.

    Returns
    -------
    dict — The experiment record.
    """
    make_output_dirs()

    record = {
        "experiment_id": experiment_id,
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "granularity": granularity,
        "dataset_version": dataset_version,
        "region": region,
        "training_range": training_range,
        "validation_range": validation_range,
        "hyperparameters": json.dumps(hyperparameters, default=str),
        "MAE": round(float(mae_val), 4),
        "RMSE": round(float(rmse_val), 4),
        "notes": notes or "",
    }

    _EXPERIMENTS.append(record)
    _persist_to_csv(record)
    _persist_to_json()

    logger.info("[TRACKER] Logged experiment '%s': MAE=%.4f, RMSE=%.4f",
                experiment_id, mae_val, rmse_val)

    return record


def _persist_to_csv(record: Dict[str, Any]):
    """Append a single experiment record to the CSV log."""
    EXPERIMENT_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = EXPERIMENT_LOG_CSV.exists()
    fieldnames = list(record.keys())
    with open(EXPERIMENT_LOG_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def _persist_to_json():
    """Overwrite the JSON experiment log with the full in-memory list."""
    EXPERIMENT_LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG_JSON, "w", encoding="utf-8") as fh:
        json.dump(_EXPERIMENTS, fh, indent=2, default=str)


def load_experiments() -> list:
    """Load all experiments from the CSV log."""
    if not EXPERIMENT_LOG_CSV.exists():
        return []
    records = []
    with open(EXPERIMENT_LOG_CSV, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))
    return records
