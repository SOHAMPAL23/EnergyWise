"""
ml/validation/duplicates.py
────────────────────────────
Duplicate Validation

Checks:
  1. Duplicate rows (all columns identical)
  2. Duplicate timestamps (same timestamp, possibly different demand values)

Reports every finding — never silently drops data.
"""

import logging
from typing import Dict, Any

import pandas as pd

from ml.configs.config import TIMESTAMP_COL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def validate_duplicates(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Detect duplicate rows and duplicate timestamps.

    Parameters
    ----------
    df            : DataFrame to inspect.
    timestamp_col : Timestamp column to check for duplicates.
    strict        : If True, raise ValueError on duplicate timestamps.

    Returns
    -------
    dict with status and findings.
    """
    findings = []
    has_error = False

    # ── Check 1: Duplicate rows ───────────────────────────────────────────
    n_dup_rows = int(df.duplicated().sum())
    if n_dup_rows == 0:
        findings.append({
            "level": "INFO",
            "check": "duplicate_rows",
            "message": "No duplicate rows found.",
        })
    else:
        findings.append({
            "level": "WARNING",
            "check": "duplicate_rows",
            "message": f"{n_dup_rows} duplicate row(s) detected. "
                       "These will be dropped (keeping first) during preprocessing.",
        })

    # ── Check 2: Duplicate timestamps ────────────────────────────────────
    if timestamp_col in df.columns:
        n_dup_ts = int(df[timestamp_col].duplicated().sum())
        if n_dup_ts == 0:
            findings.append({
                "level": "INFO",
                "check": "duplicate_timestamps",
                "message": "No duplicate timestamps found.",
            })
        else:
            # Show example duplicates
            dup_mask = df[timestamp_col].duplicated(keep=False)
            dup_examples = df[dup_mask][timestamp_col].head(5).tolist()
            findings.append({
                "level": "ERROR",
                "check": "duplicate_timestamps",
                "message": f"{n_dup_ts} duplicate timestamp(s) detected. "
                           f"Examples: {dup_examples}. "
                           "Duplicate timestamps must be resolved before splitting.",
            })
            has_error = True
    else:
        findings.append({
            "level": "WARNING",
            "check": "duplicate_timestamps",
            "message": f"Timestamp column '{timestamp_col}' not found — cannot check timestamp duplicates.",
        })

    # ── Emit logs ─────────────────────────────────────────────────────────
    for f in findings:
        level = f["level"]
        msg = f"[DUPLICATES][{level}] {f['check']}: {f['message']}"
        if level == "ERROR":
            logger.error(msg)
        elif level == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

    status = "FAIL" if has_error else "PASS"
    if has_error and strict:
        errors = [f["message"] for f in findings if f["level"] == "ERROR"]
        raise ValueError(f"[DUPLICATES VALIDATION FAILED] {'; '.join(errors)}")

    return {
        "status": status,
        "n_duplicate_rows": n_dup_rows,
        "n_duplicate_timestamps": n_dup_ts if timestamp_col in df.columns else "unknown",
        "findings": findings,
    }
