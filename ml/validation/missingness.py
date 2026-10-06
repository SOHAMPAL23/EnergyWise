"""
ml/validation/missingness.py
────────────────────────────
Missing-Value Validation

Checks:
  1. Number and percentage of null demand values
  2. Rows with any null values
  3. Consecutive null run analysis
"""

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from ml.configs.config import TARGET_COL_RAW, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# Warning threshold: if >5% of demand values are null, warn
NULL_WARNING_PCT = 5.0


def validate_missingness(
    df: pd.DataFrame,
    target_col: str = TARGET_COL_RAW,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Validate missing values in the target demand column.

    Parameters
    ----------
    df         : DataFrame to inspect.
    target_col : Demand target column name.
    strict     : If True, raise on >5% nulls.

    Returns
    -------
    dict with status and findings.
    """
    findings = []
    has_error = False

    n_rows = len(df)
    if target_col not in df.columns:
        findings.append({
            "level": "ERROR",
            "check": "target_column_present",
            "message": f"Target column '{target_col}' not found — cannot check missingness.",
        })
        logger.error("[MISSINGNESS][ERROR] %s", findings[-1]["message"])
        return {"status": "FAIL", "findings": findings}

    demand = df[target_col]
    n_null = int(demand.isna().sum())
    null_pct = 100 * n_null / n_rows if n_rows > 0 else 0.0

    # ── Check 1: Count and percentage ────────────────────────────────────
    if n_null == 0:
        findings.append({
            "level": "INFO",
            "check": "null_count",
            "message": f"No null values in '{target_col}'.",
        })
    elif null_pct <= NULL_WARNING_PCT:
        findings.append({
            "level": "WARNING",
            "check": "null_count",
            "message": f"{n_null} null values ({null_pct:.2f}%) in '{target_col}'. "
                       "Will be forward-filled during preprocessing.",
        })
    else:
        findings.append({
            "level": "ERROR",
            "check": "null_count",
            "message": f"{n_null} null values ({null_pct:.2f}%) in '{target_col}'. "
                       "Exceeds 5% threshold — investigate before proceeding.",
        })
        has_error = True

    # ── Check 2: Consecutive null runs ───────────────────────────────────
    null_mask = demand.isna()
    if n_null > 0:
        runs = _consecutive_null_runs(null_mask)
        max_run = max(runs, default=0)
        findings.append({
            "level": "WARNING" if max_run <= 24 else "ERROR",
            "check": "consecutive_nulls",
            "message": f"Longest consecutive null run: {max_run} rows. "
                       f"Total null runs: {len(runs)}.",
        })
        if max_run > 24:
            has_error = True

    # ── Emit logs ─────────────────────────────────────────────────────────
    for f in findings:
        level = f["level"]
        msg = f"[MISSINGNESS][{level}] {f['check']}: {f['message']}"
        if level == "ERROR":
            logger.error(msg)
        elif level == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

    status = "FAIL" if has_error else "PASS"
    if has_error and strict:
        errors = [f["message"] for f in findings if f["level"] == "ERROR"]
        raise ValueError(f"[MISSINGNESS VALIDATION FAILED] {'; '.join(errors)}")

    return {
        "status": status,
        "n_null": n_null,
        "null_pct": round(null_pct, 4),
        "findings": findings,
    }


def _consecutive_null_runs(null_mask: pd.Series):
    """Return list of lengths of consecutive null runs."""
    runs = []
    count = 0
    for v in null_mask:
        if v:
            count += 1
        else:
            if count > 0:
                runs.append(count)
                count = 0
    if count > 0:
        runs.append(count)
    return runs
