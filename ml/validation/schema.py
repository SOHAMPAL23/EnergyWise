"""
ml/validation/schema.py
───────────────────────
Schema and Data-Type Validation

Checks that the loaded raw DataFrame conforms to the expected schema:
  - Required columns exist
  - Timestamp column is datetime-compatible
  - Target column is numeric (float/int)

Reports: ERROR (raises), WARNING (logs), INFO (logs).
Never silently modifies the DataFrame.
"""

import logging
from typing import Dict, Any

import pandas as pd

from ml.configs.config import TIMESTAMP_COL, TARGET_COL_RAW, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def validate_schema(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL_RAW,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate the DataFrame schema against the expected OPSD structure.

    Parameters
    ----------
    df            : Raw or partially processed DataFrame.
    timestamp_col : Expected timestamp column name.
    target_col    : Expected demand target column name.
    strict        : If True, raises ValueError on any ERROR-level finding.

    Returns
    -------
    dict with keys: status ("PASS" | "FAIL"), findings (list of dicts)
    """
    findings = []
    has_error = False

    # ── Check 1: Timestamp column exists ─────────────────────────────────
    if timestamp_col not in df.columns:
        findings.append({
            "level": "ERROR",
            "check": "timestamp_column_exists",
            "message": f"Timestamp column '{timestamp_col}' not found. "
                       f"Available: {list(df.columns)}",
        })
        has_error = True
    else:
        findings.append({
            "level": "INFO",
            "check": "timestamp_column_exists",
            "message": f"Timestamp column '{timestamp_col}' found.",
        })

    # ── Check 2: Target column exists ────────────────────────────────────
    if target_col not in df.columns:
        findings.append({
            "level": "ERROR",
            "check": "target_column_exists",
            "message": f"Target column '{target_col}' not found. "
                       f"Available: {list(df.columns)}",
        })
        has_error = True
    else:
        findings.append({
            "level": "INFO",
            "check": "target_column_exists",
            "message": f"Target column '{target_col}' found.",
        })

    # ── Check 3: Timestamp dtype is datetime-compatible ──────────────────
    if timestamp_col in df.columns:
        dtype = df[timestamp_col].dtype
        is_dt = pd.api.types.is_datetime64_any_dtype(df[timestamp_col])
        if not is_dt:
            findings.append({
                "level": "WARNING",
                "check": "timestamp_dtype",
                "message": f"Timestamp column dtype is '{dtype}' (not datetime). "
                           "Will attempt coercion during preprocessing.",
            })
        else:
            findings.append({
                "level": "INFO",
                "check": "timestamp_dtype",
                "message": f"Timestamp column dtype: {dtype}.",
            })

    # ── Check 4: Target column is numeric ────────────────────────────────
    if target_col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        if not is_numeric:
            findings.append({
                "level": "ERROR",
                "check": "target_dtype",
                "message": f"Target column '{target_col}' is not numeric. "
                           f"Got dtype: {df[target_col].dtype}.",
            })
            has_error = True
        else:
            findings.append({
                "level": "INFO",
                "check": "target_dtype",
                "message": f"Target column '{target_col}' dtype: {df[target_col].dtype}.",
            })

    # ── Check 5: DataFrame is not empty ──────────────────────────────────
    if len(df) == 0:
        findings.append({
            "level": "ERROR",
            "check": "non_empty",
            "message": "DataFrame is empty (0 rows).",
        })
        has_error = True
    else:
        findings.append({
            "level": "INFO",
            "check": "non_empty",
            "message": f"DataFrame has {len(df)} rows.",
        })

    # ── Emit log entries ──────────────────────────────────────────────────
    for f in findings:
        level = f["level"]
        msg = f"[SCHEMA][{level}] {f['check']}: {f['message']}"
        if level == "ERROR":
            logger.error(msg)
        elif level == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

    status = "FAIL" if has_error else "PASS"

    if has_error and strict:
        errors = [f["message"] for f in findings if f["level"] == "ERROR"]
        raise ValueError(f"[SCHEMA VALIDATION FAILED] {'; '.join(errors)}")

    return {"status": status, "findings": findings}
