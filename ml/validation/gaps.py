"""
ml/validation/gaps.py
──────────────────────
Time-Gap Validation

Checks for missing time steps in the timestamp series.
A gap is defined as any interval > 1.5× the expected step (mode of differences).

Reports:
  - Total number of gaps
  - Largest gap
  - Gap inventory (up to 20 gaps)
"""

import logging
from typing import Dict, Any, List

import pandas as pd

from ml.configs.config import TIMESTAMP_COL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def validate_gaps(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Detect time gaps in the timestamp series.

    Parameters
    ----------
    df            : DataFrame with a timestamp column.
    timestamp_col : Timestamp column name.
    strict        : If True, raise on gaps > 24 hours.

    Returns
    -------
    dict with status and findings.
    """
    findings = []
    has_error = False

    if timestamp_col not in df.columns:
        findings.append({
            "level": "ERROR",
            "check": "column_exists",
            "message": f"Timestamp column '{timestamp_col}' not found.",
        })
        logger.error("[GAPS][ERROR] %s", findings[-1]["message"])
        return {"status": "FAIL", "findings": findings, "n_gaps": "unknown", "gaps": []}

    ts = df[timestamp_col].sort_values().reset_index(drop=True)

    if len(ts) < 2:
        findings.append({
            "level": "WARNING",
            "check": "sufficient_data",
            "message": "Fewer than 2 timestamps — gap analysis not possible.",
        })
        return {"status": "PASS", "findings": findings, "n_gaps": 0, "gaps": []}

    diffs = ts.diff().dropna()
    expected_step = diffs.mode()[0]
    threshold = expected_step * 1.5

    gap_mask = diffs > threshold
    gap_indices = diffs[gap_mask].index
    n_gaps = len(gap_indices)

    # ── Build gap inventory ────────────────────────────────────────────────
    gaps: List[Dict[str, Any]] = []
    for idx in gap_indices:
        gap_start = ts.iloc[idx - 1]
        gap_end = ts.iloc[idx]
        duration = gap_end - gap_start
        gaps.append({
            "gap_start": str(gap_start),
            "gap_end": str(gap_end),
            "gap_duration_hours": round(duration.total_seconds() / 3600, 2),
        })

    # Sort by duration (largest first)
    gaps.sort(key=lambda g: g["gap_duration_hours"], reverse=True)
    max_gap_hours = gaps[0]["gap_duration_hours"] if gaps else 0.0

    if n_gaps == 0:
        findings.append({
            "level": "INFO",
            "check": "time_gaps",
            "message": f"No time gaps detected. Expected step: {expected_step}.",
        })
    elif max_gap_hours <= 2.0:
        findings.append({
            "level": "INFO",
            "check": "time_gaps",
            "message": f"{n_gaps} minor gap(s) detected. Max gap: {max_gap_hours:.1f}h. "
                       "Within acceptable tolerance for daily resampling.",
        })
    elif max_gap_hours <= 24.0:
        findings.append({
            "level": "WARNING",
            "check": "time_gaps",
            "message": f"{n_gaps} gap(s) detected. Largest: {max_gap_hours:.1f}h. "
                       "Will be handled by daily mean aggregation.",
        })
    else:
        findings.append({
            "level": "ERROR",
            "check": "time_gaps",
            "message": f"{n_gaps} gap(s) detected. Largest: {max_gap_hours:.1f}h. "
                       "Large gaps may affect forecast quality — review gap inventory.",
        })
        has_error = True

    # ── Emit logs ─────────────────────────────────────────────────────────
    for f in findings:
        level = f["level"]
        msg = f"[GAPS][{level}] {f['check']}: {f['message']}"
        if level == "ERROR":
            logger.error(msg)
        elif level == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

    status = "FAIL" if has_error else "PASS"
    if has_error and strict:
        errors = [f["message"] for f in findings if f["level"] == "ERROR"]
        raise ValueError(f"[GAPS VALIDATION FAILED] {'; '.join(errors)}")

    return {
        "status": status,
        "n_gaps": n_gaps,
        "max_gap_hours": max_gap_hours,
        "expected_step": str(expected_step),
        "gaps": gaps[:20],  # cap for readability
        "findings": findings,
    }
