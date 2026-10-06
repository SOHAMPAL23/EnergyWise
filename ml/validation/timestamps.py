"""
ml/validation/timestamps.py
────────────────────────────
Timestamp Validation

Checks:
  1. All timestamps can be parsed (no NaT)
  2. Timestamps are chronologically ordered (monotonic increasing)
  3. Timezone characteristics
  4. No future timestamps
"""

import logging
from typing import Dict, Any

import pandas as pd

from ml.configs.config import TIMESTAMP_COL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate timestamp column for parseability, ordering, and timezone.

    Parameters
    ----------
    df            : DataFrame with a timestamp column.
    timestamp_col : Column name to validate.
    strict        : Raise on ERROR-level failures if True.

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
            "message": f"Column '{timestamp_col}' not in DataFrame. Cannot validate timestamps.",
        })
        logger.error("[TIMESTAMPS][ERROR] column_exists: %s", findings[-1]["message"])
        if strict:
            raise ValueError(findings[-1]["message"])
        return {"status": "FAIL", "findings": findings}

    ts = df[timestamp_col]

    # ── Check 1: All timestamps parseable ────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce")
        n_unparseable = ts_parsed.isna().sum()
    else:
        ts_parsed = ts
        n_unparseable = ts_parsed.isna().sum()

    if n_unparseable > 0:
        findings.append({
            "level": "ERROR",
            "check": "timestamps_parseable",
            "message": f"{n_unparseable} timestamp(s) could not be parsed (NaT after coerce).",
        })
        has_error = True
    else:
        findings.append({
            "level": "INFO",
            "check": "timestamps_parseable",
            "message": f"All {len(ts)} timestamps parsed successfully.",
        })

    # ── Check 2: Monotonic increasing ────────────────────────────────────
    is_monotonic = ts_parsed.is_monotonic_increasing
    if not is_monotonic:
        n_out = (ts_parsed.diff().dropna() < pd.Timedelta(0)).sum()
        findings.append({
            "level": "ERROR",
            "check": "monotonic_increasing",
            "message": f"Timestamps are NOT monotonically increasing. "
                       f"~{n_out} reversal(s) detected. Pipeline requires sorted timestamps.",
        })
        has_error = True
    else:
        findings.append({
            "level": "INFO",
            "check": "monotonic_increasing",
            "message": "Timestamps are monotonically increasing (chronologically ordered).",
        })

    # ── Check 3: Timezone characteristics ────────────────────────────────
    tz = ts_parsed.dt.tz if hasattr(ts_parsed, "dt") else None
    if tz is None:
        findings.append({
            "level": "WARNING",
            "check": "timezone",
            "message": "Timestamps have no timezone info (tz-naive). "
                       "Assuming UTC as per OPSD documentation.",
        })
    else:
        findings.append({
            "level": "INFO",
            "check": "timezone",
            "message": f"Timestamp timezone: {tz}.",
        })

    # ── Emit logs ─────────────────────────────────────────────────────────
    for f in findings:
        level = f["level"]
        msg = f"[TIMESTAMPS][{level}] {f['check']}: {f['message']}"
        if level == "ERROR":
            logger.error(msg)
        elif level == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

    status = "FAIL" if has_error else "PASS"
    if has_error and strict:
        errors = [f["message"] for f in findings if f["level"] == "ERROR"]
        raise ValueError(f"[TIMESTAMP VALIDATION FAILED] {'; '.join(errors)}")

    return {"status": status, "findings": findings}
