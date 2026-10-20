"""
ml/validation_models/guard.py
──────────────────────────────
Hold-out protection and data-access auditing.

Two independent mechanisms:

1. HoldoutVault
   Wraps the 30% test partition. While LOCKED (all of model development: EDA,
   cross-validation, tuning, final fitting) any attempt to read it raises
   HoldoutAccessError and is logged. It can be opened exactly for the stages in
   ALLOWED_STAGES. A SHA-256 fingerprint of the test data is taken at creation
   and re-checked at the end, so silent mutation is detectable too.

2. AccessAuditor
   A context manager that wraps Prophet.fit / Prophet.predict and records the
   [min, max] of the `ds` column of every dataframe the model sees. With a
   `holdout_start` it raises LeakageError the moment any row at/after the
   boundary reaches the model. This is what turns "we did not use test data
   in CV" from a claim into an observed, testable fact.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)

ALLOWED_STAGES = ("final_evaluation", "baseline_evaluation")


class LeakageError(RuntimeError):
    """Raised when hold-out data would influence model development."""


class HoldoutAccessError(LeakageError):
    """Raised when the locked test partition is read outside an allowed stage."""


def fingerprint_frame(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 over (ds, y) of a frame."""
    h = hashlib.sha256()
    ds = pd.to_datetime(df["ds"]).astype("int64").to_numpy()
    h.update(ds.tobytes())
    if "y" in df.columns:
        h.update(df["y"].to_numpy(dtype="float64").tobytes())
    return h.hexdigest()


class HoldoutVault:
    """Locked container for the hold-out test partition."""

    def __init__(self, test_df: pd.DataFrame, name: str = "test"):
        self.name = name
        self._df = test_df.copy().reset_index(drop=True)
        self.start = pd.Timestamp(self._df["ds"].min())
        self.end = pd.Timestamp(self._df["ds"].max())
        self.n_rows = len(self._df)
        self.fingerprint_at_creation = fingerprint_frame(self._df)
        self.access_log: List[Dict[str, Any]] = []

    def _log(self, stage: str, granted: bool) -> None:
        self.access_log.append({
            "vault": self.name,
            "stage": stage,
            "granted": granted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def open(self, stage: str) -> pd.DataFrame:
        """Return a COPY of the test data for an allowed stage; otherwise raise."""
        if stage not in ALLOWED_STAGES:
            self._log(stage, False)
            raise HoldoutAccessError(
                f"[LEAKAGE] Stage '{stage}' attempted to read the hold-out test set. "
                f"Allowed stages: {ALLOWED_STAGES}."
            )
        self._log(stage, True)
        return self._df.copy()

    def verify_unchanged(self) -> bool:
        return fingerprint_frame(self._df) == self.fingerprint_at_creation

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "test_start": str(self.start.date()),
            "test_end": str(self.end.date()),
            "n_rows": self.n_rows,
            "fingerprint_sha256": self.fingerprint_at_creation,
            "fingerprint_unchanged": self.verify_unchanged(),
            "access_log": list(self.access_log),
            "denied_accesses": sum(1 for a in self.access_log if not a["granted"]),
        }


class AccessAuditor:
    """
    Record (and optionally police) every dataframe that reaches Prophet.fit/predict.

    Usage
    -----
        with AccessAuditor(holdout_start=split.test_start) as aud:
            tune(...)
        aud.summary()  # max_ds_accessed < holdout_start  -> no test rows touched
    """

    def __init__(self, holdout_start: Optional[pd.Timestamp] = None, strict: bool = True,
                 label: str = ""):
        self.holdout_start = pd.Timestamp(holdout_start) if holdout_start is not None else None
        self.strict = strict
        self.label = label
        self.records: List[Dict[str, Any]] = []
        self._orig_fit = None
        self._orig_predict = None

    # -- recording ---------------------------------------------------------
    def _record(self, op: str, df: Optional[pd.DataFrame]) -> None:
        if df is None or "ds" not in df.columns or len(df) == 0:
            return
        ds = pd.to_datetime(df["ds"])
        rec = {"op": op, "min_ds": ds.min(), "max_ds": ds.max(), "n_rows": int(len(df))}
        self.records.append(rec)
        if self.holdout_start is not None and rec["max_ds"] >= self.holdout_start:
            msg = (f"[LEAKAGE] Prophet.{op} received rows up to {rec['max_ds']} which is >= "
                   f"hold-out start {self.holdout_start}.")
            if self.strict:
                raise LeakageError(msg)
            logger.error(msg)

    def __enter__(self) -> "AccessAuditor":
        auditor = self
        self._orig_fit = Prophet.fit
        self._orig_predict = Prophet.predict

        def fit(model, df, *args, **kwargs):
            auditor._record("fit", df)
            return auditor._orig_fit(model, df, *args, **kwargs)

        def predict(model, df=None, *args, **kwargs):
            auditor._record("predict", df)
            return auditor._orig_predict(model, df, *args, **kwargs)

        Prophet.fit = fit
        Prophet.predict = predict
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        Prophet.fit = self._orig_fit
        Prophet.predict = self._orig_predict
        return False

    # -- reporting ---------------------------------------------------------
    def max_ds_accessed(self) -> Optional[pd.Timestamp]:
        return max((r["max_ds"] for r in self.records), default=None)

    def summary(self) -> Dict[str, Any]:
        mx = self.max_ds_accessed()
        n_viol = 0
        if self.holdout_start is not None:
            n_viol = sum(1 for r in self.records if r["max_ds"] >= self.holdout_start)
        return {
            "label": self.label,
            "n_fit_calls": sum(1 for r in self.records if r["op"] == "fit"),
            "n_predict_calls": sum(1 for r in self.records if r["op"] == "predict"),
            "min_ds_accessed": str(min((r["min_ds"] for r in self.records), default=None)),
            "max_ds_accessed": str(mx),
            "holdout_start": str(self.holdout_start) if self.holdout_start is not None else None,
            "n_calls_touching_holdout": n_viol,
            "no_holdout_rows_accessed": bool(
                mx is not None and (self.holdout_start is None or mx < self.holdout_start)
            ),
        }


def assert_partition_before(df: pd.DataFrame, boundary: pd.Timestamp, what: str) -> None:
    """Raise LeakageError unless every `ds` in df is strictly before boundary."""
    if df["ds"].max() >= pd.Timestamp(boundary):
        raise LeakageError(
            f"[LEAKAGE] {what}: max ds {df['ds'].max()} >= hold-out boundary {boundary}."
        )
