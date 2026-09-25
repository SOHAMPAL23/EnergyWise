"""
ml/baselines/seasonal_naive.py
────────────────────────────────
Seasonal-Naive Forecaster

Strategy: Predict using the same value from the corresponding day/week in the
previous seasonal cycle.

For daily data with weekly seasonality (period=7):
  Forecast for day t = actual demand on day (t - 7) from training data.

For weekly data with annual seasonality (period=52):
  Forecast for week t = actual demand from week (t - 52) in training data.

The seasonal period is configurable and must be justified based on the
actual forecasting granularity used.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml.configs.config import (
    DS_COL,
    Y_COL,
    SEASONAL_PERIOD_DAILY,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class SeasonalNaiveForecaster:
    """
    Seasonal-naive forecaster using a rolling seasonal lag.

    For daily energy demand (period=7):
      The 7-day weekly cycle is the dominant seasonality — weekday/weekend
      patterns are strong in European grid data. Each forecast day uses the
      actual demand from the same weekday in the most recent complete week.

    For weekly energy demand (period=52):
      Each forecast week uses the demand from the same week number in the
      prior year (52-week cycle).

    Parameters
    ----------
    period : int — Seasonal period in steps. Default: 7 (weekly cycle for daily data).
    """

    def __init__(self, period: int = SEASONAL_PERIOD_DAILY):
        if period < 1:
            raise ValueError(f"[SEASONAL-NAIVE] Period must be ≥ 1. Got: {period}")
        self.period = period
        self._train_values: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, train_df: pd.DataFrame, y_col: str = Y_COL) -> "SeasonalNaiveForecaster":
        """
        Fit by storing the last `period` training values.

        Parameters
        ----------
        train_df : Training DataFrame (Prophet format: ds, y).
        y_col    : Target column.
        """
        if train_df.empty:
            raise ValueError("[SEASONAL-NAIVE] Training DataFrame is empty.")
        if len(train_df) < self.period:
            raise ValueError(
                f"[SEASONAL-NAIVE] Training set has {len(train_df)} rows "
                f"but period={self.period} requires at least {self.period} rows."
            )

        self._train_values = train_df[y_col].values.copy()
        self._fitted = True
        logger.info(
            "[SEASONAL-NAIVE] Fitted. Period=%d. Last %d training values stored.",
            self.period, len(self._train_values),
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate seasonal-naive forecasts.

        For each forecast step i (0-indexed):
          yhat[i] = train_values[n_train - period + (i % period)]

        This wraps around the last full seasonal cycle in training data.

        Parameters
        ----------
        test_df : Test DataFrame with [ds] column.

        Returns
        -------
        DataFrame with [ds, yhat].
        """
        if not self._fitted:
            raise RuntimeError("[SEASONAL-NAIVE] Model not fitted. Call fit() first.")

        n_train = len(self._train_values)
        horizon = len(test_df)

        # For each forecast step, look back `period` days from the end of training
        predictions = np.array([
            self._train_values[n_train - self.period + (i % self.period)]
            for i in range(horizon)
        ])

        result = pd.DataFrame({
            DS_COL: test_df[DS_COL].values,
            "yhat": predictions,
        })

        logger.info(
            "[SEASONAL-NAIVE] Generated %d predictions (period=%d).",
            horizon, self.period,
        )
        return result
