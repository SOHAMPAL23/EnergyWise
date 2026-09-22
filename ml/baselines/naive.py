"""
ml/baselines/naive.py
──────────────────────
Naive Forecaster

Strategy: Predict the LAST observed training value for the entire forecast horizon.
This is the simplest possible forecaster and serves as a lower-bound benchmark.

Reference:
  Hyndman, R.J. & Athanasopoulos, G. (2021) Forecasting: Principles and Practice, 3rd ed.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from ml.configs.config import DS_COL, Y_COL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class NaiveForecaster:
    """
    Naive forecaster that repeats the last training observation.

    Attributes
    ----------
    last_value : float — Last observed demand value from training set.
    """

    def __init__(self):
        self.last_value: float = None
        self._fitted: bool = False

    def fit(self, train_df: pd.DataFrame, y_col: str = Y_COL) -> "NaiveForecaster":
        """
        Fit the naive model by recording the last training value.

        Parameters
        ----------
        train_df : Training DataFrame (Prophet format: ds, y).
        y_col    : Target column name.
        """
        if train_df.empty:
            raise ValueError("[NAIVE] Training DataFrame is empty.")
        valid = train_df[y_col].dropna()
        if valid.empty:
            raise ValueError("[NAIVE] No non-null values in training target column.")
        self.last_value = float(valid.iloc[-1])
        self._fitted = True
        logger.info("[NAIVE] Fitted. Last training value: %.2f MW", self.last_value)
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate naive forecasts for the test horizon.

        Parameters
        ----------
        test_df : Test DataFrame with at least [ds] column.

        Returns
        -------
        DataFrame with [ds, yhat] columns.
        """
        if not self._fitted:
            raise RuntimeError("[NAIVE] Model not fitted. Call fit() first.")

        horizon = len(test_df)
        predictions = np.full(shape=(horizon,), fill_value=self.last_value)

        result = pd.DataFrame({
            DS_COL: test_df[DS_COL].values,
            "yhat": predictions,
        })

        logger.info("[NAIVE] Generated %d predictions (constant = %.2f MW).",
                    horizon, self.last_value)
        return result
