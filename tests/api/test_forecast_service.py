"""
tests/api/test_forecast_service.py
──────────────────────────────────
Direct unit tests for the `generate_forecast` service function.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from api.core.exceptions import ForecastError, ModelNotAvailableError, ValidationError
from api.services.forecast import MAX_HORIZON_DAYS, generate_forecast


@pytest.fixture()
def fake_registry():
    registry = MagicMock()
    model = MagicMock()

    def _predict(future_df: pd.DataFrame) -> pd.DataFrame:
        df = future_df.copy()
        df["yhat"] = [65000.0] * len(df)
        df["yhat_lower"] = [62000.0] * len(df)
        df["yhat_upper"] = [68000.0] * len(df)
        return df

    model.predict.side_effect = _predict
    registry.get_model.return_value = model
    registry.get_model_info.return_value = {"model_version": "v2"}
    registry.is_available.return_value = True
    return registry


class TestGenerateForecastService:
    def test_inverted_date_range_raises_validation_error(self):
        with pytest.raises(ValidationError, match="must be before or equal to end_date"):
            generate_forecast("daily", date(2026, 10, 20), date(2026, 10, 15))

    def test_exorbitant_horizon_raises_validation_error(self):
        with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
            generate_forecast("daily", date(2020, 1, 1), date(2028, 1, 1))

    def test_daily_forecast_generation(self, fake_registry):
        with patch("api.services.forecast.get_registry", return_value=fake_registry):
            results = generate_forecast("daily", date(2026, 10, 1), date(2026, 10, 5))
            assert len(results) == 5
            for r in results:
                assert r["granularity"] == "daily"
                assert r["model_version"] == "v2"
                assert r["forecast"] == 65000.0
                assert r["lower_bound"] == 62000.0
                assert r["upper_bound"] == 68000.0

    def test_weekly_forecast_generation(self, fake_registry):
        with patch("api.services.forecast.get_registry", return_value=fake_registry):
            results = generate_forecast("weekly", date(2026, 10, 5), date(2026, 10, 19))
            assert len(results) >= 2
            for r in results:
                assert r["granularity"] == "weekly"

    def test_model_not_available_propagates(self):
        registry = MagicMock()
        registry.get_model.side_effect = ModelNotAvailableError("Model offline")
        with patch("api.services.forecast.get_registry", return_value=registry):
            with pytest.raises(ModelNotAvailableError, match="Model offline"):
                generate_forecast("daily", date(2026, 10, 1), date(2026, 10, 2))

    def test_inference_exception_raises_forecast_error(self, fake_registry):
        broken_model = MagicMock()
        broken_model.predict.side_effect = RuntimeError("Prophet internal crash")
        fake_registry.get_model.return_value = broken_model

        with patch("api.services.forecast.get_registry", return_value=fake_registry):
            with pytest.raises(ForecastError, match="Forecasting computation failed"):
                generate_forecast("daily", date(2026, 10, 1), date(2026, 10, 2))

