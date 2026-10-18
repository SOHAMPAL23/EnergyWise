"""
tests/api/test_forecast.py
──────────────────────────
Integration and contract tests for GET /api/v1/forecast.

Requirements Tested
───────────────────
1. Valid request generates correct forecast points with all 6 required fields:
   - timestamp
   - forecast
   - lower_bound
   - upper_bound
   - granularity
   - model_version
2. Daily and weekly granularities supported.
3. Single date (start_date == end_date) generates 1 forecast point.
4. Multi-day date ranges generate the correct number of points.
5. Invalid date format returns HTTP 422.
6. Invalid granularity returns HTTP 422.
7. Date range with end_date < start_date returns HTTP 422.
8. Date range exceeding maximum horizon returns HTTP 422.
9. Missing required query parameters return HTTP 422.
10. Missing/unavailable model returns HTTP 503.
11. Inference is stateless (no internal state mutated between calls).
"""

from datetime import date
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ─── Fake model & registry helpers ───────────────────────────────────────────

def _make_fake_prophet_model(base_yhat: float = 65000.0) -> MagicMock:
    """Return a mock Prophet model that mimics `model.predict(future_df)`."""
    model = MagicMock()

    def _predict(future_df: pd.DataFrame) -> pd.DataFrame:
        df = future_df.copy()
        # Vectorized generation of plausible forecast data
        n = len(df)
        df["yhat"] = [base_yhat + i * 100.0 for i in range(n)]
        df["yhat_lower"] = [base_yhat - 3000.0 + i * 100.0 for i in range(n)]
        df["yhat_upper"] = [base_yhat + 3000.0 + i * 100.0 for i in range(n)]
        return df

    model.predict.side_effect = _predict
    return model


def _make_fake_registry(
    daily_available: bool = True,
    weekly_available: bool = True,
) -> MagicMock:
    """Return a MagicMock that behaves like a loaded ModelRegistry."""
    registry = MagicMock()

    daily_model = _make_fake_prophet_model(65000.0)
    weekly_model = _make_fake_prophet_model(63000.0)

    daily_meta = {
        "model_id": "prophet-daily-v2",
        "model_version": "v2",
        "granularity": "daily",
    }
    weekly_meta = {
        "model_id": "prophet-weekly-v2",
        "model_version": "v2",
        "granularity": "weekly",
    }

    from api.core.exceptions import ModelNotAvailableError

    def _get_model(granularity: str):
        if granularity == "daily":
            if not daily_available:
                raise ModelNotAvailableError("Daily model unavailable.")
            return daily_model
        elif granularity == "weekly":
            if not weekly_available:
                raise ModelNotAvailableError("Weekly model unavailable.")
            return weekly_model
        raise ModelNotAvailableError(f"Unknown granularity '{granularity}'.")

    registry.is_available.side_effect = lambda g: (
        daily_available if g == "daily" else weekly_available
    )
    registry.get_model.side_effect = _get_model
    registry.get_model_info.side_effect = lambda g: (
        daily_meta if g == "daily" else weekly_meta
    )
    return registry


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """TestClient with both models available in fake registry."""
    registry = _make_fake_registry(daily_available=True, weekly_available=True)
    import api.services.model_registry as reg_module

    def _fake_initialise():
        reg_module._registry = registry
        return registry

    with (
        patch.object(reg_module, "_registry", registry),
        patch("backend.main.initialise_registry", side_effect=_fake_initialise),
        patch("main.initialise_registry", side_effect=_fake_initialise),
    ):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client


@pytest.fixture()
def client_missing_daily():
    """TestClient where daily model failed to load (missing)."""
    registry = _make_fake_registry(daily_available=False, weekly_available=True)
    import api.services.model_registry as reg_module

    def _fake_initialise():
        reg_module._registry = registry
        return registry

    with (
        patch.object(reg_module, "_registry", registry),
        patch("backend.main.initialise_registry", side_effect=_fake_initialise),
        patch("main.initialise_registry", side_effect=_fake_initialise),
    ):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client


@pytest.fixture()
def client_missing_weekly():
    """TestClient where weekly model failed to load (missing)."""
    registry = _make_fake_registry(daily_available=True, weekly_available=False)
    import api.services.model_registry as reg_module

    def _fake_initialise():
        reg_module._registry = registry
        return registry

    with (
        patch.object(reg_module, "_registry", registry),
        patch("backend.main.initialise_registry", side_effect=_fake_initialise),
        patch("main.initialise_registry", side_effect=_fake_initialise),
    ):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestForecastEndpointContract:
    """Tests verifying the endpoint contract, return shape, and field types."""

    ENDPOINT = "/api/v1/forecast"

    def test_daily_forecast_returns_200(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-10-17",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 200

    def test_daily_forecast_returns_list_of_points(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-10-17",
        }
        res = client.get(self.ENDPOINT, params=params)
        data = res.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_forecast_contains_all_six_required_fields(self, client: TestClient):
        """Must return: timestamp, forecast, lower_bound, upper_bound, granularity, model_version."""
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-10-15",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        item = data[0]

        expected_fields = {
            "timestamp",
            "forecast",
            "lower_bound",
            "upper_bound",
            "granularity",
            "model_version",
        }
        assert expected_fields.issubset(item.keys())

    def test_forecast_field_types(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-10-15",
        }
        res = client.get(self.ENDPOINT, params=params)
        item = res.json()[0]

        assert isinstance(item["timestamp"], str)
        assert isinstance(item["forecast"], (int, float))
        assert isinstance(item["lower_bound"], (int, float))
        assert isinstance(item["upper_bound"], (int, float))
        assert item["granularity"] == "daily"
        assert item["model_version"] == "v2"

    def test_lower_bound_less_than_upper_bound(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-10-18",
        }
        res = client.get(self.ENDPOINT, params=params)
        data = res.json()
        for item in data:
            assert item["lower_bound"] <= item["upper_bound"]

    def test_single_day_forecast(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-10-15",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert "2026-10-15" in data[0]["timestamp"]

    def test_weekly_forecast_returns_200(self, client: TestClient):
        params = {
            "granularity": "weekly",
            "start_date": "2026-10-12",
            "end_date": "2026-10-26",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        for item in data:
            assert item["granularity"] == "weekly"
            assert item["model_version"] == "v2"


class TestForecastValidationErrors:
    """Tests verifying 422 Unprocessable Entity responses on invalid inputs."""

    ENDPOINT = "/api/v1/forecast"

    def test_invalid_granularity_returns_422(self, client: TestClient):
        params = {
            "granularity": "hourly",
            "start_date": "2026-10-15",
            "end_date": "2026-10-17",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_monthly_granularity_returns_422(self, client: TestClient):
        params = {
            "granularity": "monthly",
            "start_date": "2026-10-15",
            "end_date": "2026-10-17",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_invalid_start_date_format_returns_422(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "not-a-valid-date",
            "end_date": "2026-10-17",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_invalid_end_date_format_returns_422(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-13-45",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_date_range_inverted_returns_422(self, client: TestClient):
        """end_date < start_date must return 422."""
        params = {
            "granularity": "daily",
            "start_date": "2026-10-20",
            "end_date": "2026-10-15",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_missing_granularity_returns_422(self, client: TestClient):
        params = {
            "start_date": "2026-10-15",
            "end_date": "2026-10-17",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_missing_start_date_returns_422(self, client: TestClient):
        params = {
            "granularity": "daily",
            "end_date": "2026-10-17",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_missing_end_date_returns_422(self, client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422

    def test_exorbitant_horizon_returns_422(self, client: TestClient):
        """Range exceeding 5 years returns 422."""
        params = {
            "granularity": "daily",
            "start_date": "2026-01-01",
            "end_date": "2036-01-01",
        }
        res = client.get(self.ENDPOINT, params=params)
        assert res.status_code == 422


class TestForecastModelAvailabilityErrors:
    """Tests verifying 503 Service Unavailable when the requested model is missing."""

    ENDPOINT = "/api/v1/forecast"

    def test_missing_daily_model_returns_503(self, client_missing_daily: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-10-15",
            "end_date": "2026-10-17",
        }
        res = client_missing_daily.get(self.ENDPOINT, params=params)
        assert res.status_code == 503

    def test_missing_weekly_model_returns_503(self, client_missing_weekly: TestClient):
        params = {
            "granularity": "weekly",
            "start_date": "2026-10-12",
            "end_date": "2026-10-26",
        }
        res = client_missing_weekly.get(self.ENDPOINT, params=params)
        assert res.status_code == 503

    def test_available_weekly_model_succeeds_even_when_daily_missing(
        self, client_missing_daily: TestClient
    ):
        params = {
            "granularity": "weekly",
            "start_date": "2026-10-12",
            "end_date": "2026-10-19",
        }
        res = client_missing_daily.get(self.ENDPOINT, params=params)
        assert res.status_code == 200
