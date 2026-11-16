"""
tests/api/test_real_models_e2e.py
─────────────────────────────────
End-to-end integration test with real ML model artifacts on disk.
Exercises the actual lifespan loading, SHA-256 verification, and Prophet inference.
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture(scope="module")
def real_api_client():
    """TestClient that uses the actual disk artifacts and real lifespan."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestRealModelsE2E:
    def test_real_health_reports_healthy(self, real_api_client: TestClient):
        res = real_api_client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["models"]["daily"]["available"] is True
        assert data["models"]["weekly"]["available"] is True

    def test_real_daily_forecast(self, real_api_client: TestClient):
        params = {
            "granularity": "daily",
            "start_date": "2026-11-01",
            "end_date": "2026-11-03",
        }
        res = real_api_client.get("/api/v1/forecast", params=params)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 3
        for item in data:
            assert item["granularity"] == "daily"
            assert item["model_version"] == "v2"
            # German demand is typically between 30,000 MW and 90,000 MW
            assert 30000 <= item["forecast"] <= 90000
            assert item["lower_bound"] <= item["upper_bound"]

    def test_real_weekly_forecast(self, real_api_client: TestClient):
        params = {
            "granularity": "weekly",
            "start_date": "2026-11-02",
            "end_date": "2026-11-16",
        }
        res = real_api_client.get("/api/v1/forecast", params=params)
        assert res.status_code == 200
        data = res.json()
        assert len(data) >= 2
        for item in data:
            assert item["granularity"] == "weekly"
            assert item["model_version"] == "v2"
            assert 30000 <= item["forecast"] <= 90000
            assert item["lower_bound"] <= item["upper_bound"]

    def test_real_post_forecast(self, real_api_client: TestClient):
        payload = {
            "granularity": "daily",
            "start_date": "2026-11-05",
            "end_date": "2026-11-07",
        }
        res = real_api_client.post("/api/v1/forecast", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 3

