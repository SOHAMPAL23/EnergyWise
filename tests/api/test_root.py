"""
tests/api/test_root.py
──────────────────────
Tests for root endpoints and middleware:
- GET /
- GET /health (shortcut)
- Response headers (X-Process-Time, X-Request-ID)
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.api.test_health import _make_fake_registry


@pytest.fixture()
def client():
    registry = _make_fake_registry(daily_available=True, weekly_available=True)
    import api.services.model_registry as reg_module

    def _fake_init():
        reg_module._registry = registry
        return registry

    with (
        patch.object(reg_module, "_registry", registry),
        patch("backend.main.initialise_registry", side_effect=_fake_init),
        patch("main.initialise_registry", side_effect=_fake_init),
    ):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestRootEndpoints:
    def test_root_returns_service_info(self, client: TestClient):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert "service" in data
        assert "version" in data
        assert "docs_url" in data
        assert "health_url" in data
        assert "forecast_url" in data

    def test_root_health_shortcut_returns_200(self, client: TestClient):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "models" in data

    def test_observability_headers_present(self, client: TestClient):
        res = client.get("/")
        assert res.status_code == 200
        assert "x-process-time" in res.headers
        assert "x-request-id" in res.headers

    def test_custom_request_id_preserved(self, client: TestClient):
        custom_id = "test-req-id-12345"
        res = client.get("/", headers={"X-Request-ID": custom_id})
        assert res.status_code == 200
        assert res.headers["x-request-id"] == custom_id

