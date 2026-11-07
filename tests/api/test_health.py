"""
tests/api/test_health.py
─────────────────────────
Integration-style tests for GET /api/v1/health.

Strategy
─────────
* The ModelRegistry is patched with a lightweight fake so these tests:
    - run without Prophet installed,
    - run without model files on disk,
    - remain fast (no I/O).
* `initialise_registry` is also patched to prevent the lifespan from
  triggering real artifact loading.
* A second fixture exercises the degraded-model path.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─── Fake registry helpers ────────────────────────────────────────────────────

def _make_fake_registry(
    daily_available: bool = True,
    weekly_available: bool = True,
) -> MagicMock:
    """Return a MagicMock that behaves like a fully-loaded ModelRegistry."""
    registry = MagicMock()

    daily_meta = {
        "model_id": "prophet-daily-v2",
        "model_version": "v2",
        "granularity": "daily",
        "created_at": "2026-10-07T05:17:15.847375+00:00",
        "split": {
            "train_start": "2015-01-01",
            "train_end": "2019-01-09",
            "n_train": 1470,
        },
    }
    weekly_meta = {
        "model_id": "prophet-weekly-v2",
        "model_version": "v2",
        "granularity": "weekly",
        "created_at": "2026-10-07T05:17:15.847375+00:00",
        "split": {
            "train_start": "2015-01-01",
            "train_end": "2019-01-09",
            "n_train": 210,
        },
    }

    registry.is_available.side_effect = lambda g: (
        daily_available if g == "daily" else weekly_available
    )
    registry.get_model_info.side_effect = lambda g: (
        daily_meta if g == "daily" else weekly_meta
    )
    registry.load_error.return_value = None
    registry.manifest_generated_at.return_value = "2026-10-07T05:17:23.851904+00:00"
    registry.manifest_summary.return_value = {
        "daily": {"model_id": "prophet-daily-v2", "model_version": "v2"},
        "weekly": {"model_id": "prophet-weekly-v2", "model_version": "v2"},
    }
    return registry



# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def healthy_client():
    """TestClient with both models available — patches kept alive for the test."""
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
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture()
def degraded_client():
    """TestClient where the weekly model failed to load."""
    registry = _make_fake_registry(daily_available=True, weekly_available=False)
    registry.load_error.side_effect = lambda g: (
        "Model file not found." if g == "weekly" else None
    )
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
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for GET /api/v1/health"""

    ENDPOINT = "/api/v1/health"

    # ── HTTP contract ─────────────────────────────────────────────────────────

    def test_returns_200_when_healthy(self, healthy_client):
        response = healthy_client.get(self.ENDPOINT)
        assert response.status_code == 200

    def test_returns_200_when_degraded(self, degraded_client):
        """Degraded is still HTTP 200 — the service is alive."""
        response = degraded_client.get(self.ENDPOINT)
        assert response.status_code == 200

    def test_content_type_is_json(self, healthy_client):
        response = healthy_client.get(self.ENDPOINT)
        assert "application/json" in response.headers["content-type"]

    # ── Schema validation ─────────────────────────────────────────────────────

    def test_response_has_required_top_level_keys(self, healthy_client):
        data: Dict[str, Any] = healthy_client.get(self.ENDPOINT).json()
        required = {"status", "timestamp", "service", "version", "environment",
                    "python_version", "platform", "models", "manifest"}
        assert required.issubset(data.keys())

    def test_status_is_healthy(self, healthy_client):
        data = healthy_client.get(self.ENDPOINT).json()
        assert data["status"] == "healthy"

    def test_status_is_degraded_when_model_missing(self, degraded_client):
        data = degraded_client.get(self.ENDPOINT).json()
        assert data["status"] == "degraded"

    # ── Model info ────────────────────────────────────────────────────────────

    def test_models_block_contains_daily_and_weekly(self, healthy_client):
        data = healthy_client.get(self.ENDPOINT).json()
        assert "daily" in data["models"]
        assert "weekly" in data["models"]

    def test_daily_model_available_true(self, healthy_client):
        data = healthy_client.get(self.ENDPOINT).json()
        assert data["models"]["daily"]["available"] is True

    def test_daily_model_version(self, healthy_client):
        data = healthy_client.get(self.ENDPOINT).json()
        assert data["models"]["daily"]["model_version"] == "v2"

    def test_daily_model_id(self, healthy_client):
        data = healthy_client.get(self.ENDPOINT).json()
        assert data["models"]["daily"]["model_id"] == "prophet-daily-v2"

    def test_weekly_model_available_false_in_degraded(self, degraded_client):
        data = degraded_client.get(self.ENDPOINT).json()
        assert data["models"]["weekly"]["available"] is False

    def test_weekly_model_error_present_in_degraded(self, degraded_client):
        data = degraded_client.get(self.ENDPOINT).json()
        assert data["models"]["weekly"].get("error") is not None

    # ── Manifest block ────────────────────────────────────────────────────────

    def test_manifest_block_present(self, healthy_client):
        data = healthy_client.get(self.ENDPOINT).json()
        assert "manifest" in data
        assert "generated_at" in data["manifest"]

    def test_manifest_summary_present(self, healthy_client):
        data = healthy_client.get(self.ENDPOINT).json()
        summary = data["manifest"]["summary"]
        assert summary is not None
        assert "daily" in summary
        assert "weekly" in summary

    # ── No auth required ──────────────────────────────────────────────────────

    def test_no_api_key_required(self, healthy_client):
        """Health is a public endpoint — no X-API-Key header needed."""
        response = healthy_client.get(self.ENDPOINT)
        assert response.status_code != 401

    def test_endpoint_accessible_without_headers(self, healthy_client):
        response = healthy_client.get(self.ENDPOINT, headers={})
        assert response.status_code == 200
