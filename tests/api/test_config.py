"""
tests/api/test_config.py
────────────────────────
Unit tests for application configuration and settings.
"""

from pathlib import Path

import pytest

from api.core.config import Settings, get_settings


class TestSettings:
    def test_default_settings(self):
        settings = Settings()
        assert settings.APP_NAME == "EnergyWise Demand Forecasting API"
        assert settings.ENVIRONMENT in ("development", "staging", "production")
        assert settings.API_V1_PREFIX == "/api/v1"
        assert isinstance(settings.ARTIFACTS_DIR, Path)
        assert isinstance(settings.MANIFEST_PATH, Path)
        assert settings.MODEL_VERSION_DAILY == "v2"
        assert settings.MODEL_VERSION_WEEKLY == "v2"

    def test_get_settings_cached_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_path_validator(self):
        settings = Settings(ARTIFACTS_DIR="custom/dir", MANIFEST_PATH="custom/manifest.json")
        assert isinstance(settings.ARTIFACTS_DIR, Path)
        assert isinstance(settings.MANIFEST_PATH, Path)
        assert settings.ARTIFACTS_DIR == Path("custom/dir")
        assert settings.MANIFEST_PATH == Path("custom/manifest.json")

