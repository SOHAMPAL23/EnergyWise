"""
tests/api/test_exceptions.py
────────────────────────────
Unit tests for domain exception hierarchy.
"""

from http import HTTPStatus

import pytest

from api.core.exceptions import (
    ArtifactCorruptionError,
    AuthenticationError,
    AuthorizationError,
    EnergyWiseError,
    ForecastError,
    ModelNotAvailableError,
    NotFoundError,
    ValidationError,
)


class TestExceptions:
    def test_base_exception(self):
        exc = EnergyWiseError("Something failed", context={"code": 123})
        assert str(exc) == "Something failed"
        assert exc.detail == "Something failed"
        assert exc.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
        assert exc.context == {"code": 123}

    def test_default_details(self):
        assert NotFoundError().status_code == HTTPStatus.NOT_FOUND.value
        assert ValidationError().status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert AuthenticationError().status_code == HTTPStatus.UNAUTHORIZED.value
        assert AuthorizationError().status_code == HTTPStatus.FORBIDDEN.value
        assert ModelNotAvailableError().status_code == HTTPStatus.SERVICE_UNAVAILABLE.value
        assert ForecastError().status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
        assert ArtifactCorruptionError().status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value

    def test_custom_details_override_defaults(self):
        exc = ModelNotAvailableError("Daily model missing on disk")
        assert exc.detail == "Daily model missing on disk"
        assert exc.status_code == 503

