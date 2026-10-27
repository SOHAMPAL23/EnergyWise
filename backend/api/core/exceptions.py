"""
api/core/exceptions.py
───────────────────────
Custom exception hierarchy for the EnergyWise API.

All domain exceptions inherit from `EnergyWiseError` so callers can catch a
single base class. Each concrete exception carries an HTTP status code that the
global exception handler (registered in `main.py`) uses to build a structured
JSON error response.
"""

from http import HTTPStatus
from typing import Any, Optional


class EnergyWiseError(Exception):
    """Base class for all application-level exceptions."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value
    default_detail: str = "An unexpected error occurred."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        context: Optional[Any] = None,
    ) -> None:
        self.detail = detail or self.default_detail
        self.context = context
        super().__init__(self.detail)


# ── 4xx Client Errors ────────────────────────────────────────────────────────

class NotFoundError(EnergyWiseError):
    """Resource not found."""
    status_code = HTTPStatus.NOT_FOUND.value
    default_detail = "The requested resource was not found."


class ValidationError(EnergyWiseError):
    """Request payload failed domain validation (beyond Pydantic schema)."""
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY.value
    default_detail = "Request validation failed."


class AuthenticationError(EnergyWiseError):
    """Missing or invalid API credentials."""
    status_code = HTTPStatus.UNAUTHORIZED.value
    default_detail = "Authentication required."


class AuthorizationError(EnergyWiseError):
    """Authenticated but insufficient permissions."""
    status_code = HTTPStatus.FORBIDDEN.value
    default_detail = "You do not have permission to perform this action."


# ── 5xx Server / ML Errors ────────────────────────────────────────────────────

class ModelNotAvailableError(EnergyWiseError):
    """The ML model artifact is missing or could not be loaded."""
    status_code = HTTPStatus.SERVICE_UNAVAILABLE.value
    default_detail = "The forecasting model is not available."


class ForecastError(EnergyWiseError):
    """An error occurred during the forecast computation."""
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    default_detail = "Forecast computation failed."


class ArtifactCorruptionError(EnergyWiseError):
    """An artifact file failed its SHA-256 integrity check."""
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    default_detail = "A model artifact failed its integrity check."
