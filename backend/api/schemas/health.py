"""
api/schemas/health.py
──────────────────────
Pydantic response schemas for GET /api/v1/health.

These are the only models that define the contract between the health service
and the HTTP layer. Pydantic v2 is used throughout.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ModelVersionInfo(BaseModel):
    """Status and metadata for a single granularity model."""

    model_config = {"protected_namespaces": ()}

    available: bool = Field(
        ...,
        description="Whether the model artifact loaded successfully.",
    )
    model_id: Optional[str] = Field(
        None,
        description="Full model identifier, e.g. 'prophet-daily-v2'.",
        examples=["prophet-daily-v2"],
    )
    model_version: Optional[str] = Field(
        None,
        description="Semantic version string, e.g. 'v2'.",
        examples=["v2"],
    )
    granularity: Optional[str] = Field(
        None,
        description="Forecast granularity: 'daily' or 'weekly'.",
        examples=["daily"],
    )
    trained_on: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp of when the artifact was created.",
    )
    train_start: Optional[str] = Field(
        None,
        description="First date in the training window (YYYY-MM-DD).",
    )
    train_end: Optional[str] = Field(
        None,
        description="Last date in the training window (YYYY-MM-DD).",
    )
    n_train_rows: Optional[int] = Field(
        None,
        description="Number of rows in the training set.",
    )
    error: Optional[str] = Field(
        None,
        description="Load error message if `available` is False.",
    )
    metadata_error: Optional[str] = Field(
        None,
        description="Non-fatal error reading extended metadata.",
    )


class ManifestInfo(BaseModel):
    """Top-level manifest metadata reported in the health response."""

    generated_at: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp when manifest.json was last generated.",
    )
    summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-granularity test-metric summary from the manifest.",
    )


class HealthResponse(BaseModel):
    """
    Full health response payload.

    HTTP 200 is returned for both `healthy` and `degraded` status.
    Consumers should inspect `status` to determine operational state.
    """

    status: Literal["healthy", "degraded"] = Field(
        ...,
        description="'healthy' if all models loaded; 'degraded' if any failed.",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of this health check.",
    )
    service: str = Field(..., description="Human-readable service name.")
    version: str = Field(..., description="API version string.")
    environment: str = Field(
        ...,
        description="Runtime environment: development | staging | production.",
    )
    python_version: str = Field(..., description="Python interpreter version.")
    platform: str = Field(..., description="OS / platform string.")
    models: Dict[str, ModelVersionInfo] = Field(
        ...,
        description="Per-granularity model availability and version information.",
    )
    manifest: ManifestInfo = Field(
        ...,
        description="Summary metadata from the artifact manifest.",
    )

    model_config = {"json_schema_extra": {"example": {
        "status": "healthy",
        "timestamp": "2026-10-14T06:34:00.000000+00:00",
        "service": "EnergyWise Demand Forecasting API",
        "version": "0.1.0",
        "environment": "development",
        "python_version": "3.11.9",
        "platform": "Windows-10-10.0.26200-SP0",
        "models": {
            "daily": {
                "available": True,
                "model_id": "prophet-daily-v2",
                "model_version": "v2",
                "granularity": "daily",
                "trained_on": "2026-10-07T05:17:15.847375+00:00",
                "train_start": "2015-01-01",
                "train_end": "2019-01-09",
                "n_train_rows": 1470,
            },
            "weekly": {
                "available": True,
                "model_id": "prophet-weekly-v2",
                "model_version": "v2",
                "granularity": "weekly",
                "trained_on": "2026-10-07T05:17:15.847375+00:00",
                "train_start": "2015-01-01",
                "train_end": "2019-01-09",
                "n_train_rows": 210,
            },
        },
        "manifest": {
            "generated_at": "2026-10-07T05:17:23.851904+00:00",
            "summary": {
                "daily": {"model_id": "prophet-daily-v2", "model_version": "v2"},
                "weekly": {"model_id": "prophet-weekly-v2", "model_version": "v2"},
            },
        },
    }}}
