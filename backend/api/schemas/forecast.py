"""
backend/api/schemas/forecast.py
───────────────────────────────
Pydantic schemas for the Forecast API (GET /api/v1/forecast).
"""

from datetime import date
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ForecastItem(BaseModel):
    """
    A single forecast point for a given period (day or week).

    Fields
    ──────
    timestamp     : ISO-8601 string of the start of the forecasted interval.
    forecast      : Predicted mean demand in Megawatts (MW) (yhat).
    lower_bound   : Lower limit of the 95% uncertainty interval in MW (yhat_lower).
    upper_bound   : Upper limit of the 95% uncertainty interval in MW (yhat_upper).
    granularity   : Interval granularity ("daily" or "weekly").
    model_version : Version tag of the model artifact used for inference.
    """

    timestamp: str = Field(
        ...,
        description="Forecast interval timestamp (ISO-8601 format, e.g. '2026-10-15T00:00:00').",
        examples=["2026-10-15T00:00:00"],
    )
    forecast: float = Field(
        ...,
        description="Point prediction of electricity load in Megawatts (MW).",
        examples=[66728.98],
    )
    lower_bound: float = Field(
        ...,
        description="Lower bound of the 95% prediction interval in Megawatts (MW).",
        examples=[63329.40],
    )
    upper_bound: float = Field(
        ...,
        description="Upper bound of the 95% prediction interval in Megawatts (MW).",
        examples=[70270.23],
    )
    granularity: Literal["daily", "weekly"] = Field(
        ...,
        description="Granularity of the forecasted period ('daily' or 'weekly').",
        examples=["daily"],
    )
    model_version: str = Field(
        ...,
        description="Version tag of the model artifact used for inference.",
        examples=["v2"],
    )

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "timestamp": "2026-10-15T00:00:00",
                "forecast": 66728.98,
                "lower_bound": 63329.40,
                "upper_bound": 70270.23,
                "granularity": "daily",
                "model_version": "v2",
            }
        },
    }


# Type alias for list of forecast items
ForecastResponse = List[ForecastItem]
