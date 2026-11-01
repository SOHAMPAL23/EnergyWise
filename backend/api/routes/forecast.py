"""
backend/api/routes/forecast.py
──────────────────────────────
GET /api/v1/forecast

Generates national grid electricity load forecasts using pretrained Prophet models.
"""

from datetime import date
import logging
from typing import List, Literal

from fastapi import APIRouter, Query, status

from api.schemas.forecast import ForecastItem, ForecastResponse
from api.services.forecast import generate_forecast

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forecast"])


@router.get(
    "/forecast",
    response_model=List[ForecastItem],
    status_code=status.HTTP_200_OK,
    summary="Generate electricity demand forecast",
    description=(
        "Returns point predictions and 95% Bayesian uncertainty intervals for "
        "German national grid electricity load over the requested date range.\n\n"
        "**Supported granularities**:\n"
        "* `daily`: Day-ahead aggregation (calendar days, UTC midnight start)\n"
        "* `weekly`: Week-ahead aggregation (weeks, Monday start)\n\n"
        "**Error responses**:\n"
        "* `422 Unprocessable Entity`: Invalid date format, invalid granularity, or start_date > end_date.\n"
        "* `503 Service Unavailable`: Model artifact is missing or could not be loaded."
    ),
    responses={
        200: {
            "description": "Successful forecast generation.",
            "model": List[ForecastItem],
        },
        422: {
            "description": "Validation error (invalid date, invalid granularity, or invalid date range).",
        },
        503: {
            "description": "Model unavailable (artifact missing or corrupted).",
        },
    },
)
async def get_forecast(
    granularity: Literal["daily", "weekly"] = Query(
        ...,
        description="Forecast aggregation granularity ('daily' or 'weekly').",
        examples=["daily"],
    ),
    start_date: date = Query(
        ...,
        description="Inclusive start date of the forecast period (YYYY-MM-DD).",
        examples=["2026-10-15"],
    ),
    end_date: date = Query(
        ...,
        description="Inclusive end date of the forecast period (YYYY-MM-DD).",
        examples=["2026-10-20"],
    ),
) -> List[ForecastItem]:
    """
    Handle GET /api/v1/forecast request.

    Validates inputs, retrieves the pretrained model artifact from the in-memory
    registry, computes the stateless forecast, and returns structured forecast points.
    """
    logger.info(
        "Forecast request received: granularity=%s, start_date=%s, end_date=%s",
        granularity,
        start_date,
        end_date,
    )
    results = generate_forecast(
        granularity=granularity,
        start_date=start_date,
        end_date=end_date,
    )
    return results
