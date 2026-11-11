"""
backend/api/routes/forecast.py
──────────────────────────────
Forecast API endpoints:
- GET /api/v1/forecast: Query forecast points via query parameters.
- POST /api/v1/forecast: Query forecast points via JSON request body.
- GET /api/v1/forecast/summary: Aggregate statistics for the requested horizon.
- GET /api/v1/forecast/models: Model registry inspection and metadata.
"""

from datetime import date
import logging
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, Query, status

from api.schemas.forecast import ForecastItem, ForecastRequest, ForecastResponse, ForecastSummary
from api.services.forecast import generate_forecast
from api.services.model_registry import GRANULARITIES, get_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forecast"])


@router.get(
    "/forecast/models",
    status_code=status.HTTP_200_OK,
    summary="List forecasting models",
    description="Returns metadata and availability status for all registered forecasting models.",
)
async def list_models() -> Dict[str, Any]:
    """Inspect model registry status and metadata for available granularities."""
    registry = get_registry()
    models = {}
    for g in GRANULARITIES:
        is_avail = registry.is_available(g)
        item: Dict[str, Any] = {"available": is_avail}
        if is_avail:
            try:
                item["metadata"] = registry.get_model_info(g)
            except Exception as exc:
                item["metadata_error"] = str(exc)
        else:
            item["error"] = registry.load_error(g)
        models[g] = item

    return {
        "supported_granularities": list(GRANULARITIES),
        "manifest_generated_at": registry.manifest_generated_at(),
        "models": models,
    }


@router.get(
    "/forecast/summary",
    response_model=ForecastSummary,
    status_code=status.HTTP_200_OK,
    summary="Get forecast summary statistics",
    description="Returns aggregate metrics (count, mean, min, max) for the requested forecast window.",
)
async def get_forecast_summary(
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
) -> ForecastSummary:
    """Generate forecast and compute summary statistics over the generated series."""
    points = generate_forecast(
        granularity=granularity,
        start_date=start_date,
        end_date=end_date,
    )
    forecasts = [p["forecast"] for p in points]
    mean_val = sum(forecasts) / len(forecasts) if forecasts else 0.0
    min_val = min(forecasts) if forecasts else 0.0
    max_val = max(forecasts) if forecasts else 0.0
    version = points[0]["model_version"] if points else "unknown"

    return ForecastSummary(
        granularity=granularity,
        start_date=str(start_date),
        end_date=str(end_date),
        count=len(points),
        mean_forecast_mw=round(mean_val, 2),
        min_forecast_mw=round(min_val, 2),
        max_forecast_mw=round(max_val, 2),
        model_version=version,
    )


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


@router.post(
    "/forecast",
    response_model=List[ForecastItem],
    status_code=status.HTTP_200_OK,
    summary="Generate electricity demand forecast (POST body)",
    description="Same forecasting logic as GET /forecast, accepting query parameters as a JSON body.",
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
async def post_forecast(
    payload: ForecastRequest,
) -> List[ForecastItem]:
    """
    Handle POST /api/v1/forecast request with JSON body.
    """
    logger.info(
        "Forecast POST request received: granularity=%s, start_date=%s, end_date=%s",
        payload.granularity,
        payload.start_date,
        payload.end_date,
    )
    return generate_forecast(
        granularity=payload.granularity,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
