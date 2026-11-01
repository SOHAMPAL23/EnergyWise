"""
api/routes/health.py
─────────────────────
GET /api/v1/health

Returns service status, model availability, and model version information.
This endpoint is intentionally unauthenticated so load-balancer probes and
monitoring systems can call it without credentials.
"""

import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from api.schemas.health import HealthResponse
from api.services.health import get_health_status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Reports the overall service status, availability of each ML model, "
        "and the current model version. No authentication required. "
        "Returns HTTP 200 for both `healthy` and `degraded` states — "
        "inspect the `status` field to determine operational health."
    ),
    responses={
        200: {"description": "Health status (healthy or degraded)"},
        500: {"description": "Unexpected error assembling health payload"},
    },
)
async def health_check() -> JSONResponse:
    """
    Health check endpoint.

    Always returns HTTP 200 so load-balancers keep the service in rotation
    even when a model is degraded. Consumers should read the `status` field:

    * `"healthy"` — all models loaded and ready.
    * `"degraded"` — one or more models failed to load; forecasting may be
      partially unavailable.
    """
    payload = get_health_status()

    http_status = (
        status.HTTP_200_OK
        if payload["status"] == "healthy"
        else status.HTTP_200_OK   # still 200 — service is alive, model is degraded
    )

    logger.debug("Health check: status=%s", payload["status"])
    return JSONResponse(content=payload, status_code=http_status)
