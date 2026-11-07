"""
api/services/health.py
───────────────────────
Business logic for the health endpoint.

The service layer is deliberately kept thin: it reads from the ModelRegistry
(the source of truth for model state) and assembles a structured dict that
the schema layer validates and the route returns.

No DB connection is attempted here unless it's explicitly required in the
future. The health check should be fast and dependency-light.
"""

import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from api.core.config import get_settings
from api.services.model_registry import GRANULARITIES, get_registry

logger = logging.getLogger(__name__)


def get_health_status() -> Dict[str, Any]:
    """
    Assemble the health status payload.

    Returns a dict that maps directly onto `HealthResponse` (validated by
    Pydantic in the route layer).

    The service does NOT raise on unhealthy models — it records the status
    so the route can decide the appropriate HTTP status code.
    """
    settings = get_settings()
    registry = get_registry()

    # ── Per-model status ─────────────────────────────────────────────────────
    models: Dict[str, Any] = {}
    overall_model_healthy = True

    for granularity in GRANULARITIES:
        available = registry.is_available(granularity)
        if not available:
            overall_model_healthy = False

        info: Dict[str, Any] = {"available": available}

        if available:
            try:
                meta = registry.get_model_info(granularity)
                info["model_id"] = meta.get("model_id")
                info["model_version"] = meta.get("model_version")
                info["granularity"] = meta.get("granularity")
                info["trained_on"] = meta.get("created_at")
                # Expose train window from split block
                split = meta.get("split", {})
                info["train_start"] = split.get("train_start")
                info["train_end"] = split.get("train_end")
                info["n_train_rows"] = split.get("n_train")
            except Exception as exc:
                logger.warning("Could not read metadata for %s: %s", granularity, exc)
                info["metadata_error"] = str(exc)
        else:
            error_msg = registry.load_error(granularity)
            if error_msg:
                info["error"] = error_msg

        models[granularity] = info

    # ── Manifest metadata ────────────────────────────────────────────────────
    manifest_info: Dict[str, Any] = {
        "generated_at": registry.manifest_generated_at(),
        "summary": registry.manifest_summary(),
    }

    # ── Overall service status ───────────────────────────────────────────────
    service_status = "healthy" if overall_model_healthy else "degraded"

    return {
        "status": service_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "models": models,
        "manifest": manifest_info,
    }
