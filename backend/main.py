"""
backend/main.py
───────────────
Application entry point for the EnergyWise Demand Forecasting API.

Responsibilities
─────────────────
* Configure logging.
* Define the FastAPI lifespan (startup / shutdown).
* Register all route prefixes.
* Register global exception handlers.
* Serve with Uvicorn when run directly.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Ensure backend directory is in sys.path
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import get_settings
from api.core.exceptions import EnergyWiseError
from api.core.logging import configure_logging
from api.routes import forecast, health
from api.services.model_registry import initialise_registry

# ── Settings (cached singleton) ───────────────────────────────────────────────
settings = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────
configure_logging()
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup / shutdown lifecycle for the FastAPI application.

    Startup
    -------
    * Load and verify all ML model artifacts from disk.
    * Fail loudly with a warning if models can't load (service still starts
      so /health can report degraded status).

    Shutdown
    --------
    * Placeholder for future resource cleanup (e.g. DB connection pool).
    """
    logger.info("=== EnergyWise API starting up ===")
    logger.info("Environment : %s", settings.ENVIRONMENT)
    logger.info("Artifacts   : %s", settings.ARTIFACTS_DIR)

    # Load ML artifacts — non-fatal; errors are surfaced via /health
    try:
        registry = initialise_registry()
        loaded = sum(1 for g in ("daily", "weekly") if registry.is_available(g))
        logger.info("Model registry initialised: %d/2 models available.", loaded)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error during registry initialisation: %s", exc)

    yield  # ── application is running ──

    logger.info("=== EnergyWise API shutting down ===")


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "REST API for the EnergyWise Energy Demand Forecasting service. "
            "Provides Prophet-based daily and weekly load forecasts for the "
            "German national grid (ENTSO-E, OPSD dataset)."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handlers ─────────────────────────────────────────────
    @app.exception_handler(EnergyWiseError)
    async def energywise_error_handler(
        request: Request, exc: EnergyWiseError
    ) -> JSONResponse:
        logger.warning(
            "Domain error [%s %s]: %s",
            request.method,
            request.url.path,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "type": type(exc).__name__},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled exception [%s %s]", request.method, request.url.path
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router, prefix=settings.API_V1_PREFIX)
    app.include_router(forecast.router, prefix=settings.API_V1_PREFIX)

    return app


# ── Singleton app instance (used by uvicorn) ──────────────────────────────────
app = create_app()


# ── Development entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        workers=settings.WORKERS,
    )
