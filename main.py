"""
main.py
───────
Application entry point for the EnergyWise Demand Forecasting API.
Re-exports `create_app` and `app` from `backend.main`.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path so backend and api modules are importable
_backend_dir = Path(__file__).resolve().parent / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from backend.main import app, create_app, lifespan, settings  # noqa: F401
from api.services.model_registry import initialise_registry  # noqa: F401

__all__ = ["app", "create_app", "lifespan", "settings", "initialise_registry"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        workers=settings.WORKERS,
    )
