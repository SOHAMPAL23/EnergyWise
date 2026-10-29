"""
api/core/logging.py
────────────────────
Structured logging setup for the API process.

Call `configure_logging()` once at application startup (inside lifespan or
__main__). All other modules should only call `logging.getLogger(__name__)`.
"""

import logging
import sys
from typing import Optional

from api.core.config import get_settings


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure the root logger.

    Args:
        level: Override LOG_LEVEL from settings (useful in tests).
    """
    settings = get_settings()
    effective_level = (level or settings.LOG_LEVEL).upper()

    logging.basicConfig(
        level=effective_level,
        format=settings.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,        # overwrite any existing basicConfig call
    )

    # Suppress noisy third-party loggers in production
    if settings.ENVIRONMENT == "production":
        for noisy in ("prophet", "cmdstanpy", "httpx", "uvicorn.access"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured | level=%s | env=%s",
        effective_level,
        settings.ENVIRONMENT,
    )
