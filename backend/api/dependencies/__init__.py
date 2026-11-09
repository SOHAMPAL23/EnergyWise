"""
backend/api/dependencies
────────────────────────
FastAPI route dependencies package.
"""

from api.dependencies.db import get_db

__all__ = ["get_db"]
