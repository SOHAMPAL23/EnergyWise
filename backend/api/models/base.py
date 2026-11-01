"""
api/models/base.py
───────────────────
SQLAlchemy declarative base and session factory.

All ORM models should inherit from `Base`.
Only import `get_db` in route handlers or services that need persistence;
routes that are purely ML-serving do NOT need a database session.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from api.core.config import get_settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def _build_engine():
    settings = get_settings()
    return create_engine(
        settings.DATABASE_URL,
        echo=settings.DB_ECHO_SQL,
        pool_pre_ping=True,   # detect stale connections before using them
        future=True,
    )


# Module-level singletons (lazy-initialised on first import)
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal
