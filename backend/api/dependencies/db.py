"""
api/dependencies/db.py
───────────────────────
FastAPI dependency that yields a SQLAlchemy Session per request.

Use this only in routes/services that actually need persistence.
ML-serving routes should NOT inject this dependency.

Usage:
    @router.post("/something")
    async def endpoint(db: Session = Depends(get_db)):
        ...
"""

from collections.abc import Generator

from sqlalchemy.orm import Session

from api.models.base import get_session_factory


def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session scoped to the request lifetime.

    The session is always closed in `finally`, so connections are returned
    to the pool even if an exception propagates through the handler.
    """
    SessionLocal = get_session_factory()
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
