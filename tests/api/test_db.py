"""
tests/api/test_db.py
────────────────────
Unit tests for SQLAlchemy database dependencies, engine creation, and ORM models.
"""

from unittest.mock import patch

import pytest
from sqlalchemy import text

from api.core.config import Settings
from api.dependencies.db import get_db
from api.models.base import Base, get_engine, get_session_factory, reset_db_engine
from api.models.forecast_log import ForecastLog


@pytest.fixture(autouse=True)
def sqlite_db_env(tmp_path):
    """Use an in-memory or temporary SQLite database for tests."""
    db_file = tmp_path / "test.db"
    test_settings = Settings(DATABASE_URL=f"sqlite:///{db_file}")

    reset_db_engine()
    with patch("api.models.base.get_settings", return_value=test_settings):
        # Create all tables
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        yield
        reset_db_engine()


class TestDatabase:
    def test_get_db_session_lifecycle(self):
        gen = get_db()
        session = next(gen)
        assert session is not None

        # Execute simple query
        res = session.execute(text("SELECT 1")).scalar()
        assert res == 1

        # Session clean closure
        with pytest.raises(StopIteration):
            next(gen)

    def test_forecast_log_orm_crud(self):
        gen = get_db()
        session = next(gen)

        log = ForecastLog(
            granularity="daily",
            start_date="2026-10-15",
            end_date="2026-10-20",
            n_points=6,
            model_version="v2",
            client_ip="127.0.0.1",
        )
        session.add(log)
        session.commit()
        session.refresh(log)

        assert log.id is not None
        assert log.granularity == "daily"
        assert log.n_points == 6
        assert "ForecastLog" in repr(log)

        # Query back
        retrieved = session.query(ForecastLog).filter_by(id=log.id).first()
        assert retrieved is not None
        assert retrieved.start_date == "2026-10-15"

        try:
            next(gen)
        except StopIteration:
            pass

