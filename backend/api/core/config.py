"""
api/core/config.py
──────────────────
Application-wide settings loaded from environment variables (or .env file).

Priority (highest → lowest):
    1. Real environment variable
    2. .env file at PROJECT_ROOT/.env
    3. Default value defined here

Never commit secrets. Use .env.example as the template.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _find_project_root() -> Path:
    """Traverse upwards to locate the project root (contains ml/ or .git)."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / "ml").exists() or (parent / "requirements.txt").exists():
            return parent
    return current.parent.parent.parent.parent

_PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """All application settings. Env-var names are the uppercase field names."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service metadata ──────────────────────────────────────────────────────
    APP_NAME: str = "EnergyWise Demand Forecasting API"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1                 # increase in production with gunicorn

    # ── API versioning ────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # ── Security ──────────────────────────────────────────────────────────────
    # API key for simple bearer-token auth (expand to OAuth2/JWT later)
    API_KEY: str = Field(
        default="dev-insecure-key-change-in-production",
        description="Bearer token used to authenticate API requests.",
    )
    API_KEY_NAME: str = "X-API-Key"

    # ── Database (PostgreSQL) ──────────────────────────────────────────────────
    # Only required if/when a persistence layer is wired in.
    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://energywise:energywise@localhost:5432/energywise",
        description="SQLAlchemy-compatible PostgreSQL DSN.",
    )
    DB_ECHO_SQL: bool = False        # set True to log every SQL statement

    # ── ML / Artifact paths ───────────────────────────────────────────────────
    ARTIFACTS_DIR: Path = _PROJECT_ROOT / "ml" / "artifacts"
    MANIFEST_PATH: Path = _PROJECT_ROOT / "ml" / "artifacts" / "manifest.json"

    # Which model versions are considered current (must match manifest.json)
    MODEL_VERSION_DAILY: str = "v2"
    MODEL_VERSION_WEEKLY: str = "v2"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("ARTIFACTS_DIR", "MANIFEST_PATH", mode="before")
    @classmethod
    def _to_path(cls, v):
        return Path(v)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
