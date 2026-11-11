"""
tests/api/test_security.py
──────────────────────────
Unit and integration tests for API key authentication security.
"""

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.core.config import get_settings
from api.core.security import require_api_key


@pytest.fixture()
def secured_app():
    app = FastAPI()

    @app.get("/protected")
    async def protected_endpoint(key: str = Depends(require_api_key)):
        return {"status": "authorized", "key_received": key}

    return app


class TestSecurityDependency:
    def test_valid_api_key_succeeds(self, secured_app: FastAPI):
        settings = get_settings()
        client = TestClient(secured_app)
        res = client.get("/protected", headers={"X-API-Key": settings.API_KEY})
        assert res.status_code == 200
        assert res.json()["status"] == "authorized"

    def test_missing_api_key_returns_401(self, secured_app: FastAPI):
        client = TestClient(secured_app)
        res = client.get("/protected")
        assert res.status_code == 401
        assert "WWW-Authenticate" in res.headers
        assert "Invalid or missing API key" in res.json()["detail"]

    def test_invalid_api_key_returns_401(self, secured_app: FastAPI):
        client = TestClient(secured_app)
        res = client.get("/protected", headers={"X-API-Key": "wrong-secret-key"})
        assert res.status_code == 401
        assert "WWW-Authenticate" in res.headers
        assert "Invalid or missing API key" in res.json()["detail"]

    def test_empty_api_key_returns_401(self, secured_app: FastAPI):
        client = TestClient(secured_app)
        res = client.get("/protected", headers={"X-API-Key": ""})
        assert res.status_code == 401

