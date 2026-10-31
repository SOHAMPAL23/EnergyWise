"""
api/core/security.py
─────────────────────
Simple API-key authentication.

The API key is passed as the `X-API-Key` header (configurable via
`settings.API_KEY_NAME`).  Keys are compared in constant time to prevent
timing-oracle attacks.

Usage (in a route):
    @router.get("/protected")
    async def protected(key: str = Depends(require_api_key)):
        ...

To make a route PUBLIC, simply omit the dependency.
"""

import hmac
import logging

from fastapi import Header, HTTPException, status

from api.core.config import get_settings

logger = logging.getLogger(__name__)


async def require_api_key(
    x_api_key: str = Header(
        ...,
        alias="X-API-Key",
        description="API key for service authentication.",
    ),
) -> str:
    """
    FastAPI dependency that validates the `X-API-Key` header.

    Returns the validated key string on success so downstream code can log it.
    Raises HTTP 401 on failure.
    """
    settings = get_settings()
    expected = settings.API_KEY

    # Constant-time comparison prevents timing side-channels
    is_valid = hmac.compare_digest(x_api_key.encode(), expected.encode())

    if not is_valid:
        logger.warning("Invalid API key presented.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": f'ApiKey realm="{settings.APP_NAME}"'},
        )

    return x_api_key
