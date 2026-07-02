"""
Small API-key authentication dependency.

This protects operational and history APIs without complicating the project
with user management. Streaming and health endpoints remain public for local
development.
"""

from fastapi import Header
from fastapi import HTTPException
from fastapi import status

from app.core.config import API_KEY


async def require_api_key(x_api_key: str | None = Header(default=None)):
    """
    Require a matching X-API-Key header when API_KEY is configured.
    """

    if not API_KEY:
        return

    if x_api_key == API_KEY:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key.",
    )
