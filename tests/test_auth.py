import pytest
from fastapi import HTTPException

from app.core import auth


@pytest.mark.anyio
async def test_require_api_key_allows_when_not_configured(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "")

    assert await auth.require_api_key(None) is None


@pytest.mark.anyio
async def test_require_api_key_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "secret")

    with pytest.raises(HTTPException):
        await auth.require_api_key("wrong")


@pytest.mark.anyio
async def test_require_api_key_accepts_matching_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "secret")

    assert await auth.require_api_key("secret") is None
