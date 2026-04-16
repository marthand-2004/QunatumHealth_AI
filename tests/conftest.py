"""Shared pytest fixtures for QuantumHealthAI tests."""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from app import app


@pytest.fixture
async def client():
    """Async HTTP test client backed by the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_db():
    """Return a mock Motor database."""
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=AsyncMock())
    return db
