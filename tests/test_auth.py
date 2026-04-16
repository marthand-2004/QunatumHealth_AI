"""
Unit tests for auth endpoints — Task 2.1.
Tests: POST /api/auth/register, POST /api/auth/login,
       POST /api/auth/refresh, GET /api/auth/me
Requirements: 1.1, 1.2, 1.4, 1.7
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
from datetime import datetime

from httpx import AsyncClient, ASGITransport

from app import app
from backend.core.database import get_db
from backend.core.security import hash_password, create_access_token, create_refresh_token


# ── Shared test data ──────────────────────────────────────────────────────────

SAMPLE_OID = ObjectId()
SAMPLE_EMAIL = "test@example.com"
SAMPLE_PASSWORD = "SecurePass123"
SAMPLE_ROLE = "patient"

SAMPLE_USER_DOC = {
    "_id": SAMPLE_OID,
    "email": SAMPLE_EMAIL,
    "password_hash": hash_password(SAMPLE_PASSWORD),
    "role": SAMPLE_ROLE,
    "created_at": datetime.utcnow(),
    "is_active": True,
}


def _make_mock_db():
    """Return a mock Motor database."""
    mock_db = MagicMock()
    return mock_db


def _override_db(mock_db):
    """Return a FastAPI dependency override that yields mock_db."""
    def _dep():
        return mock_db
    return _dep


# ── POST /api/auth/register ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_success():
    """Req 1.1 — new user registration returns 201 with profile."""
    mock_db = _make_mock_db()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=None)
    mock_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=SAMPLE_OID))
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/register", json={
                "email": SAMPLE_EMAIL,
                "password": SAMPLE_PASSWORD,
                "role": SAMPLE_ROLE,
            })
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == SAMPLE_EMAIL
    assert data["role"] == SAMPLE_ROLE
    assert data["is_active"] is True
    assert "id" in data


@pytest.mark.asyncio
async def test_register_duplicate_email_returns_409():
    """Req 1.7 — duplicate email returns HTTP 409."""
    mock_db = _make_mock_db()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=SAMPLE_USER_DOC)
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/register", json={
                "email": SAMPLE_EMAIL,
                "password": SAMPLE_PASSWORD,
                "role": SAMPLE_ROLE,
            })
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 409
    detail = resp.json()["detail"].lower()
    assert "email" in detail or "account" in detail or "exists" in detail


@pytest.mark.asyncio
async def test_register_missing_fields_returns_422():
    """Req 1.1 — missing required fields returns HTTP 422."""
    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/register", json={"email": SAMPLE_EMAIL})
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_invalid_email_returns_422():
    """Req 1.1 — invalid email format returns HTTP 422."""
    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/register", json={
                "email": "not-an-email",
                "password": SAMPLE_PASSWORD,
                "role": SAMPLE_ROLE,
            })
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 422


# ── POST /api/auth/login ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_login_success_returns_token_pair():
    """Req 1.2 — valid credentials return access + refresh tokens."""
    mock_db = _make_mock_db()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=SAMPLE_USER_DOC)
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/login", json={
                "email": SAMPLE_EMAIL,
                "password": SAMPLE_PASSWORD,
            })
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401():
    """Req 1.2 — wrong password returns HTTP 401."""
    mock_db = _make_mock_db()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=SAMPLE_USER_DOC)
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/login", json={
                "email": SAMPLE_EMAIL,
                "password": "WrongPassword!",
            })
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_email_returns_401():
    """Req 1.2 — unknown email returns HTTP 401."""
    mock_db = _make_mock_db()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=None)
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/login", json={
                "email": "nobody@example.com",
                "password": SAMPLE_PASSWORD,
            })
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_missing_fields_returns_422():
    """Req 1.1 — missing password returns HTTP 422."""
    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/login", json={"email": SAMPLE_EMAIL})
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 422


# ── POST /api/auth/refresh ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_refresh_success():
    """Req 1.2 — valid refresh token issues new token pair."""
    refresh_token = create_refresh_token({"sub": str(SAMPLE_OID), "role": SAMPLE_ROLE})

    mock_db = _make_mock_db()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=SAMPLE_USER_DOC)
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/refresh", json={"refresh_token": refresh_token})
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_refresh_invalid_token_returns_401():
    """Req 1.2 — invalid refresh token returns HTTP 401."""
    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/refresh", json={"refresh_token": "not.a.valid.token"})
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_with_access_token_returns_401():
    """Req 1.2 — passing an access token to /refresh returns HTTP 401 (wrong token type)."""
    access_token = create_access_token({"sub": str(SAMPLE_OID), "role": SAMPLE_ROLE})

    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/auth/refresh", json={"refresh_token": access_token})
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


# ── GET /api/auth/me ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_me_success():
    """Req 1.1 — valid access token returns user profile."""
    access_token = create_access_token({"sub": str(SAMPLE_OID), "role": SAMPLE_ROLE})

    mock_db = _make_mock_db()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=SAMPLE_USER_DOC)
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == SAMPLE_EMAIL
    assert data["role"] == SAMPLE_ROLE


@pytest.mark.asyncio
async def test_me_no_token_returns_403():
    """Req 1.3 — missing Authorization header returns 403 (HTTPBearer default)."""
    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/auth/me")
    finally:
        app.dependency_overrides.pop(get_db, None)

    # HTTPBearer returns 403 when no credentials are provided
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_me_invalid_token_returns_401():
    """Req 1.3 — invalid token returns HTTP 401."""
    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/auth/me",
                headers={"Authorization": "Bearer invalid.token.here"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_with_refresh_token_returns_401():
    """Req 1.3 — passing a refresh token to /me returns HTTP 401."""
    refresh_token = create_refresh_token({"sub": str(SAMPLE_OID), "role": SAMPLE_ROLE})

    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {refresh_token}"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


# ── Password hashing (Req 1.4) ────────────────────────────────────────────────

def test_password_is_bcrypt_hashed():
    """Req 1.4 — stored password_hash is a bcrypt hash, not plaintext."""
    hashed = hash_password(SAMPLE_PASSWORD)
    assert hashed != SAMPLE_PASSWORD
    assert hashed.startswith("$2b$")
    assert hash_password(SAMPLE_PASSWORD) != hash_password(SAMPLE_PASSWORD)  # salted


def test_password_verify_correct():
    """Req 1.4 — verify_password returns True for correct password."""
    from backend.core.security import verify_password
    hashed = hash_password(SAMPLE_PASSWORD)
    assert verify_password(SAMPLE_PASSWORD, hashed) is True


def test_password_verify_wrong():
    """Req 1.4 — verify_password returns False for wrong password."""
    from backend.core.security import verify_password
    hashed = hash_password(SAMPLE_PASSWORD)
    assert verify_password("wrong_password", hashed) is False
