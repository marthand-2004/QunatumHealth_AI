"""
Tests for JWT middleware and RBAC dependency — Task 2.2.
Covers: 401 on expired token, 401 on invalid token,
        403 on wrong role, 200 on correct role.
Requirements: 1.3, 1.5, 1.6
"""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from bson import ObjectId
from httpx import AsyncClient, ASGITransport
from jose import jwt

from app import app
from backend.core.config import settings
from backend.core.database import get_db
from backend.core.security import create_access_token, hash_password

# ── Shared fixtures ───────────────────────────────────────────────────────────

SAMPLE_OID = ObjectId()
SAMPLE_EMAIL = "rbac_test@example.com"
SAMPLE_ROLE = "patient"

SAMPLE_USER_DOC = {
    "_id": SAMPLE_OID,
    "email": SAMPLE_EMAIL,
    "password_hash": hash_password("TestPass123"),
    "role": SAMPLE_ROLE,
    "created_at": datetime.utcnow(),
    "is_active": True,
}

DOCTOR_OID = ObjectId()
DOCTOR_USER_DOC = {
    "_id": DOCTOR_OID,
    "email": "doctor@example.com",
    "password_hash": hash_password("DoctorPass123"),
    "role": "doctor",
    "created_at": datetime.utcnow(),
    "is_active": True,
}


def _make_mock_db(user_doc: dict | None = SAMPLE_USER_DOC):
    """Return a mock Motor database that returns *user_doc* on find_one."""
    mock_db = MagicMock()
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=user_doc)
    mock_db.__getitem__ = MagicMock(return_value=mock_col)
    return mock_db


def _override_db(mock_db):
    def _dep():
        return mock_db
    return _dep


def _expired_access_token(user_doc: dict) -> str:
    """Create an access token that expired 1 minute ago."""
    payload = {
        "sub": str(user_doc["_id"]),
        "role": user_doc["role"],
        "type": "access",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# ── Helper: a minimal protected endpoint ─────────────────────────────────────
# We use GET /api/auth/me as the protected endpoint for 401/403 tests since it
# already uses get_current_user internally.  For role-specific tests we use the
# clinical dashboard endpoint which requires doctor/admin.

# ── 401 on expired token ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_expired_token_returns_401():
    """Req 1.3 — expired JWT is rejected with HTTP 401."""
    expired_token = _expired_access_token(SAMPLE_USER_DOC)

    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {expired_token}"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


# ── 401 on invalid / malformed token ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_token_returns_401():
    """Req 1.3 — malformed JWT is rejected with HTTP 401."""
    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/auth/me",
                headers={"Authorization": "Bearer this.is.not.a.valid.jwt"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_signature_returns_401():
    """Req 1.3 — JWT signed with wrong secret is rejected with HTTP 401."""
    payload = {
        "sub": str(SAMPLE_OID),
        "role": SAMPLE_ROLE,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
    }
    bad_token = jwt.encode(payload, "wrong-secret", algorithm=settings.ALGORITHM)

    mock_db = _make_mock_db()
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {bad_token}"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 401


# ── 403 on wrong role ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_patient_accessing_doctor_route_returns_403():
    """Req 1.5, 1.6 — patient token on a doctor-only route returns HTTP 403."""
    # Use a valid patient access token
    patient_token = create_access_token({"sub": str(SAMPLE_OID), "role": SAMPLE_ROLE})

    mock_db = _make_mock_db(SAMPLE_USER_DOC)
    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/clinical/high-risk",
                headers={"Authorization": f"Bearer {patient_token}"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 403


# ── 200 on correct role ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_doctor_accessing_doctor_route_returns_200():
    """Req 1.5, 1.6 — doctor token on a doctor-only route returns HTTP 200."""
    doctor_token = create_access_token({"sub": str(DOCTOR_OID), "role": "doctor"})

    mock_db = _make_mock_db(DOCTOR_USER_DOC)

    # Predictions collection: aggregate returns an async cursor with no results
    mock_predictions_col = MagicMock()
    mock_predictions_col.aggregate = MagicMock(return_value=_async_cursor([]))

    # Users collection: find_one returns the doctor doc (for auth lookup)
    mock_users_col = AsyncMock()
    mock_users_col.find_one = AsyncMock(return_value=DOCTOR_USER_DOC)
    mock_users_col.find = MagicMock(return_value=_async_cursor([]))

    # Alerts collection: find returns empty cursor
    mock_alerts_col = MagicMock()
    mock_alerts_col.find = MagicMock(return_value=_async_cursor([]))

    def _getitem(name):
        if name == "users":
            return mock_users_col
        if name == "alerts":
            return mock_alerts_col
        return mock_predictions_col

    mock_db.__getitem__ = MagicMock(side_effect=_getitem)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/clinical/high-risk",
                headers={"Authorization": f"Bearer {doctor_token}"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    # 200 means the role check passed (endpoint may return empty list)
    assert resp.status_code == 200


# ── get_current_user unit tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_current_user_valid_token():
    """get_current_user returns user doc for a valid access token."""
    from backend.core.deps import get_current_user
    from unittest.mock import patch

    token = create_access_token({"sub": str(SAMPLE_OID), "role": SAMPLE_ROLE})
    mock_db = _make_mock_db(SAMPLE_USER_DOC)

    user = await get_current_user(db=mock_db, token=token)
    assert user["email"] == SAMPLE_EMAIL
    assert user["role"] == SAMPLE_ROLE


@pytest.mark.asyncio
async def test_get_current_user_expired_token_raises_401():
    """get_current_user raises HTTP 401 for an expired token."""
    from fastapi import HTTPException
    from backend.core.deps import get_current_user

    expired_token = _expired_access_token(SAMPLE_USER_DOC)
    mock_db = _make_mock_db(SAMPLE_USER_DOC)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(db=mock_db, token=expired_token)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_invalid_token_raises_401():
    """get_current_user raises HTTP 401 for a malformed token."""
    from fastapi import HTTPException
    from backend.core.deps import get_current_user

    mock_db = _make_mock_db(SAMPLE_USER_DOC)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(db=mock_db, token="not.a.jwt")

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_role_correct_role_passes():
    """require_role returns user doc when role matches."""
    from backend.core.deps import require_role

    token = create_access_token({"sub": str(SAMPLE_OID), "role": SAMPLE_ROLE})
    mock_db = _make_mock_db(SAMPLE_USER_DOC)

    dep = require_role(["patient", "doctor"])
    # Simulate calling the inner dependency with a resolved current_user
    user = await dep(current_user=SAMPLE_USER_DOC)
    assert user["role"] == SAMPLE_ROLE


@pytest.mark.asyncio
async def test_require_role_wrong_role_raises_403():
    """require_role raises HTTP 403 when role does not match."""
    from fastapi import HTTPException
    from backend.core.deps import require_role

    dep = require_role(["doctor", "admin"])

    with pytest.raises(HTTPException) as exc_info:
        await dep(current_user=SAMPLE_USER_DOC)  # patient role

    assert exc_info.value.status_code == 403


# ── Async cursor helper ───────────────────────────────────────────────────────

class _async_cursor:
    """Minimal async cursor stub that yields items from a list."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._items:
            yield item
