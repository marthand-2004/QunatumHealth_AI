"""
Tests for the combined POST /api/predict/ endpoint — Task 9.1.

Covers:
- Quantum fallback on exception
- Quantum fallback on timeout
- Both models succeed (quantum + classical side by side)
- High-risk alert creation when any score > 75

Requirements: 5.5, 6.4, 6.5, 14.2
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from httpx import ASGITransport, AsyncClient

from app import app
from backend.core.database import get_db
from backend.core.deps import get_current_user
from backend.core.security import create_access_token

# ── Shared test data ──────────────────────────────────────────────────────────

SAMPLE_OID = ObjectId()
SAMPLE_FEATURES = [5.0, 6.5, 1.2, 200.0, 150.0, 14.0, 25.0, 45.0, 120.0, 80.0, 0.0, 3.0, 7.0, 4.0]
assert len(SAMPLE_FEATURES) == 14

SAMPLE_USER_DOC = {
    "_id": SAMPLE_OID,
    "email": "patient@example.com",
    "role": "patient",
    "is_active": True,
}

QUANTUM_SCORES = {"diabetes": 42.0, "cvd": 55.0, "ckd": 30.0}
CLASSICAL_SCORES = {"diabetes": 38.0, "cvd": 50.0, "ckd": 28.0}
HIGH_RISK_SCORES = {"diabetes": 80.0, "cvd": 85.0, "ckd": 76.0}


def _make_access_token() -> str:
    return create_access_token({"sub": str(SAMPLE_OID), "role": "patient"})


def _make_mock_db():
    """Build a mock Motor database with per-collection AsyncMocks."""
    fv_oid = ObjectId()
    pred_oid = ObjectId()

    fv_col = AsyncMock()
    fv_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=fv_oid))
    fv_col.find_one = AsyncMock(return_value=None)

    pred_col = AsyncMock()
    pred_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=pred_oid))

    alerts_col = AsyncMock()
    alerts_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))

    mock_db = MagicMock()

    def _getitem(name: str):
        if name == "feature_vectors":
            return fv_col
        if name == "predictions":
            return pred_col
        if name == "alerts":
            return alerts_col
        return AsyncMock()

    mock_db.__getitem__ = MagicMock(side_effect=_getitem)
    mock_db._fv_col = fv_col
    mock_db._pred_col = pred_col
    mock_db._alerts_col = alerts_col
    return mock_db


def _override_db(mock_db):
    def _dep():
        return mock_db
    return _dep


def _override_current_user():
    """Override get_current_user to bypass JWT validation in tests."""
    async def _dep():
        return SAMPLE_USER_DOC
    return _dep


# ── Test: quantum fallback on exception ───────────────────────────────────────

@pytest.mark.asyncio
async def test_combined_predict_quantum_exception_falls_back_to_classical():
    """Req 5.5, 14.2 — quantum exception triggers classical fallback."""
    mock_db = _make_mock_db()

    app.dependency_overrides[get_db] = _override_db(mock_db)
    app.dependency_overrides[get_current_user] = _override_current_user()
    try:
        with (
            patch("backend.routers.predict.predict_quantum", side_effect=RuntimeError("circuit error")),
            patch("backend.routers.predict.predict_classical", return_value=CLASSICAL_SCORES),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/predict/",
                    json={"features": SAMPLE_FEATURES},
                    headers={"Authorization": f"Bearer {_make_access_token()}"},
                )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 201
    data = resp.json()
    assert data["model_used"] == "classical"
    assert data["fallback_used"] is True
    assert data["quantum_scores"] is None
    assert data["classical_scores"] == CLASSICAL_SCORES
    assert data["risk_scores"] == CLASSICAL_SCORES


# ── Test: quantum fallback on timeout ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_combined_predict_quantum_timeout_falls_back_to_classical():
    """Req 5.5, 14.2 — quantum timeout triggers classical fallback."""
    mock_db = _make_mock_db()

    app.dependency_overrides[get_db] = _override_db(mock_db)
    app.dependency_overrides[get_current_user] = _override_current_user()
    try:
        with (
            patch("backend.routers.predict.predict_quantum", side_effect=asyncio.TimeoutError()),
            patch("backend.routers.predict.predict_classical", return_value=CLASSICAL_SCORES),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/predict/",
                    json={"features": SAMPLE_FEATURES},
                    headers={"Authorization": f"Bearer {_make_access_token()}"},
                )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 201
    data = resp.json()
    assert data["model_used"] == "classical"
    assert data["fallback_used"] is True
    assert data["quantum_scores"] is None
    assert data["classical_scores"] == CLASSICAL_SCORES


# ── Test: both models succeed ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_combined_predict_both_models_succeed():
    """Req 6.5 — when both succeed, quantum scores are primary and both are returned."""
    mock_db = _make_mock_db()

    app.dependency_overrides[get_db] = _override_db(mock_db)
    app.dependency_overrides[get_current_user] = _override_current_user()
    try:
        with (
            patch("backend.routers.predict.predict_quantum", return_value=QUANTUM_SCORES),
            patch("backend.routers.predict.predict_classical", return_value=CLASSICAL_SCORES),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/predict/",
                    json={"features": SAMPLE_FEATURES},
                    headers={"Authorization": f"Bearer {_make_access_token()}"},
                )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 201
    data = resp.json()
    assert data["model_used"] == "quantum"
    assert data["fallback_used"] is False
    assert data["quantum_scores"] == QUANTUM_SCORES
    assert data["classical_scores"] == CLASSICAL_SCORES
    # Primary risk_scores should be quantum
    assert data["risk_scores"] == QUANTUM_SCORES


# ── Test: high-risk alert creation ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_combined_predict_high_risk_creates_alert():
    """Req 10.6 — any risk score > 75 triggers alert insertion into alerts collection."""
    mock_db = _make_mock_db()

    app.dependency_overrides[get_db] = _override_db(mock_db)
    app.dependency_overrides[get_current_user] = _override_current_user()
    try:
        with (
            patch("backend.routers.predict.predict_quantum", return_value=HIGH_RISK_SCORES),
            patch("backend.routers.predict.predict_classical", return_value=HIGH_RISK_SCORES),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/predict/",
                    json={"features": SAMPLE_FEATURES},
                    headers={"Authorization": f"Bearer {_make_access_token()}"},
                )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 201
    mock_db._alerts_col.insert_one.assert_called_once()
    alert_doc = mock_db._alerts_col.insert_one.call_args[0][0]
    assert "patient_id" in alert_doc
    assert "risk_scores" in alert_doc
    assert "timestamp" in alert_doc


@pytest.mark.asyncio
async def test_combined_predict_low_risk_no_alert():
    """No alert is created when all risk scores are <= 75."""
    mock_db = _make_mock_db()

    app.dependency_overrides[get_db] = _override_db(mock_db)
    app.dependency_overrides[get_current_user] = _override_current_user()
    try:
        with (
            patch("backend.routers.predict.predict_quantum", return_value=QUANTUM_SCORES),
            patch("backend.routers.predict.predict_classical", return_value=CLASSICAL_SCORES),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/predict/",
                    json={"features": SAMPLE_FEATURES},
                    headers={"Authorization": f"Bearer {_make_access_token()}"},
                )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 201
    mock_db._alerts_col.insert_one.assert_not_called()


# ── Test: prediction is persisted ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_combined_predict_persists_both_scores():
    """Req 5.7 — PredictionResult with quantum_scores and classical_scores is persisted."""
    mock_db = _make_mock_db()

    app.dependency_overrides[get_db] = _override_db(mock_db)
    app.dependency_overrides[get_current_user] = _override_current_user()
    try:
        with (
            patch("backend.routers.predict.predict_quantum", return_value=QUANTUM_SCORES),
            patch("backend.routers.predict.predict_classical", return_value=CLASSICAL_SCORES),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/predict/",
                    json={"features": SAMPLE_FEATURES},
                    headers={"Authorization": f"Bearer {_make_access_token()}"},
                )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 201
    mock_db._pred_col.insert_one.assert_called_once()
    persisted = mock_db._pred_col.insert_one.call_args[0][0]
    assert persisted["quantum_scores"] == QUANTUM_SCORES
    assert persisted["classical_scores"] == CLASSICAL_SCORES
    assert persisted["model_used"] == "quantum"
