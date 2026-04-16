"""
Smoke tests for Pydantic data models — Task 1 scaffolding validation.
Verifies that all models can be instantiated and serialized correctly.
"""
from datetime import datetime

import pytest
from bson import ObjectId

from backend.models.user import User
from backend.models.lifestyle import LifestyleProfile
from backend.models.document import Document, LabParameter
from backend.models.prediction import FeatureVector, PredictionResult, FEATURE_DIM, FEATURE_NAMES
from backend.models.recommendation import Recommendation
from backend.models.report import Report
from backend.models.chat import ChatSession, ChatMessage
from backend.models.audit import AuditLog
from backend.models.common import PyObjectId


SAMPLE_OID = str(ObjectId())


# ── User ─────────────────────────────────────────────────────────────────────

def test_user_model():
    u = User(
        email="patient@example.com",
        password_hash="$2b$12$hash",
        role="patient",
        created_at=datetime.utcnow(),
    )
    assert u.role == "patient"
    assert u.is_active is True


def test_user_invalid_role():
    with pytest.raises(Exception):
        User(email="x@x.com", password_hash="h", role="superuser", created_at=datetime.utcnow())


# ── LifestyleProfile ──────────────────────────────────────────────────────────

def test_lifestyle_profile_model():
    lp = LifestyleProfile(
        user_id=SAMPLE_OID,
        bmi=24.5,
        family_history={"diabetes": True, "cvd": False, "ckd": False},
        smoking_status="never",
        alcohol_frequency="occasional",
        exercise_frequency=3,
        diet_type="balanced",
        sleep_hours=7.5,
        stress_level=4,
        medications=[],
        updated_at=datetime.utcnow(),
    )
    assert lp.bmi == 24.5
    assert lp.stress_level == 4


# ── LabParameter ─────────────────────────────────────────────────────────────

def test_lab_parameter_model():
    lp = LabParameter(
        name="glucose",
        value=5.5,
        unit="mmol/L",
        reference_range=(3.9, 6.1),
        is_abnormal=False,
        raw_text="Glucose 5.5 mmol/L",
    )
    assert lp.name == "glucose"
    assert lp.is_abnormal is False


def test_lab_parameter_round_trip():
    """Basic round-trip: serialize → deserialize → equal."""
    lp = LabParameter(
        name="hba1c",
        value=6.2,
        unit="%",
        reference_range=(4.0, 5.6),
        is_abnormal=True,
        raw_text="HbA1c 6.2%",
    )
    restored = LabParameter.model_validate_json(lp.model_dump_json())
    assert restored == lp


# ── Document ─────────────────────────────────────────────────────────────────

def test_document_model():
    doc = Document(
        user_id=SAMPLE_OID,
        filename="report.pdf",
        file_hash="abc123",
        file_size_bytes=1024,
        upload_time=datetime.utcnow(),
    )
    assert doc.ocr_status == "pending"
    assert doc.verified is False


# ── FeatureVector ─────────────────────────────────────────────────────────────

def test_feature_vector_14_dimensions():
    fv = FeatureVector(
        user_id=SAMPLE_OID,
        document_id=SAMPLE_OID,
        features=[float(i) for i in range(FEATURE_DIM)],
    )
    assert len(fv.features) == FEATURE_DIM


def test_feature_vector_wrong_dimensions():
    with pytest.raises(Exception):
        FeatureVector(
            user_id=SAMPLE_OID,
            document_id=SAMPLE_OID,
            features=[1.0, 2.0],  # only 2 dims
        )


def test_feature_vector_feature_names():
    fv = FeatureVector(
        user_id=SAMPLE_OID,
        document_id=SAMPLE_OID,
        features=[0.0] * FEATURE_DIM,
    )
    assert fv.feature_names == FEATURE_NAMES


# ── PredictionResult ──────────────────────────────────────────────────────────

def test_prediction_result_model():
    pr = PredictionResult(
        user_id=SAMPLE_OID,
        feature_vector_id=SAMPLE_OID,
        model_used="quantum",
        risk_scores={"diabetes": 42.3, "cvd": 67.1, "ckd": 18.9},
        timestamp=datetime.utcnow(),
    )
    assert pr.model_used == "quantum"
    assert "diabetes" in pr.risk_scores


# ── Recommendation ────────────────────────────────────────────────────────────

def test_recommendation_model():
    rec = Recommendation(
        prediction_id=SAMPLE_OID,
        disease="diabetes",
        text="Reduce sugar intake.",
        priority=1,
        source="rule",
    )
    assert rec.requires_physician is False


# ── Report ────────────────────────────────────────────────────────────────────

def test_report_model():
    r = Report(
        patient_id=SAMPLE_OID,
        generated_by=SAMPLE_OID,
        prediction_id=SAMPLE_OID,
        file_path="/data/reports/report.pdf",
        created_at=datetime.utcnow(),
    )
    assert r.share_token is None


# ── ChatSession / ChatMessage ─────────────────────────────────────────────────

def test_chat_session_model():
    cs = ChatSession(user_id=SAMPLE_OID)
    assert cs.user_id == SAMPLE_OID


def test_chat_message_model():
    cm = ChatMessage(
        session_id=SAMPLE_OID,
        role="user",
        content="What does my HbA1c mean?",
    )
    assert cm.role == "user"


# ── AuditLog ──────────────────────────────────────────────────────────────────

def test_audit_log_model():
    al = AuditLog(
        doctor_id=SAMPLE_OID,
        patient_id=SAMPLE_OID,
        endpoint="/api/clinical/patient/123",
    )
    assert al.doctor_id == SAMPLE_OID
