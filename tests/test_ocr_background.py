"""
Unit tests for background OCR processing — Task 4.2.
Tests: run_ocr_background, get_ocr_result, GET /api/ocr/result/{job_id}
Requirements: 3.2, 3.3, 3.6
"""
import io
import os
import tempfile
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call
from bson import ObjectId

from cryptography.fernet import Fernet

from backend.services.ocr_service import (
    _merge_deduplicate,
    _ocr_image_bytes,
    get_ocr_result,
    run_ocr_background,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_OID = ObjectId()
SAMPLE_USER_OID = ObjectId()


def _make_mock_db(doc=None):
    """Return a mock Motor database whose 'documents' collection returns `doc`."""
    mock_col = AsyncMock()
    mock_col.find_one = AsyncMock(return_value=doc)
    mock_col.update_one = AsyncMock(return_value=None)
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)
    return mock_db, mock_col


def _make_encrypted_file(content: bytes) -> str:
    """Write Fernet-encrypted content to a temp file; return the path."""
    key = Fernet.generate_key()
    fernet = Fernet(key)
    encrypted = fernet.encrypt(content)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png.enc")
    tmp.write(encrypted)
    tmp.close()
    # Patch _get_fernet to use the same key
    return tmp.name, fernet


# ── _merge_deduplicate ────────────────────────────────────────────────────────

def test_merge_deduplicate_removes_duplicates():
    """Duplicate lines across paddle and tesseract output are removed."""
    paddle = ["Glucose: 5.5 mmol/L", "HbA1c: 6.2%"]
    tess = ["Glucose: 5.5 mmol/L", "Creatinine: 80 umol/L"]
    result = _merge_deduplicate(paddle, tess)
    assert result.count("Glucose: 5.5 mmol/L") == 1
    assert "HbA1c: 6.2%" in result
    assert "Creatinine: 80 umol/L" in result


def test_merge_deduplicate_preserves_order():
    """Paddle lines appear before tesseract-only lines."""
    paddle = ["A", "B"]
    tess = ["C", "A"]
    result = _merge_deduplicate(paddle, tess)
    assert result == ["A", "B", "C"]


def test_merge_deduplicate_empty_inputs():
    """Empty inputs produce empty output."""
    assert _merge_deduplicate([], []) == []


def test_merge_deduplicate_strips_blank_lines():
    """Blank / whitespace-only lines are excluded."""
    result = _merge_deduplicate(["  ", "Real line"], ["", "Real line"])
    assert result == ["Real line"]


# ── _ocr_image_bytes ──────────────────────────────────────────────────────────

def test_ocr_image_bytes_uses_paddle_when_available():
    """When PaddleOCR is available it is called and its lines are returned."""
    with patch("backend.services.ocr_service._run_paddle_ocr", return_value=["Paddle line"]) as mock_paddle, \
         patch("backend.services.ocr_service._run_tesseract_ocr", return_value=[]) as mock_tess:
        result = _ocr_image_bytes(b"fake-image")
    mock_paddle.assert_called_once()
    assert "Paddle line" in result


def test_ocr_image_bytes_falls_back_to_tesseract_on_paddle_failure():
    """When PaddleOCR raises, Tesseract result is returned."""
    with patch("backend.services.ocr_service._run_paddle_ocr", side_effect=RuntimeError("no paddle")), \
         patch("backend.services.ocr_service._run_tesseract_ocr", return_value=["Tess line"]):
        result = _ocr_image_bytes(b"fake-image")
    assert "Tess line" in result


def test_ocr_image_bytes_returns_empty_when_both_fail():
    """When both engines fail, an empty list is returned (no exception raised)."""
    with patch("backend.services.ocr_service._run_paddle_ocr", side_effect=RuntimeError("no paddle")), \
         patch("backend.services.ocr_service._run_tesseract_ocr", side_effect=RuntimeError("no tess")):
        result = _ocr_image_bytes(b"fake-image")
    assert result == []


# ── run_ocr_background ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_ocr_background_sets_complete_on_success():
    """Req 3.2, 3.3 — successful OCR sets ocr_status='complete' with extracted_text."""
    mock_db, mock_col = _make_mock_db()

    with patch("backend.services.ocr_service._decrypt_file", return_value=b"fake-png-bytes"), \
         patch("backend.services.ocr_service._ocr_image_bytes", return_value=["Glucose: 5.5"]):
        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.png.enc", str(SAMPLE_USER_OID))

    # update_one should have been called at least twice: once for "processing", once for "complete"
    assert mock_col.update_one.call_count >= 2
    # Last call should set ocr_status="complete"
    last_call_args = mock_col.update_one.call_args_list[-1]
    set_fields = last_call_args[0][1]["$set"]
    assert set_fields["ocr_status"] == "complete"
    assert "Glucose: 5.5" in set_fields["extracted_text"]


@pytest.mark.asyncio
async def test_run_ocr_background_sets_failed_on_decrypt_error():
    """Req 3.2 — decryption failure sets ocr_status='failed'."""
    mock_db, mock_col = _make_mock_db()

    with patch("backend.services.ocr_service._decrypt_file", side_effect=Exception("bad key")):
        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.png.enc", str(SAMPLE_USER_OID))

    last_call_args = mock_col.update_one.call_args_list[-1]
    set_fields = last_call_args[0][1]["$set"]
    assert set_fields["ocr_status"] == "failed"


@pytest.mark.asyncio
async def test_run_ocr_background_zero_text_notification():
    """Req 3.6 — zero extracted text sets zero_text_notification=True and notification field."""
    mock_db, mock_col = _make_mock_db()

    with patch("backend.services.ocr_service._decrypt_file", return_value=b"fake-png-bytes"), \
         patch("backend.services.ocr_service._ocr_image_bytes", return_value=[]):
        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.png.enc", str(SAMPLE_USER_OID))

    last_call_args = mock_col.update_one.call_args_list[-1]
    set_fields = last_call_args[0][1]["$set"]
    assert set_fields["ocr_status"] == "complete"
    assert set_fields["zero_text_notification"] is True
    assert "notification" in set_fields
    assert len(set_fields["notification"]) > 0


@pytest.mark.asyncio
async def test_run_ocr_background_no_notification_when_text_extracted():
    """Req 3.6 — when text is extracted, zero_text_notification is False."""
    mock_db, mock_col = _make_mock_db()

    with patch("backend.services.ocr_service._decrypt_file", return_value=b"fake-png-bytes"), \
         patch("backend.services.ocr_service._ocr_image_bytes", return_value=["Some text"]):
        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.png.enc", str(SAMPLE_USER_OID))

    last_call_args = mock_col.update_one.call_args_list[-1]
    set_fields = last_call_args[0][1]["$set"]
    assert set_fields["zero_text_notification"] is False


@pytest.mark.asyncio
async def test_run_ocr_background_marks_processing_first():
    """OCR sets ocr_status='processing' before starting extraction."""
    mock_db, mock_col = _make_mock_db()

    with patch("backend.services.ocr_service._decrypt_file", return_value=b"fake-png-bytes"), \
         patch("backend.services.ocr_service._ocr_image_bytes", return_value=["text"]):
        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.png.enc", str(SAMPLE_USER_OID))

    first_call_args = mock_col.update_one.call_args_list[0]
    set_fields = first_call_args[0][1]["$set"]
    assert set_fields["ocr_status"] == "processing"


# ── get_ocr_result ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_ocr_result_returns_complete_document():
    """Returns extracted_text and lab_parameters for a completed job."""
    doc = {
        "_id": SAMPLE_OID,
        "user_id": SAMPLE_USER_OID,
        "filename": "report.pdf",
        "ocr_status": "complete",
        "extracted_text": "Glucose: 5.5",
        "lab_parameters": [],
        "zero_text_notification": False,
        "notification": None,
        "ocr_completed_at": datetime.utcnow(),
    }
    mock_db, _ = _make_mock_db(doc=doc)

    result = await get_ocr_result(mock_db, str(SAMPLE_OID), str(SAMPLE_USER_OID))

    assert result["ocr_status"] == "complete"
    assert result["extracted_text"] == "Glucose: 5.5"
    assert result["job_id"] == str(SAMPLE_OID)


@pytest.mark.asyncio
async def test_get_ocr_result_pending_returns_202():
    """Req 3.2 — pending job raises HTTP 202."""
    from fastapi import HTTPException
    doc = {
        "_id": SAMPLE_OID,
        "user_id": SAMPLE_USER_OID,
        "ocr_status": "pending",
    }
    mock_db, _ = _make_mock_db(doc=doc)

    with pytest.raises(HTTPException) as exc_info:
        await get_ocr_result(mock_db, str(SAMPLE_OID), str(SAMPLE_USER_OID))
    assert exc_info.value.status_code == 202


@pytest.mark.asyncio
async def test_get_ocr_result_processing_returns_202():
    """Req 3.2 — processing job raises HTTP 202."""
    from fastapi import HTTPException
    doc = {
        "_id": SAMPLE_OID,
        "user_id": SAMPLE_USER_OID,
        "ocr_status": "processing",
    }
    mock_db, _ = _make_mock_db(doc=doc)

    with pytest.raises(HTTPException) as exc_info:
        await get_ocr_result(mock_db, str(SAMPLE_OID), str(SAMPLE_USER_OID))
    assert exc_info.value.status_code == 202


@pytest.mark.asyncio
async def test_get_ocr_result_not_found_returns_404():
    """Returns HTTP 404 when document does not exist."""
    from fastapi import HTTPException
    mock_db, _ = _make_mock_db(doc=None)

    with pytest.raises(HTTPException) as exc_info:
        await get_ocr_result(mock_db, str(SAMPLE_OID), str(SAMPLE_USER_OID))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_ocr_result_invalid_job_id_returns_404():
    """Returns HTTP 404 for a non-ObjectId job_id."""
    from fastapi import HTTPException
    mock_db, _ = _make_mock_db()

    with pytest.raises(HTTPException) as exc_info:
        await get_ocr_result(mock_db, "not-an-object-id", str(SAMPLE_USER_OID))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_ocr_result_failed_job_returns_status():
    """A failed OCR job returns ocr_status='failed' without raising."""
    doc = {
        "_id": SAMPLE_OID,
        "user_id": SAMPLE_USER_OID,
        "filename": "report.pdf",
        "ocr_status": "failed",
        "extracted_text": "",
        "lab_parameters": [],
        "zero_text_notification": False,
        "notification": None,
        "ocr_completed_at": datetime.utcnow(),
    }
    mock_db, _ = _make_mock_db(doc=doc)

    result = await get_ocr_result(mock_db, str(SAMPLE_OID), str(SAMPLE_USER_OID))
    assert result["ocr_status"] == "failed"
