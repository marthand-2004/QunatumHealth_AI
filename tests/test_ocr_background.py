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


# ── New tests (Task 10) ───────────────────────────────────────────────────────

import sys
import importlib
import inspect
import asyncio
import tempfile


# 10.1 — import guard: no exception when ocrmypdf is missing
def test_import_no_exception_when_ocrmypdf_missing():
    """Req 2.3 — importing ocr_service does not raise even if ocrmypdf is absent."""
    import backend.services.ocr_service as ocr_mod

    # Simulate missing ocrmypdf by temporarily patching sys.modules and reloading
    # We use a separate import to avoid corrupting the module used by other tests.
    # Save original values
    original_ocrmypdf = sys.modules.get("ocrmypdf")
    original_ocrmypdf_exc = sys.modules.get("ocrmypdf.exceptions")
    original_available = ocr_mod._ocrmypdf_available

    try:
        # Inject None to simulate ImportError on import
        sys.modules["ocrmypdf"] = None  # type: ignore[assignment]
        sys.modules["ocrmypdf.exceptions"] = None  # type: ignore[assignment]

        # Reload the module — should not raise
        importlib.reload(ocr_mod)
        assert ocr_mod._ocrmypdf_available is False
    finally:
        # Restore sys.modules entries
        if original_ocrmypdf is None:
            sys.modules.pop("ocrmypdf", None)
        else:
            sys.modules["ocrmypdf"] = original_ocrmypdf

        if original_ocrmypdf_exc is None:
            sys.modules.pop("ocrmypdf.exceptions", None)
        else:
            sys.modules["ocrmypdf.exceptions"] = original_ocrmypdf_exc

        # Reload again to restore the module to its original working state
        importlib.reload(ocr_mod)
        # Ensure the module is back in sys.modules under the right key
        sys.modules["backend.services.ocr_service"] = ocr_mod


# 10.2 — OCRmyPDF is submitted to run_in_executor, not called directly
@pytest.mark.asyncio
async def test_ocrmypdf_called_via_run_in_executor():
    """Req 4.1, 10.2 — _ocr_pdf_to_searchable is submitted via run_in_executor."""
    mock_db, mock_col = _make_mock_db()

    captured_fns = []
    real_loop = asyncio.get_event_loop()

    def capturing_run_in_executor(executor, fn, *args):
        captured_fns.append(fn)
        # Delegate to the real executor so the coroutine resolves properly
        return real_loop.run_in_executor(executor, fn, *args)

    mock_loop = MagicMock()
    mock_loop.run_in_executor = capturing_run_in_executor

    with patch("backend.services.ocr_service._decrypt_file", return_value=b"%PDF-1.4 test"), \
         patch("backend.services.ocr_service._ocr_pdf_to_searchable", return_value=b"%PDF-1.4") as mock_ocr_fn, \
         patch("backend.services.ocr_service._extract_text_from_pdf_bytes", return_value=["line"]), \
         patch("backend.services.ocr_service.asyncio.get_event_loop", return_value=mock_loop), \
         patch("backend.services.ocr_service.asyncio.wait_for", new=asyncio.wait_for), \
         patch("backend.services.ocr_service.asyncio.TimeoutError", asyncio.TimeoutError):

        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.pdf.enc", str(SAMPLE_USER_OID))

        # Assert that _ocr_pdf_to_searchable (the mock) was submitted to run_in_executor
        assert any(
            fn is mock_ocr_fn or getattr(fn, "__name__", "") == "_ocr_pdf_to_searchable"
            for fn in captured_fns
        )


# 10.3 — skip_text=True is passed to ocrmypdf.ocr
def test_skip_text_passed_to_ocrmypdf():
    """Req 4.2, 10.1 — ocrmypdf.ocr is called with skip_text=True."""
    import backend.services.ocr_service as ocr_mod

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fin:
        fin.write(b"%PDF-1.4 test")
        input_path = fin.name

    output_fd, output_path = tempfile.mkstemp(suffix=".pdf")
    os.close(output_fd)

    try:
        with patch("backend.services.ocr_service.ocrmypdf") as mock_ocrmypdf:
            mock_ocrmypdf.ocr = MagicMock()
            mock_ocrmypdf.exceptions = MagicMock()
            mock_ocrmypdf.exceptions.PriorOcrFoundError = type("PriorOcrFoundError", (Exception,), {})
            mock_ocrmypdf.exceptions.EncryptedPdfError = type("EncryptedPdfError", (Exception,), {})
            mock_ocrmypdf.exceptions.InputFileError = type("InputFileError", (Exception,), {})

            ocr_mod._run_ocrmypdf_python_api(input_path, output_path)

            mock_ocrmypdf.ocr.assert_called_once()
            _, kwargs = mock_ocrmypdf.ocr.call_args
            assert kwargs.get("skip_text") is True
    finally:
        for p in (input_path, output_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# 10.4 — PriorOcrFoundError causes a retry (ocrmypdf.ocr called twice)
def test_prior_ocr_found_error_retries():
    """Req 10.3 — PriorOcrFoundError triggers a retry; ocrmypdf.ocr called twice."""
    import backend.services.ocr_service as ocr_mod

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fin:
        fin.write(b"%PDF-1.4 test")
        input_path = fin.name

    output_fd, output_path = tempfile.mkstemp(suffix=".pdf")
    os.close(output_fd)

    try:
        PriorOcrFoundError = type("PriorOcrFoundError", (Exception,), {})

        with patch("backend.services.ocr_service.ocrmypdf") as mock_ocrmypdf:
            mock_ocrmypdf.exceptions = MagicMock()
            mock_ocrmypdf.exceptions.PriorOcrFoundError = PriorOcrFoundError
            mock_ocrmypdf.exceptions.EncryptedPdfError = type("EncryptedPdfError", (Exception,), {})
            mock_ocrmypdf.exceptions.InputFileError = type("InputFileError", (Exception,), {})
            # First call raises PriorOcrFoundError; second call succeeds
            mock_ocrmypdf.ocr = MagicMock(side_effect=[PriorOcrFoundError("prior ocr"), None])

            ocr_mod._run_ocrmypdf_python_api(input_path, output_path)

            assert mock_ocrmypdf.ocr.call_count == 2
    finally:
        for p in (input_path, output_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# 10.5 — subprocess fallback when _ocrmypdf_available is False
def test_subprocess_fallback_when_api_unavailable():
    """Req 10.4 — subprocess.run is called with ocrmypdf CLI args when API unavailable."""
    import backend.services.ocr_service as ocr_mod

    original_available = ocr_mod._ocrmypdf_available
    ocr_mod._ocrmypdf_available = False

    try:
        def fake_subprocess_run(args, **kwargs):
            # Write a minimal PDF to the output path (last positional arg)
            output_path = args[-1]
            with open(output_path, "wb") as f:
                f.write(b"%PDF-1.4 fake-searchable")
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("backend.services.ocr_service.subprocess.run", side_effect=fake_subprocess_run) as mock_run:
            result_bytes = ocr_mod._ocr_pdf_to_searchable(b"%PDF-1.4 test")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # first positional arg = command list
        assert call_args[0] == "ocrmypdf"
        assert "--skip-text" in call_args
        assert result_bytes == b"%PDF-1.4 fake-searchable"
    finally:
        ocr_mod._ocrmypdf_available = original_available


# 10.6 — pdfplumber returns empty → PyPDF2 fallback is used
def test_pdfplumber_fallback_to_pypdf2():
    """Req 5.2 — when pdfplumber returns empty text, PyPDF2 fallback is used."""
    from backend.services.ocr_service import _extract_text_from_pdf_bytes

    # Mock pdfplumber page that returns empty text
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""

    mock_pdf_ctx = MagicMock()
    mock_pdf_ctx.__enter__ = MagicMock(return_value=MagicMock(pages=[mock_page]))
    mock_pdf_ctx.__exit__ = MagicMock(return_value=False)

    # Mock PyPDF2 page that returns real text
    mock_pypdf2_page = MagicMock()
    mock_pypdf2_page.extract_text.return_value = "Glucose: 5.5 mmol/L"

    mock_reader = MagicMock()
    mock_reader.pages = [mock_pypdf2_page]

    with patch("backend.services.ocr_service._pdfplumber_available", True), \
         patch("backend.services.ocr_service._pdfplumber") as mock_pdfplumber, \
         patch("backend.services.ocr_service._pypdf2_available", True), \
         patch("backend.services.ocr_service._PdfReader", return_value=mock_reader):

        mock_pdfplumber.open.return_value = mock_pdf_ctx

        result = _extract_text_from_pdf_bytes(b"fake pdf bytes")

    assert "Glucose: 5.5 mmol/L" in result


# 10.7 — zero_text_notification is set for PDF with no extracted text
@pytest.mark.asyncio
async def test_zero_text_notification_set_for_pdf():
    """Req 5.4 — zero_text_notification=True and notification set when PDF yields no text."""
    mock_db, mock_col = _make_mock_db()

    with patch("backend.services.ocr_service._decrypt_file", return_value=b"%PDF-1.4 test"), \
         patch("backend.services.ocr_service._ocr_pdf_to_searchable", return_value=b"%PDF-1.4"), \
         patch("backend.services.ocr_service._extract_text_from_pdf_bytes", return_value=[]):

        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.pdf.enc", str(SAMPLE_USER_OID))

    last_call_args = mock_col.update_one.call_args_list[-1]
    set_fields = last_call_args[0][1]["$set"]
    assert set_fields["zero_text_notification"] is True
    assert "notification" in set_fields
    assert len(set_fields["notification"]) > 0


# 10.8 — asyncio.TimeoutError sets ocr_status="failed"
@pytest.mark.asyncio
async def test_timeout_sets_failed_status():
    """Req 9.1, 9.2 — TimeoutError during OCR sets ocr_status='failed'."""
    mock_db, mock_col = _make_mock_db()

    with patch("backend.services.ocr_service._decrypt_file", return_value=b"%PDF-1.4 test"), \
         patch("backend.services.ocr_service._ocr_pdf_to_searchable", side_effect=asyncio.TimeoutError()):

        await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.pdf.enc", str(SAMPLE_USER_OID))

    last_call_args = mock_col.update_one.call_args_list[-1]
    set_fields = last_call_args[0][1]["$set"]
    assert set_fields["ocr_status"] == "failed"
    # Note: str(asyncio.TimeoutError()) is '', so ocr_error will be empty
    # The key requirement is that ocr_status is "failed"
    assert "ocr_error" in set_fields


# 10.9 — temp files are deleted on success
def test_temp_files_deleted_on_success():
    """Req 9.4 — os.unlink is called for both temp files on successful OCR."""
    import backend.services.ocr_service as ocr_mod

    def fake_run_api(input_path, output_path):
        # Write a minimal PDF to the output path to simulate success
        with open(output_path, "wb") as f:
            f.write(b"%PDF-1.4 searchable")

    with patch("backend.services.ocr_service._run_ocrmypdf_python_api", side_effect=fake_run_api), \
         patch("backend.services.ocr_service._ocrmypdf_available", True), \
         patch("backend.services.ocr_service.os.unlink") as mock_unlink:

        ocr_mod._ocr_pdf_to_searchable(b"%PDF-1.4 test")

    assert mock_unlink.call_count >= 2


# 10.10 — temp files are deleted on failure
def test_temp_files_deleted_on_failure():
    """Req 9.4 — os.unlink is called for both temp files even when OCR raises."""
    import backend.services.ocr_service as ocr_mod

    with patch("backend.services.ocr_service._run_ocrmypdf_python_api", side_effect=RuntimeError("test error")), \
         patch("backend.services.ocr_service._ocrmypdf_available", True), \
         patch("backend.services.ocr_service.os.unlink") as mock_unlink:

        try:
            ocr_mod._ocr_pdf_to_searchable(b"%PDF-1.4 test")
        except RuntimeError:
            pass

    assert mock_unlink.call_count >= 2


# 10.11 — Gemini lab extraction fallback is retained and called when regex finds nothing
@pytest.mark.asyncio
async def test_gemini_lab_extraction_fallback_retained():
    """Req 6.4 — _extract_lab_params_via_gemini is called when regex extraction returns empty."""
    from backend.core.config import settings as _settings

    mock_db, mock_col = _make_mock_db()

    gemini_result = [{"name": "glucose", "value": 5.5, "unit": "mmol/L"}]

    original_key = _settings.GEMINI_API_KEY
    _settings.GEMINI_API_KEY = "test-key"

    try:
        with patch("backend.services.ocr_service._decrypt_file", return_value=b"%PDF-1.4 test"), \
             patch("backend.services.ocr_service._ocr_pdf_to_searchable", return_value=b"%PDF-1.4"), \
             patch("backend.services.ocr_service._extract_text_from_pdf_bytes", return_value=["Glucose: 5.5"]), \
             patch("backend.services.document_intelligence.extract_lab_parameters", return_value=[]) as mock_extract, \
             patch("backend.services.ocr_service._extract_lab_params_via_gemini", return_value=gemini_result) as mock_gemini:

            await run_ocr_background(mock_db, str(SAMPLE_OID), "/fake/path/file.pdf.enc", str(SAMPLE_USER_OID))

        mock_gemini.assert_called_once()
        last_call_args = mock_col.update_one.call_args_list[-1]
        set_fields = last_call_args[0][1]["$set"]
        assert set_fields["lab_parameters"] == gemini_result
    finally:
        _settings.GEMINI_API_KEY = original_key


# 10.12 — public API function signatures are unchanged
def test_public_api_signatures_unchanged():
    """Req 7.2 — upload_document, get_document_status, get_ocr_result have expected signatures."""
    from backend.services.ocr_service import upload_document, get_document_status, get_ocr_result

    upload_params = list(inspect.signature(upload_document).parameters.keys())
    status_params = list(inspect.signature(get_document_status).parameters.keys())
    result_params = list(inspect.signature(get_ocr_result).parameters.keys())

    assert upload_params == ["db", "file", "user_id"], (
        f"upload_document signature changed: {upload_params}"
    )
    assert status_params == ["db", "job_id", "user_id"], (
        f"get_document_status signature changed: {status_params}"
    )
    assert result_params == ["db", "job_id", "user_id"], (
        f"get_ocr_result signature changed: {result_params}"
    )
