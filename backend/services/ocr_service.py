"""OCR pipeline service — file upload, encryption, status retrieval, and background OCR.

Requirements: 3.2, 3.3, 3.6
"""
import asyncio
import hashlib
import io
import logging
import os
import time
from datetime import datetime

from bson import ObjectId
from cryptography.fernet import Fernet
from fastapi import HTTPException, UploadFile, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OCR library imports — wrapped so tests can run without them
# ---------------------------------------------------------------------------
try:
    from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore
    _paddle_available = True
except Exception:
    _PaddleOCR = None
    _paddle_available = False

try:
    import pytesseract as _pytesseract  # type: ignore
    from PIL import Image as _PILImage  # type: ignore
    _tesseract_available = True
except Exception:
    _pytesseract = None
    _PILImage = None
    _tesseract_available = False

try:
    import fitz as _fitz  # PyMuPDF  # type: ignore
    _pymupdf_available = True
except Exception:
    _fitz = None
    _pymupdf_available = False

# pdfplumber — pure-Python PDF text extractor (no external binary needed)
try:
    import pdfplumber as _pdfplumber  # type: ignore
    _pdfplumber_available = True
except Exception:
    _pdfplumber = None
    _pdfplumber_available = False

# PyPDF2 fallback
try:
    from PyPDF2 import PdfReader as _PdfReader  # type: ignore
    _pypdf2_available = True
except Exception:
    _PdfReader = None
    _pypdf2_available = False

# Pillow for image handling
try:
    from PIL import Image as _PILImage  # type: ignore
    _pillow_available = True
except Exception:
    _PILImage = None
    _pillow_available = False
    _PdfReader = None
    _pypdf2_available = False

# Supported MIME types and their canonical extensions
ALLOWED_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

# Module-level Fernet instance — consistent within a process lifetime
_fernet_instance: Fernet | None = None


def _get_fernet() -> Fernet:
    """Return a stable Fernet instance for this process."""
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance
    key = settings.FILE_ENCRYPTION_KEY
    if key:
        try:
            _fernet_instance = Fernet(key.encode() if isinstance(key, str) else key)
            return _fernet_instance
        except Exception:
            pass
    # Generate once and reuse for the process lifetime
    _fernet_instance = Fernet(Fernet.generate_key())
    return _fernet_instance


async def upload_document(
    db: AsyncIOMotorDatabase,
    file: UploadFile,
    user_id: str,
) -> str:
    """Validate, encrypt, and persist an uploaded medical document.

    Returns the MongoDB document _id (job ID) as a string.

    Raises:
        HTTP 413 — file exceeds 20 MB
        HTTP 415 — unsupported content type
    """
    # ── Content-type validation ───────────────────────────────────────────────
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Accepted formats: PDF, JPG, PNG."
            ),
        )

    # ── Read file contents ────────────────────────────────────────────────────
    contents = await file.read()

    # ── Size validation ───────────────────────────────────────────────────────
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(contents) / (1024 * 1024):.1f} MB exceeds the "
                f"20 MB limit. Please upload a smaller file."
            ),
        )

    # ── Encrypt and write to disk ─────────────────────────────────────────────
    fernet = _get_fernet()
    encrypted = fernet.encrypt(contents)

    file_hash = hashlib.sha256(contents).hexdigest()
    ext = ALLOWED_CONTENT_TYPES[content_type]
    stored_filename = f"{file_hash}{ext}.enc"
    dest_path = os.path.join(settings.UPLOAD_DIR, stored_filename)

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(encrypted)

    # ── Persist Document record ───────────────────────────────────────────────
    doc = {
        "user_id": ObjectId(user_id),
        "filename": file.filename or stored_filename,
        "file_hash": file_hash,
        "file_size_bytes": len(contents),
        "upload_time": datetime.utcnow(),
        "ocr_status": "pending",
        "lab_parameters": [],
        "verified": False,
        "verified_at": None,
    }
    result = await db["documents"].insert_one(doc)
    return str(result.inserted_id)


async def get_document_status(
    db: AsyncIOMotorDatabase,
    job_id: str,
    user_id: str,
) -> dict:
    """Return the ocr_status for a document owned by the requesting user.

    Raises:
        HTTP 404 — document not found or not owned by user
    """
    if not ObjectId.is_valid(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    doc = await db["documents"].find_one(
        {"_id": ObjectId(job_id), "user_id": ObjectId(user_id)}
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return {
        "job_id": str(doc["_id"]),
        "ocr_status": doc["ocr_status"],
        "filename": doc.get("filename"),
        "upload_time": doc.get("upload_time"),
    }


# ---------------------------------------------------------------------------
# Background OCR processing
# ---------------------------------------------------------------------------

def _decrypt_file(file_path: str) -> bytes:
    """Decrypt an AES-256 (Fernet) encrypted file from disk."""
    fernet = _get_fernet()
    with open(file_path, "rb") as f:
        encrypted = f.read()
    return fernet.decrypt(encrypted)


def _run_paddle_ocr(image_bytes: bytes) -> list[str]:
    """Run PaddleOCR on raw image bytes. Returns list of text lines."""
    if not _paddle_available or _PaddleOCR is None:
        raise RuntimeError("PaddleOCR not available")
    ocr = _PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = ocr.ocr(np.array(img), cls=True)
    lines: list[str] = []
    if result:
        for block in result:
            if block:
                for line in block:
                    if line and len(line) >= 2 and line[1]:
                        lines.append(line[1][0])
    return lines


def _run_tesseract_ocr(image_bytes: bytes) -> list[str]:
    """Run pytesseract on raw image bytes. Returns list of text lines."""
    if not _tesseract_available or _pytesseract is None or _PILImage is None:
        raise RuntimeError("pytesseract not available")
    img = _PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    text = _pytesseract.image_to_string(img)
    return [line for line in text.splitlines() if line.strip()]


def _extract_images_from_pdf(pdf_bytes: bytes) -> list[bytes]:
    """Convert each PDF page to a PNG image. Returns list of PNG bytes."""
    if not _pymupdf_available or _fitz is None:
        raise RuntimeError("PyMuPDF not available")
    doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        images.append(pix.tobytes("png"))
    return images


def _extract_text_gemini_vision(image_bytes: bytes, mime_type: str = "image/jpeg") -> list[str]:
    """Use Gemini Vision to extract text from an image (lab report).
    
    Returns list of text lines. Falls back gracefully if API key not set.
    """
    if not settings.GEMINI_API_KEY:
        return []
    try:
        import google.generativeai as genai  # type: ignore
        import base64

        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = (
            "Extract ALL text from this medical lab report image. "
            "Return each line of text exactly as it appears, preserving numbers, units, and reference ranges. "
            "Focus on test names, result values, units, and reference intervals. "
            "Return plain text only, one item per line."
        )

        image_data = base64.b64encode(image_bytes).decode("utf-8")
        response = model.generate_content([
            prompt,
            {"mime_type": mime_type, "data": image_data}
        ])
        text = response.text or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        logger.info("Gemini Vision extracted %d lines", len(lines))
        return lines
    except Exception as exc:
        logger.warning("Gemini Vision OCR failed: %s", exc)
        return []


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    """Extract text from PDF using pdfplumber (pure Python, no binaries needed)."""
    lines: list[str] = []

    # Try pdfplumber first (best text extraction)
    if _pdfplumber_available and _pdfplumber:
        try:
            with _pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    for line in text.splitlines():
                        if line.strip():
                            lines.append(line.strip())
            if lines:
                return lines
        except Exception as exc:
            logger.debug("pdfplumber failed: %s", exc)

    # Fallback to PyPDF2
    if _pypdf2_available and _PdfReader:
        try:
            reader = _PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    if line.strip():
                        lines.append(line.strip())
            if lines:
                return lines
        except Exception as exc:
            logger.debug("PyPDF2 failed: %s", exc)

    return lines


def _merge_deduplicate(paddle_lines: list[str], tess_lines: list[str]) -> list[str]:
    """Merge two lists of text lines, deduplicating identical lines."""
    seen: set[str] = set()
    merged: list[str] = []
    for line in paddle_lines + tess_lines:
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            merged.append(stripped)
    return merged


def _ocr_image_bytes(image_bytes: bytes) -> list[str]:
    """Try PaddleOCR first; fall back to Tesseract for plain-text regions."""
    paddle_lines: list[str] = []
    tess_lines: list[str] = []

    try:
        paddle_lines = _run_paddle_ocr(image_bytes)
    except Exception as exc:
        logger.debug("PaddleOCR failed: %s", exc)

    try:
        tess_lines = _run_tesseract_ocr(image_bytes)
    except Exception as exc:
        logger.debug("Tesseract failed: %s", exc)

    return _merge_deduplicate(paddle_lines, tess_lines)


async def run_ocr_background(
    db: AsyncIOMotorDatabase,
    job_id: str,
    file_path: str,
    user_id: str,
) -> None:
    """Background task: decrypt file, run OCR, update Document record.

    - Tries PaddleOCR first (better table detection), falls back to Tesseract.
    - Merges and deduplicates results.
    - Must complete within 30 seconds.
    - Sets ocr_status to "complete" or "failed".
    - Sets zero_text_notification=True when no text is extracted.

    Requirements: 3.2, 3.3, 3.6
    """
    start = time.monotonic()
    deadline = 30.0  # seconds

    async def _update(fields: dict) -> None:
        await db["documents"].update_one(
            {"_id": ObjectId(job_id)},
            {"$set": fields},
        )

    # Mark as processing
    await _update({"ocr_status": "processing"})

    try:
        # Decrypt file (run in thread to avoid blocking event loop)
        raw_bytes: bytes = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _decrypt_file, file_path),
            timeout=max(1.0, deadline - (time.monotonic() - start)),
        )

        # Determine if PDF or image
        is_pdf = file_path.lower().endswith(".pdf.enc") or file_path.lower().endswith(".pdf")

        all_lines: list[str] = []

        if is_pdf:
            # Try fast pure-Python PDF extraction first (pdfplumber/PyPDF2)
            fast_lines = _extract_text_from_pdf_bytes(raw_bytes)
            if fast_lines:
                all_lines = fast_lines
                logger.info("PDF text extracted via pdfplumber for job %s", job_id)
            else:
                # Fall back to image-based OCR
                try:
                    page_images: list[bytes] = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, _extract_images_from_pdf, raw_bytes
                        ),
                        timeout=max(1.0, deadline - (time.monotonic() - start)),
                    )
                except Exception as exc:
                    logger.warning("PDF→image conversion failed: %s.", exc)
                    page_images = []

                for page_bytes in page_images:
                    remaining = deadline - (time.monotonic() - start)
                    if remaining <= 0:
                        break
                    try:
                        lines = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, _ocr_image_bytes, page_bytes
                            ),
                            timeout=remaining,
                        )
                        all_lines.extend(lines)
                    except asyncio.TimeoutError:
                        logger.warning("OCR timed out on a PDF page for job %s", job_id)
                        break
                    except Exception as exc:
                        logger.warning("OCR error on PDF page for job %s: %s", job_id, exc)
        else:
            # Image file — try Gemini Vision first (fast, no local install needed)
            remaining = deadline - (time.monotonic() - start)

            # Detect mime type from file extension
            if file_path.lower().endswith(".png.enc") or file_path.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"

            # Try Gemini Vision (best for scanned lab reports)
            gemini_lines = _extract_text_gemini_vision(raw_bytes, mime_type)
            if gemini_lines:
                all_lines = gemini_lines
            else:
                # Fall back to local OCR engines
                try:
                    all_lines = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, _ocr_image_bytes, raw_bytes
                        ),
                        timeout=max(1.0, remaining),
                    )
                except asyncio.TimeoutError:
                    logger.warning("OCR timed out for job %s", job_id)

        extracted_text = "\n".join(all_lines)
        zero_text = len(extracted_text.strip()) == 0

        # Run document intelligence to extract structured lab parameters
        lab_parameters: list[dict] = []
        if not zero_text:
            try:
                from backend.services.document_intelligence import extract_lab_parameters
                params = extract_lab_parameters(extracted_text)
                lab_parameters = [p.model_dump() for p in params]
                logger.info("Extracted %d lab parameters for job %s", len(lab_parameters), job_id)
            except Exception as exc:
                logger.warning("Lab parameter extraction failed for job %s: %s", job_id, exc)

        update_fields: dict = {
            "ocr_status": "complete",
            "extracted_text": extracted_text,
            "lab_parameters": lab_parameters,
            "ocr_completed_at": datetime.utcnow(),
            "zero_text_notification": zero_text,
        }
        if zero_text:
            # Requirement 3.6: notify patient on zero-text extraction
            update_fields["notification"] = (
                "No text could be extracted from your document. "
                "Please verify your lab values manually."
            )

        await _update(update_fields)
        logger.info("OCR complete for job %s (%.1fs)", job_id, time.monotonic() - start)

    except Exception as exc:
        logger.exception("OCR failed for job %s: %s", job_id, exc)
        await _update({
            "ocr_status": "failed",
            "ocr_error": str(exc),
            "ocr_completed_at": datetime.utcnow(),
        })


async def get_ocr_result(
    db: AsyncIOMotorDatabase,
    job_id: str,
    user_id: str,
) -> dict:
    """Return the OCR result (extracted text + raw lab parameter candidates).

    Raises:
        HTTP 404 — document not found or not owned by user
        HTTP 202 — OCR still in progress
    """
    if not ObjectId.is_valid(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    doc = await db["documents"].find_one(
        {"_id": ObjectId(job_id), "user_id": ObjectId(user_id)}
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    ocr_status = doc.get("ocr_status", "pending")
    if ocr_status in ("pending", "processing"):
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"OCR is still {ocr_status}. Poll /api/ocr/status/{job_id}.",
        )

    return {
        "job_id": str(doc["_id"]),
        "ocr_status": ocr_status,
        "filename": doc.get("filename"),
        "extracted_text": doc.get("extracted_text", ""),
        "lab_parameters": doc.get("lab_parameters", []),
        "zero_text_notification": doc.get("zero_text_notification", False),
        "notification": doc.get("notification"),
        "ocr_completed_at": doc.get("ocr_completed_at"),
    }
