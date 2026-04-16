"""PDF report generation service.

Uses ReportLab for PDF layout. If ReportLab is not installed, falls back to
a minimal plain-bytes PDF so the rest of the system can still be tested.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""
from __future__ import annotations

import io
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── ReportLab optional import ─────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed — using minimal fallback PDF generator.")


# ── Fallback PDF ──────────────────────────────────────────────────────────────

def _minimal_pdf(message: str = "QuantumHealthAI Report") -> bytes:
    """Return a minimal valid PDF containing *message* as plain text."""
    content = (
        "%PDF-1.4\n"
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        "   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        f"4 0 obj\n<< /Length {len(message) + 50} >>\nstream\n"
        f"BT /F1 12 Tf 72 720 Td ({message}) Tj ET\nendstream\nendobj\n"
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        "xref\n0 6\n0000000000 65535 f \n"
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    )
    return content.encode("latin-1")


# ── PDF generation ────────────────────────────────────────────────────────────

def _build_pdf_reportlab(
    patient: dict,
    prediction: dict,
    lab_parameters: list[dict],
    recommendations: list[dict],
) -> bytes:
    """Build a formatted PDF using ReportLab and return the bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceAfter=8,
        spaceBefore=14,
    )
    normal = styles["Normal"]

    story: list[Any] = []

    # ── Title ─────────────────────────────────────────────────────────────
    story.append(Paragraph("QuantumHealthAI — Health Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.3 * cm))

    # ── Patient info ──────────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ["Patient ID", str(patient.get("_id", patient.get("id", "N/A")))],
        ["Email", patient.get("email", "N/A")],
        ["Report Generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
    ]
    pt = Table(patient_data, colWidths=[5 * cm, 12 * cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.3 * cm))

    # ── Prediction metadata ───────────────────────────────────────────────
    story.append(Paragraph("Prediction Details", heading_style))
    ts = prediction.get("timestamp")
    ts_str = ts.strftime("%Y-%m-%d %H:%M UTC") if isinstance(ts, datetime) else str(ts or "N/A")
    pred_data = [
        ["Model Used", prediction.get("model_used", "N/A").capitalize()],
        ["Prediction Timestamp", ts_str],
    ]
    pred_table = Table(pred_data, colWidths=[5 * cm, 12 * cm])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.3 * cm))

    # ── Risk scores ───────────────────────────────────────────────────────
    story.append(Paragraph("Disease Risk Scores", heading_style))
    risk_scores: dict = prediction.get("risk_scores", {})
    risk_data = [["Disease", "Risk Score (%)", "Level"]]
    for disease, score in risk_scores.items():
        level = "Low" if score < 30 else ("Moderate" if score < 75 else "High")
        risk_data.append([disease.capitalize(), f"{score:.1f}%", level])
    if len(risk_data) == 1:
        risk_data.append(["N/A", "N/A", "N/A"])
    rt = Table(risk_data, colWidths=[5 * cm, 5 * cm, 7 * cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
    ]))
    story.append(rt)
    story.append(Spacer(1, 0.3 * cm))

    # ── Lab values ────────────────────────────────────────────────────────
    if lab_parameters:
        story.append(Paragraph("Lab Values", heading_style))
        lab_data = [["Parameter", "Value", "Unit", "Abnormal"]]
        for lp in lab_parameters:
            lab_data.append([
                lp.get("name", ""),
                str(lp.get("value", "")),
                lp.get("unit", ""),
                "Yes" if lp.get("is_abnormal") else "No",
            ])
        lt = Table(lab_data, colWidths=[5 * cm, 4 * cm, 4 * cm, 4 * cm])
        lt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
        ]))
        story.append(lt)
        story.append(Spacer(1, 0.3 * cm))

    # ── SHAP summary ──────────────────────────────────────────────────────
    shap_values: dict = prediction.get("shap_values") or {}
    if shap_values:
        story.append(Paragraph("SHAP Feature Importance (Top Factors)", heading_style))
        shap_data = [["Feature", "SHAP Value (avg abs)"]]
        for disease, vals in shap_values.items():
            if isinstance(vals, list) and vals:
                avg = sum(abs(v) for v in vals) / len(vals)
                shap_data.append([f"{disease} — avg", f"{avg:.4f}"])
        if len(shap_data) > 1:
            st = Table(shap_data, colWidths=[9 * cm, 8 * cm])
            st.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(st)
            story.append(Spacer(1, 0.3 * cm))

    # ── Recommendations ───────────────────────────────────────────────────
    if recommendations:
        story.append(Paragraph("Recommendations", heading_style))
        for rec in recommendations:
            text = rec.get("text", "")
            disease = rec.get("disease", "")
            physician = " ⚕ Consult physician." if rec.get("requires_physician") else ""
            story.append(Paragraph(f"<b>[{disease.capitalize()}]</b> {text}{physician}", normal))
            story.append(Spacer(1, 0.15 * cm))

    # ── Disclaimer ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    disclaimer = (
        "<i>This report is generated for informational purposes only and does not "
        "constitute medical advice. Please consult a licensed physician for diagnosis "
        "and treatment decisions.</i>"
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disclaimer", parent=normal, fontSize=8, textColor=colors.grey)))

    doc.build(story)
    return buf.getvalue()


def generate_pdf(
    patient: dict,
    prediction: dict,
    lab_parameters: list[dict],
    recommendations: list[dict],
) -> bytes:
    """Generate a PDF report and return the raw bytes.

    Uses ReportLab when available; falls back to a minimal PDF otherwise.
    Requirement 11.1, 11.2
    """
    if _REPORTLAB_AVAILABLE:
        try:
            return _build_pdf_reportlab(patient, prediction, lab_parameters, recommendations)
        except Exception as exc:  # pragma: no cover
            logger.error("ReportLab PDF generation failed: %s", exc)
            return _minimal_pdf(f"QuantumHealthAI Report — generation error: {exc}")
    return _minimal_pdf("QuantumHealthAI Report (ReportLab not installed)")


# ── Persistence helpers ───────────────────────────────────────────────────────

def _reports_dir() -> str:
    """Return (and create if needed) the directory for storing PDF files."""
    path = os.path.join(settings.UPLOAD_DIR, "reports")
    os.makedirs(path, exist_ok=True)
    return path


async def create_report(
    db: AsyncIOMotorDatabase,
    patient_id: str,
    generated_by: str,
    prediction_id: str,
    pdf_bytes: bytes,
) -> str:
    """Save PDF to disk and create a Report document in MongoDB.

    Returns the new report_id as a string.
    Requirement 11.1, 11.4
    """
    report_id = str(ObjectId())
    filename = f"report_{report_id}.pdf"
    file_path = os.path.join(_reports_dir(), filename)

    with open(file_path, "wb") as fh:
        fh.write(pdf_bytes)

    doc = {
        "_id": ObjectId(report_id),
        "patient_id": patient_id,
        "generated_by": generated_by,
        "prediction_id": prediction_id,
        "file_path": file_path,
        "share_token": None,
        "share_expires_at": None,
        "created_at": datetime.utcnow(),
    }
    await db["reports"].insert_one(doc)
    return report_id


async def generate_share_token(report_id: str, db: AsyncIOMotorDatabase) -> dict:
    """Create a 72-hour share token for *report_id* and persist it.

    Returns a dict with ``share_token`` and ``share_expires_at``.
    Requirement 11.3, 11.5
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=72)

    result = await db["reports"].update_one(
        {"_id": ObjectId(report_id)},
        {"$set": {"share_token": token, "share_expires_at": expires_at}},
    )
    if result.matched_count == 0:
        raise ValueError(f"Report {report_id} not found.")

    return {"share_token": token, "share_expires_at": expires_at}
