"""Recommendation engine service.

Two-stage pipeline:
1. Rule-based engine applies clinical threshold rules.
2. LLM enrichment adds lifestyle-contextual recommendations.
3. SHAP-based prioritization sorts by feature magnitude.
4. Physician referral is always included when any risk score > 75.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.core.config import settings
from backend.models.lifestyle import LifestyleProfile
from backend.models.recommendation import Recommendation


# Internal dataclass used during pipeline to carry SHAP magnitude alongside rec data
@dataclass
class _RecCandidate:
    disease: str
    text: str
    source: str          # "rule" | "llm"
    requires_physician: bool = False
    magnitude: float = 0.0  # SHAP magnitude for sorting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature index mapping (matches FEATURE_NAMES in prediction.py)
# ---------------------------------------------------------------------------

_FEAT = {
    "glucose": 0,
    "hba1c": 1,
    "creatinine": 2,
    "cholesterol": 3,
    "triglycerides": 4,
    "hemoglobin": 5,
    "bmi": 6,
    "age": 7,
    "systolic_bp": 8,
    "diastolic_bp": 9,
    "smoking_encoded": 10,
    "exercise_frequency": 11,
    "sleep_hours": 12,
    "stress_level": 13,
}

# ---------------------------------------------------------------------------
# Clinical threshold rules
# Each rule: (disease, feature_key, operator, threshold, recommendation_text)
# operator: "gt" | "lt"
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, str, str, float, str]] = [
    # Diabetes
    ("diabetes", "hba1c",    "gt", 6.5,  "Consult endocrinologist for diabetes management"),
    ("diabetes", "glucose",  "gt", 7.0,  "Monitor fasting blood glucose daily"),
    # CVD
    ("cvd", "cholesterol",   "gt", 5.2,  "Adopt heart-healthy diet, reduce saturated fats"),
    ("cvd", "systolic_bp",   "gt", 140.0, "Monitor blood pressure daily, reduce sodium intake"),
    # CKD
    ("ckd", "creatinine",    "gt", 106.0, "Increase water intake, reduce protein consumption"),
    # General (applied to all diseases)
    ("*", "bmi",             "gt", 30.0,  "Engage in regular physical activity, consult nutritionist"),
    ("*", "stress_level",    "gt", 7.0,   "Practice stress management techniques"),
    ("*", "exercise_frequency", "lt", 2.0, "Aim for 30 minutes of moderate exercise daily"),
]

_DISEASES = ["diabetes", "cvd", "ckd"]

# Minimum recommendations per high-risk disease (Requirement 9.1)
_MIN_RECS = 3

# Physician referral threshold (Requirement 9.5)
_PHYSICIAN_THRESHOLD = 75.0

# Risk threshold to generate recommendations (Requirement 9.1)
_RISK_THRESHOLD = 30.0

# ---------------------------------------------------------------------------
# Fallback template recommendations per disease (used when rule count < 3)
# ---------------------------------------------------------------------------

_FALLBACK_RECS: dict[str, list[str]] = {
    "diabetes": [
        "Maintain a balanced diet low in refined carbohydrates and sugars",
        "Schedule regular HbA1c monitoring every 3 months",
        "Aim for at least 150 minutes of moderate aerobic activity per week",
        "Discuss medication adherence with your healthcare provider",
    ],
    "cvd": [
        "Follow a Mediterranean-style diet rich in fruits, vegetables, and whole grains",
        "Avoid smoking and limit alcohol consumption",
        "Maintain a healthy weight through regular exercise",
        "Schedule a lipid panel check every 6 months",
    ],
    "ckd": [
        "Limit sodium intake to less than 2 g per day",
        "Monitor blood pressure regularly and keep it below 130/80 mmHg",
        "Avoid nephrotoxic medications such as NSAIDs without physician guidance",
        "Schedule regular kidney function tests (eGFR, urine albumin)",
    ],
}


# ---------------------------------------------------------------------------
# Rule-based engine
# ---------------------------------------------------------------------------


def _apply_rules(
    disease: str,
    feature_vector: list[float],
    shap_values: Optional[list[float]],
) -> list[_RecCandidate]:
    """Apply clinical threshold rules for a single disease.

    Returns a list of _RecCandidate objects (source="rule").
    Magnitude is the absolute SHAP value of the triggering feature.
    """
    recs: list[_RecCandidate] = []

    for rule_disease, feat_key, op, threshold, text in _RULES:
        if rule_disease not in (disease, "*"):
            continue

        feat_idx = _FEAT.get(feat_key)
        if feat_idx is None or feat_idx >= len(feature_vector):
            continue

        value = feature_vector[feat_idx]
        triggered = (op == "gt" and value > threshold) or (op == "lt" and value < threshold)
        if not triggered:
            continue

        magnitude = abs(shap_values[feat_idx]) if shap_values and feat_idx < len(shap_values) else 0.0

        recs.append(_RecCandidate(
            disease=disease,
            text=text,
            source="rule",
            requires_physician=False,
            magnitude=magnitude,
        ))

    return recs


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------


def _build_llm_prompt(
    disease: str,
    risk_score: float,
    feature_vector: list[float],
    lifestyle: Optional[LifestyleProfile],
    existing_texts: list[str],
) -> str:
    feat_lines = "\n".join(
        f"  - {name}: {feature_vector[idx]:.2f}"
        for name, idx in _FEAT.items()
        if idx < len(feature_vector)
    )
    lifestyle_lines = ""
    if lifestyle:
        lifestyle_lines = (
            f"  - BMI: {lifestyle.bmi}\n"
            f"  - Smoking: {lifestyle.smoking_status}\n"
            f"  - Alcohol: {lifestyle.alcohol_frequency}\n"
            f"  - Exercise: {lifestyle.exercise_frequency} days/week\n"
            f"  - Diet: {lifestyle.diet_type}\n"
            f"  - Sleep: {lifestyle.sleep_hours} hours/night\n"
            f"  - Stress level: {lifestyle.stress_level}/10\n"
            f"  - Medications: {', '.join(lifestyle.medications) or 'none'}"
        )

    existing = "\n".join(f"  - {t}" for t in existing_texts) if existing_texts else "  (none)"

    return (
        "You are a clinical health advisor. Generate 2 concise, actionable health "
        f"recommendations for a patient with a {disease.upper()} risk score of {risk_score:.1f}%. "
        "Incorporate the patient's lifestyle context below. "
        "Do NOT repeat the existing recommendations. "
        "Return ONLY a JSON array of strings, e.g. [\"rec1\", \"rec2\"].\n\n"
        f"Lab values:\n{feat_lines}\n\n"
        f"Lifestyle profile:\n{lifestyle_lines}\n\n"
        f"Existing recommendations (do not repeat):\n{existing}"
    )


def _call_gemini_recs(prompt: str) -> list[str]:
    import json
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _call_openai_recs(prompt: str) -> list[str]:
    import json
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)
    # Handle both {"recommendations": [...]} and plain list
    if isinstance(data, list):
        return data
    for key in ("recommendations", "items", "results"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return list(data.values())[0] if data else []


def _llm_enrich(
    disease: str,
    risk_score: float,
    feature_vector: list[float],
    lifestyle: Optional[LifestyleProfile],
    existing_texts: list[str],
) -> list[_RecCandidate]:
    """Generate LLM-based recommendations. Falls back to template on failure."""
    prompt = _build_llm_prompt(disease, risk_score, feature_vector, lifestyle, existing_texts)

    llm_texts: list[str] = []

    if settings.GEMINI_API_KEY:
        try:
            llm_texts = _call_gemini_recs(prompt)
        except Exception as exc:
            logger.warning("Gemini recommendation failed: %s", exc)

    if not llm_texts and settings.OPENAI_API_KEY:
        try:
            llm_texts = _call_openai_recs(prompt)
        except Exception as exc:
            logger.warning("OpenAI recommendation failed: %s", exc)

    if not llm_texts:
        # Template fallback: pick unused fallback recs
        fallbacks = _FALLBACK_RECS.get(disease, [])
        llm_texts = [t for t in fallbacks if t not in existing_texts][:2]

    return [
        _RecCandidate(disease=disease, text=text, source="llm", requires_physician=False, magnitude=0.0)
        for text in llm_texts
        if text.strip()
    ]


# ---------------------------------------------------------------------------
# SHAP-based prioritization
# ---------------------------------------------------------------------------


def _assign_priorities(candidates: list[_RecCandidate]) -> list[_RecCandidate]:
    """Sort candidates by SHAP magnitude (descending) and return ordered list.

    Rule-based recs are sorted by magnitude; LLM recs follow; physician recs lead.
    """
    physician = [c for c in candidates if c.requires_physician]
    rule_recs = [c for c in candidates if c.source == "rule" and not c.requires_physician]
    llm_recs = [c for c in candidates if c.source == "llm" and not c.requires_physician]

    rule_recs_sorted = sorted(rule_recs, key=lambda c: c.magnitude, reverse=True)

    return physician + rule_recs_sorted + llm_recs


# ---------------------------------------------------------------------------
# Physician referral
# ---------------------------------------------------------------------------


def _physician_referral_candidate(disease: str) -> _RecCandidate:
    return _RecCandidate(
        disease=disease,
        text="Consult a licensed physician promptly for a comprehensive clinical evaluation",
        source="rule",
        requires_physician=True,
        magnitude=float("inf"),  # always sorts first
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_recommendations(
    prediction_doc: dict,
    feature_vector: list[float],
    lifestyle_profile: Optional[LifestyleProfile],
) -> list[Recommendation]:
    """Full recommendation pipeline.

    Parameters
    ----------
    prediction_doc:
        MongoDB prediction document (must contain ``_id``, ``risk_scores``,
        and optionally ``shap_values``).
    feature_vector:
        14-dimensional feature vector list.
    lifestyle_profile:
        Patient's lifestyle profile (may be None).

    Returns
    -------
    Flat list of Recommendation objects across all diseases, sorted by
    priority (ascending = highest priority first).

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    prediction_id = str(prediction_doc.get("_id", ""))
    risk_scores: dict[str, float] = prediction_doc.get("risk_scores", {})
    shap_by_disease: dict[str, list[float]] = prediction_doc.get("shap_values") or {}

    # Determine if any disease exceeds the physician threshold (Req 9.5)
    any_critical = any(score > _PHYSICIAN_THRESHOLD for score in risk_scores.values())

    all_recs: list[Recommendation] = []

    for disease in _DISEASES:
        risk_score = risk_scores.get(disease, 0.0)

        # Only generate recommendations for diseases with risk > 30 (Req 9.1)
        if risk_score <= _RISK_THRESHOLD:
            continue

        shap_values = shap_by_disease.get(disease)

        # 1. Rule-based recommendations (Req 9.2)
        candidates = _apply_rules(disease, feature_vector, shap_values)

        # 2. LLM enrichment (Req 9.3)
        existing_texts = [c.text for c in candidates]
        llm_candidates = _llm_enrich(
            disease, risk_score, feature_vector, lifestyle_profile, existing_texts,
        )
        candidates.extend(llm_candidates)

        # 3. Physician referral when any risk > 75 (Req 9.5)
        if any_critical:
            candidates.append(_physician_referral_candidate(disease))

        # 4. SHAP-based prioritization (Req 9.4)
        ordered = _assign_priorities(candidates)

        # 5. Ensure minimum 3 recommendations (Req 9.1)
        if len(ordered) < _MIN_RECS:
            existing_texts_all = {c.text for c in ordered}
            fallbacks = _FALLBACK_RECS.get(disease, [])
            for fb_text in fallbacks:
                if len(ordered) >= _MIN_RECS:
                    break
                if fb_text not in existing_texts_all:
                    ordered.append(_RecCandidate(
                        disease=disease,
                        text=fb_text,
                        source="rule",
                        requires_physician=False,
                        magnitude=0.0,
                    ))

        # Convert to Recommendation models with sequential priority numbers
        for i, candidate in enumerate(ordered, start=1):
            all_recs.append(Recommendation(
                prediction_id=prediction_id,
                disease=candidate.disease,
                text=candidate.text,
                priority=i,
                source=candidate.source,  # type: ignore[arg-type]
                requires_physician=candidate.requires_physician,
            ))

    return all_recs
