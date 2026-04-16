"""XAI / SHAP explainability service.

Provides SHAP feature attributions, Chart.js waterfall chart data, and
LLM-generated natural language explanations for disease risk predictions.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level",
]

FEATURE_LABELS: dict[str, str] = {
    "glucose": "Blood Glucose",
    "hba1c": "HbA1c",
    "creatinine": "Creatinine",
    "cholesterol": "Cholesterol",
    "triglycerides": "Triglycerides",
    "hemoglobin": "Hemoglobin",
    "bmi": "BMI",
    "age": "Age",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "smoking_encoded": "Smoking Status",
    "exercise_frequency": "Exercise Frequency",
    "sleep_hours": "Sleep Hours",
    "stress_level": "Stress Level",
}

# Disease-specific feature weights (relative importance for linear approximation)
_DISEASE_WEIGHTS: dict[str, list[float]] = {
    "diabetes":  [0.25, 0.30, 0.05, 0.05, 0.05, 0.03, 0.08, 0.05, 0.03, 0.02, 0.04, 0.02, 0.01, 0.02],
    "cvd":       [0.05, 0.03, 0.05, 0.20, 0.15, 0.05, 0.08, 0.08, 0.12, 0.08, 0.06, 0.03, 0.01, 0.01],
    "ckd":       [0.08, 0.05, 0.30, 0.05, 0.05, 0.10, 0.05, 0.08, 0.08, 0.05, 0.03, 0.02, 0.03, 0.03],
}

# ---------------------------------------------------------------------------
# Optional SHAP import with mock fallback
# ---------------------------------------------------------------------------

try:
    import shap as _shap  # type: ignore

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — using linear approximation fallback for SHAP values")

# ---------------------------------------------------------------------------
# Optional sklearn import (needed for TreeExplainer)
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _linear_shap_approximation(
    features: list[float],
    risk_score: float,
    disease: str,
) -> list[float]:
    """Kernel-based linear approximation of SHAP values.

    Used for quantum model outputs (no tree structure) and as a fallback
    when the shap library is not installed.

    The approximation distributes the deviation from the base score (50.0)
    across features proportionally to their weighted contribution.

    Returns a list of 14 SHAP values that sum to (risk_score - base_value).
    """
    base_value = 50.0
    delta = risk_score - base_value

    weights = _DISEASE_WEIGHTS.get(disease, [1.0 / 14] * 14)
    x = np.array(features, dtype=float)

    # Normalise features to [-1, 1] range (rough population normalisation)
    x_norm = np.clip(x / (np.abs(x).max() + 1e-9), -1.0, 1.0)

    # Weighted signed contributions
    raw = np.array(weights) * x_norm
    raw_sum = np.sum(raw)

    if abs(raw_sum) < 1e-9:
        # Uniform distribution when all features are zero
        shap_vals = [delta / 14.0] * 14
    else:
        shap_vals = list((raw / raw_sum) * delta)

    return shap_vals


def _compute_shap_tree(
    features: list[float],
    disease: str,
) -> tuple[list[float], float]:
    """Compute SHAP values using TreeExplainer on the classical RF model.

    Returns (shap_values, base_value) where base_value is the expected
    model output (in [0, 100] risk score space).
    """
    from backend.services.classical_ml import _models, _SKLEARN_AVAILABLE as _sk

    if not _sk or not _models:
        raise RuntimeError("sklearn models not available")

    disease_models = _models.get(disease, {})
    rf = disease_models.get("rf")
    if rf is None:
        raise RuntimeError(f"RF model for {disease} not loaded")

    x = np.array(features, dtype=float).reshape(1, -1)

    explainer = _shap.TreeExplainer(rf)
    shap_output = explainer.shap_values(x)

    # shap_values shape: (n_classes, n_samples, n_features) or (n_samples, n_features)
    if isinstance(shap_output, list):
        # Multi-class: take class-1 (positive class) SHAP values
        sv = np.array(shap_output[1]).flatten()
        base_val = float(explainer.expected_value[1]) * 100.0
    else:
        sv = np.array(shap_output).flatten()
        base_val = float(explainer.expected_value) * 100.0

    # Scale SHAP values from probability space to risk score space [0, 100]
    shap_vals = list(sv * 100.0)
    return shap_vals, base_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shap_values(
    features: list[float],
    disease: str,
    model_used: str,
) -> list[float]:
    """Compute SHAP values for each of the 14 features.

    - For classical model: uses ``shap.TreeExplainer`` on the RF model.
    - For quantum model: uses a linear kernel-based approximation.
    - Falls back to linear approximation if shap is not installed.

    Parameters
    ----------
    features:
        14-dimensional feature vector.
    disease:
        One of "diabetes", "cvd", "ckd".
    model_used:
        "classical" or "quantum".

    Returns
    -------
    List of 14 SHAP values (one per feature).
    """
    if len(features) != 14:
        raise ValueError(f"Expected 14 features, got {len(features)}")

    # Compute a risk score for the linear approximation base
    from backend.services.classical_ml import predict_classical
    risk_scores = predict_classical(features)
    risk_score = risk_scores.get(disease, 50.0)

    if model_used == "classical" and _SHAP_AVAILABLE and _SKLEARN_AVAILABLE:
        try:
            shap_vals, _ = _compute_shap_tree(features, disease)
            return shap_vals
        except Exception as exc:
            logger.warning("TreeExplainer failed (%s), falling back to linear approximation", exc)

    # Quantum model or fallback: linear approximation
    return _linear_shap_approximation(features, risk_score, disease)


def build_waterfall_chart_data(
    feature_names: list[str],
    shap_values: list[float],
    base_value: float,
    prediction: float,
) -> dict:
    """Build Chart.js-compatible waterfall chart data for SHAP values.

    The waterfall chart shows how each feature pushes the prediction up or
    down from the base (expected) value to the final prediction.

    Parameters
    ----------
    feature_names:
        List of feature name strings (length 14).
    shap_values:
        List of SHAP values corresponding to each feature (length 14).
    base_value:
        The model's expected output (base/average prediction).
    prediction:
        The final model prediction (risk score).

    Returns
    -------
    Chart.js-compatible dict with ``type``, ``data``, and ``options`` keys.
    """
    labels = [FEATURE_LABELS.get(n, n) for n in feature_names]

    # Sort by absolute SHAP value descending for readability
    indexed = sorted(enumerate(shap_values), key=lambda t: abs(t[1]), reverse=True)
    sorted_labels = [labels[i] for i, _ in indexed]
    sorted_shap = [sv for _, sv in indexed]

    # Colours: positive contributions → red (risk-increasing), negative → green
    colors = [
        "rgba(239, 68, 68, 0.8)" if sv >= 0 else "rgba(34, 197, 94, 0.8)"
        for sv in sorted_shap
    ]

    # Running total for floating bar segments
    running = base_value
    bar_data: list[dict] = []
    for sv in sorted_shap:
        bar_data.append({"x": [running, running + sv]})
        running += sv

    return {
        "type": "bar",
        "data": {
            "labels": sorted_labels,
            "datasets": [
                {
                    "label": "SHAP contribution",
                    "data": [round(sv, 4) for sv in sorted_shap],
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True,
            "plugins": {
                "legend": {"display": False},
                "title": {
                    "display": True,
                    "text": f"SHAP Waterfall — Base: {base_value:.1f}, Prediction: {prediction:.1f}",
                },
                "tooltip": {
                    "callbacks": {
                        "label": "function(ctx) { return ctx.raw > 0 ? '+' + ctx.raw.toFixed(2) : ctx.raw.toFixed(2); }"
                    }
                },
            },
            "scales": {
                "x": {
                    "title": {"display": True, "text": "Risk Score Contribution"},
                },
            },
        },
        "meta": {
            "base_value": round(base_value, 4),
            "prediction": round(prediction, 4),
            "feature_order": sorted_labels,
        },
    }


def generate_llm_explanation(
    feature_names: list[str],
    shap_values: list[float],
    risk_scores: dict[str, float],
    model_used: str,
) -> str:
    """Generate a natural language explanation of the top SHAP contributors.

    Tries Gemini first, then OpenAI, then falls back to a template-based
    explanation if no API key is configured.

    Parameters
    ----------
    feature_names:
        List of 14 feature name strings.
    shap_values:
        SHAP values for the disease with the highest risk score.
    risk_scores:
        Dict of disease → risk score (0–100).
    model_used:
        "classical" or "quantum" (used in the explanation text).

    Returns
    -------
    Natural language explanation string.
    """
    # Identify top 3 contributors by absolute SHAP value
    indexed = sorted(enumerate(shap_values), key=lambda t: abs(t[1]), reverse=True)
    top3 = [(feature_names[i], sv) for i, sv in indexed[:3]]

    # Identify highest-risk disease
    primary_disease = max(risk_scores, key=lambda d: risk_scores[d])
    primary_score = risk_scores[primary_disease]

    model_label = "Classical ML (Random Forest + XGBoost)" if model_used == "classical" else "Quantum VQC"

    prompt = _build_explanation_prompt(top3, risk_scores, primary_disease, primary_score, model_label)

    # Try Gemini
    if settings.GEMINI_API_KEY:
        try:
            return _call_gemini(prompt)
        except Exception as exc:
            logger.warning("Gemini explanation failed: %s", exc)

    # Try OpenAI
    if settings.OPENAI_API_KEY:
        try:
            return _call_openai(prompt)
        except Exception as exc:
            logger.warning("OpenAI explanation failed: %s", exc)

    # Template fallback
    return _template_explanation(top3, risk_scores, primary_disease, primary_score, model_label)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _build_explanation_prompt(
    top3: list[tuple[str, float]],
    risk_scores: dict[str, float],
    primary_disease: str,
    primary_score: float,
    model_label: str,
) -> str:
    top_lines = "\n".join(
        f"  - {FEATURE_LABELS.get(name, name)}: {'+' if sv >= 0 else ''}{sv:.2f} contribution"
        for name, sv in top3
    )
    score_lines = "\n".join(
        f"  - {d.capitalize()}: {s:.1f}%" for d, s in risk_scores.items()
    )
    return (
        "You are a medical AI assistant. Explain the following disease risk prediction "
        "results in clear, empathetic language suitable for a patient. "
        "Do not provide a diagnosis. Recommend consulting a physician for high-risk scores.\n\n"
        f"Model used: {model_label}\n\n"
        f"Risk scores:\n{score_lines}\n\n"
        f"Top contributing factors for {primary_disease} (risk {primary_score:.1f}%):\n{top_lines}\n\n"
        "Write 2–3 sentences explaining what these results mean and what the patient should consider."
    )


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _call_openai(prompt: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _template_explanation(
    top3: list[tuple[str, float]],
    risk_scores: dict[str, float],
    primary_disease: str,
    primary_score: float,
    model_label: str,
) -> str:
    """Template-based fallback explanation when no LLM API key is available."""
    risk_level = "high" if primary_score >= 70 else "moderate" if primary_score >= 40 else "low"

    top_factor_name = FEATURE_LABELS.get(top3[0][0], top3[0][0]) if top3 else "unknown"
    top_factor_dir = "elevated" if top3[0][1] > 0 else "reduced" if top3 else "normal"

    other_diseases = [
        f"{d.capitalize()} ({s:.0f}%)"
        for d, s in risk_scores.items()
        if d != primary_disease
    ]
    other_str = " and ".join(other_diseases) if other_diseases else ""

    lines = [
        f"Based on the {model_label} analysis, your {primary_disease.capitalize()} risk score "
        f"is {primary_score:.1f}%, which is considered {risk_level}.",
        f"The most influential factor is your {top_factor_name}, which is {top_factor_dir} "
        f"and contributes most significantly to this result.",
    ]
    if other_str:
        lines.append(f"Your other risk scores are: {other_str}.")
    if primary_score >= 70:
        lines.append(
            "Given the elevated risk level, we strongly recommend consulting a licensed physician "
            "for a comprehensive evaluation."
        )
    elif primary_score >= 40:
        lines.append(
            "Consider discussing these results with your healthcare provider and reviewing "
            "lifestyle factors that may help reduce your risk."
        )
    else:
        lines.append(
            "Continue maintaining your healthy habits and schedule regular check-ups with your doctor."
        )

    return " ".join(lines)
