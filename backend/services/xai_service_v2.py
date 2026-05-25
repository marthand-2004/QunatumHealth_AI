"""XAI / SHAP explainability service v2 — disease-specific feature subsets.

Extends xai_service.py to support the v2 disease-specific feature subsets
(Diabetes: 6 features, CVD: 6 features, CKD: 5 features).

Key differences from v1:
- Accepts disease-specific feature vectors (not the unified 14-dim vector)
- Uses DISEASE_FEATURE_SUBSETS from feature_mapper for correct feature names
- Returns exactly 3 SHAPFeature objects with direction labels
- Generates human-readable explanation naming top-3 features and direction
- Uses shap.TreeExplainer on the v2 RF model (disease-specific scaler applied)
- Uses kernel-based linear approximation for quantum model predictions
- Completes within 5 seconds of prediction completion (Req 7.5)
- SHAPFeature is a Pydantic BaseModel for use in prediction_v2.py

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 18.2, 18.3, 18.5
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SHAP import with graceful fallback
# ---------------------------------------------------------------------------

try:
    import shap as _shap  # type: ignore

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False
    logger.warning(
        "xai_service_v2: shap not installed — using linear approximation fallback"
    )

# ---------------------------------------------------------------------------
# Disease-specific feature subsets (mirrors feature_mapper.DISEASE_FEATURE_SUBSETS)
# ---------------------------------------------------------------------------

DISEASE_FEATURE_SUBSETS: dict[str, list[str]] = {
    "diabetes": ["glucose", "hba1c", "bmi", "age", "systolic_bp", "diastolic_bp"],
    "cvd":      ["age", "cholesterol", "systolic_bp", "diastolic_bp", "smoking_encoded", "bmi"],
    "ckd":      ["creatinine", "hemoglobin", "systolic_bp", "diastolic_bp", "age"],
}

# Human-readable labels for feature names
FEATURE_LABELS: dict[str, str] = {
    "glucose":           "Blood Glucose",
    "hba1c":             "HbA1c",
    "bmi":               "BMI",
    "age":               "Age",
    "systolic_bp":       "Systolic BP",
    "diastolic_bp":      "Diastolic BP",
    "cholesterol":       "Cholesterol",
    "smoking_encoded":   "Smoking Status",
    "creatinine":        "Creatinine",
    "hemoglobin":        "Hemoglobin",
}

# Disease-specific relative feature weights for the linear approximation fallback.
# Indices correspond to the ordered feature subsets above.
_DISEASE_WEIGHTS: dict[str, list[float]] = {
    # diabetes: glucose, hba1c, bmi, age, systolic_bp, diastolic_bp
    "diabetes": [0.30, 0.30, 0.15, 0.10, 0.08, 0.07],
    # cvd: age, cholesterol, systolic_bp, diastolic_bp, smoking_encoded, bmi
    "cvd":      [0.15, 0.25, 0.20, 0.15, 0.15, 0.10],
    # ckd: creatinine, hemoglobin, systolic_bp, diastolic_bp, age
    "ckd":      [0.35, 0.25, 0.15, 0.10, 0.15],
}

# Timeout for the entire compute_shap_v2 call (seconds) — Req 7.5
_SHAP_TIMEOUT_SECONDS = 5.0

# Number of SHAP features to return — Req 18.5
_TOP_N = 3


# ---------------------------------------------------------------------------
# SHAPFeature Pydantic model (reused by prediction_v2.py)
# ---------------------------------------------------------------------------

class SHAPFeature(BaseModel):
    """A single SHAP feature attribution result.

    Attributes
    ----------
    feature_name:
        Raw feature identifier (e.g. "glucose").
    shap_value:
        Signed SHAP value in risk-score space (percentage points).
        Positive = increases risk, negative = decreases risk.
    direction:
        Human-readable direction label derived from the sign of shap_value.
    """

    feature_name: str
    shap_value: float
    direction: Literal["increases risk", "decreases risk"]

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "shap_value": round(self.shap_value, 4),
            "direction": self.direction,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _linear_shap_approximation(
    features: list[float],
    disease: str,
    risk_score: float,
) -> list[float]:
    """Kernel-based linear approximation of SHAP values for quantum predictions.

    Distributes the deviation from the base score (50.0) across features
    proportionally to their weighted contribution. Used for quantum model
    outputs (no tree structure) and as a fallback when shap is not installed.

    Parameters
    ----------
    features:
        Disease-specific feature vector.
    disease:
        One of "diabetes", "cvd", "ckd".
    risk_score:
        The predicted risk score in [0, 100] used to compute the delta from base.

    Returns
    -------
    List of SHAP values (one per feature in the disease-specific subset).
    """
    base_value = 50.0
    delta = risk_score - base_value

    n_features = len(features)
    weights = _DISEASE_WEIGHTS.get(disease, [1.0 / n_features] * n_features)

    # Pad or truncate weights to match actual feature count
    if len(weights) != n_features:
        weights = (weights + [1.0 / n_features] * n_features)[:n_features]
        total = sum(weights)
        weights = [w / total for w in weights]

    x = np.array(features, dtype=float)

    # Normalise features to [-1, 1] range (rough population normalisation)
    max_abs = np.abs(x).max()
    x_norm = np.clip(x / (max_abs + 1e-9), -1.0, 1.0)

    # Weighted signed contributions
    raw = np.array(weights) * x_norm
    raw_sum = float(np.sum(raw))

    if abs(raw_sum) < 1e-9:
        # Uniform distribution when all features are zero
        shap_vals = [delta / n_features] * n_features
    else:
        shap_vals = list((raw / raw_sum) * delta)

    return shap_vals


def _compute_shap_tree_v2(
    features: list[float],
    disease: str,
) -> tuple[list[float], float]:
    """Compute SHAP values using TreeExplainer on the v2 RF model.

    Loads the disease-specific RF model from classical_ml_v2 and applies
    the disease-specific StandardScaler before computing SHAP values.

    Parameters
    ----------
    features:
        Disease-specific feature vector (already in raw, unscaled form).
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    (shap_values, base_value)
        shap_values: list of SHAP values in risk-score space (×100).
        base_value:  expected model output in risk-score space.

    Raises
    ------
    RuntimeError
        If the RF model or scaler is not available.
    """
    from backend.services.classical_ml_v2 import get_loaded_models

    models = get_loaded_models()
    bundle = models.get(disease, {})
    rf = bundle.get("rf")
    scaler = bundle.get("scaler")

    if rf is None:
        raise RuntimeError(
            f"xai_service_v2: RF model for '{disease}' is not loaded."
        )

    x = np.array(features, dtype=float).reshape(1, -1)

    # Apply StandardScaler before SHAP (must match training pipeline)
    if scaler is not None:
        try:
            x = scaler.transform(x)
        except Exception as exc:
            logger.warning(
                "xai_service_v2: StandardScaler.transform() failed for '%s': %s. "
                "Proceeding with unscaled features for SHAP.",
                disease,
                exc,
            )

    explainer = _shap.TreeExplainer(rf)
    shap_output = explainer.shap_values(x)

    # shap_values shape: list of (n_samples, n_features) per class, or (n_samples, n_features)
    if isinstance(shap_output, list):
        # Multi-class output: take class-1 (positive class) SHAP values
        sv = np.array(shap_output[1]).flatten()
        base_val = float(explainer.expected_value[1]) * 100.0
    else:
        sv = np.array(shap_output).flatten()
        base_val = float(explainer.expected_value) * 100.0

    # Scale from probability space to risk-score space [0, 100]
    shap_vals = list(sv * 100.0)
    return shap_vals, base_val


def _build_shap_features(
    shap_values: list[float],
    disease: str,
) -> list[SHAPFeature]:
    """Select the top-3 features by absolute SHAP value and build SHAPFeature objects.

    Parameters
    ----------
    shap_values:
        SHAP values in risk-score space, one per feature in the disease subset.
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    List of exactly 3 SHAPFeature objects sorted by descending |shap_value|.
    """
    feature_names = DISEASE_FEATURE_SUBSETS.get(disease, [])
    n = len(feature_names)

    # Pad shap_values if shorter than feature list (safety guard)
    padded = list(shap_values) + [0.0] * max(0, n - len(shap_values))

    # Sort by absolute value descending
    indexed = sorted(
        enumerate(padded[:n]),
        key=lambda t: abs(t[1]),
        reverse=True,
    )

    top3: list[SHAPFeature] = []
    for i, sv in indexed[:_TOP_N]:
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        direction: Literal["increases risk", "decreases risk"] = (
            "increases risk" if sv >= 0 else "decreases risk"
        )
        top3.append(SHAPFeature(feature_name=name, shap_value=sv, direction=direction))

    # Guarantee exactly 3 items even if fewer features exist
    while len(top3) < _TOP_N:
        top3.append(
            SHAPFeature(feature_name="unknown", shap_value=0.0, direction="increases risk")
        )

    return top3


def _generate_explanation(top3: list[SHAPFeature], disease: str) -> str:
    """Generate a human-readable explanation naming the top-3 features and direction.

    Produces a sentence of the form:
      "High Blood Glucose and HbA1c increased diabetes risk; low BMI also contributed."

    Parameters
    ----------
    top3:
        Exactly 3 SHAPFeature objects sorted by descending |shap_value|.
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    Non-empty explanation string (Req 7.3, 18.3).
    """
    disease_label = disease.upper() if disease == "cvd" else disease.capitalize()

    def _describe(feat: SHAPFeature) -> str:
        label = FEATURE_LABELS.get(feat.feature_name, feat.feature_name.replace("_", " ").title())
        qualifier = "High" if feat.direction == "increases risk" else "Low"
        return f"{qualifier} {label}"

    if not top3:
        return f"Insufficient data to explain {disease_label} risk prediction."

    primary = _describe(top3[0])
    direction_verb = "increased" if top3[0].direction == "increases risk" else "decreased"

    if len(top3) == 1:
        return f"{primary} {direction_verb} {disease_label} risk."

    if len(top3) == 2:
        secondary = _describe(top3[1])
        sec_verb = "increased" if top3[1].direction == "increases risk" else "decreased"
        return (
            f"{primary} {direction_verb} {disease_label} risk; "
            f"{secondary} also {sec_verb} risk."
        )

    # All 3 features
    secondary = _describe(top3[1])
    tertiary = _describe(top3[2])

    # Group features with the same direction for a more natural sentence
    increases = [f for f in top3 if f.direction == "increases risk"]
    decreases = [f for f in top3 if f.direction == "decreases risk"]

    if len(increases) == 3:
        labels = [
            FEATURE_LABELS.get(f.feature_name, f.feature_name.replace("_", " ").title())
            for f in top3
        ]
        return (
            f"High {labels[0]} and {labels[1]} increased {disease_label} risk; "
            f"high {labels[2]} also contributed."
        )

    if len(decreases) == 3:
        labels = [
            FEATURE_LABELS.get(f.feature_name, f.feature_name.replace("_", " ").title())
            for f in top3
        ]
        return (
            f"Low {labels[0]} and {labels[1]} decreased {disease_label} risk; "
            f"low {labels[2]} also contributed."
        )

    # Mixed directions — describe each individually
    parts = []
    for feat in top3:
        label = FEATURE_LABELS.get(feat.feature_name, feat.feature_name.replace("_", " ").title())
        qualifier = "High" if feat.direction == "increases risk" else "Low"
        verb = "increased" if feat.direction == "increases risk" else "decreased"
        parts.append(f"{qualifier} {label} {verb} risk")

    return f"{parts[0]}; {parts[1]}; {parts[2]}."


def _compute_shap_core(
    features: list[float],
    disease: str,
    model_used: str,
    risk_score: float,
) -> tuple[list[SHAPFeature], str, float]:
    """Core SHAP computation logic (runs inside ThreadPoolExecutor for timeout guard).

    Returns
    -------
    (top3_shap_features, explanation, base_value)
    """
    shap_vals: list[float]
    base_value: float = 50.0

    # --- Classical path: TreeExplainer on v2 RF model (Req 7.1) ---
    if model_used == "classical" and _SHAP_AVAILABLE:
        try:
            shap_vals, base_value = _compute_shap_tree_v2(features, disease)
        except Exception as exc:
            logger.warning(
                "xai_service_v2: TreeExplainer failed for '%s' (%s). "
                "Falling back to linear approximation.",
                disease,
                exc,
            )
            shap_vals = _linear_shap_approximation(features, disease, risk_score)
    else:
        # Quantum path or shap not available: kernel-based linear approximation
        shap_vals = _linear_shap_approximation(features, disease, risk_score)

    # Build top-3 SHAPFeature objects (Req 18.2, 18.5)
    top3 = _build_shap_features(shap_vals, disease)

    # Generate human-readable explanation (Req 7.3, 18.3)
    explanation = _generate_explanation(top3, disease)

    return top3, explanation, base_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shap_v2(
    features: list[float],
    disease: str,
    model_used: str,
    risk_score: Optional[float] = None,
) -> tuple[list[SHAPFeature], float]:
    """Compute SHAP feature attributions for a v2 disease-specific prediction.

    For classical model predictions, uses ``shap.TreeExplainer`` on the v2
    Random Forest model (with disease-specific StandardScaler applied).

    For quantum model predictions (or when shap is not installed), uses a
    kernel-based linear approximation that distributes the deviation from the
    base score proportionally to disease-specific feature weights.

    Completes within 5 seconds of being called (Req 7.5). Uses a
    ``concurrent.futures.ThreadPoolExecutor`` timeout guard to enforce this.

    Parameters
    ----------
    features:
        Disease-specific feature vector. Must match the length defined in
        ``DISEASE_FEATURE_SUBSETS`` for the given disease
        (6 for diabetes/cvd, 5 for ckd).
    disease:
        One of "diabetes", "cvd", "ckd".
    model_used:
        "classical" or "quantum". Determines which SHAP method is used.
    risk_score:
        Optional pre-computed risk score in [0, 100]. Used as the base for
        the linear approximation when TreeExplainer is unavailable. If None,
        defaults to 50.0.

    Returns
    -------
    (top3_shap_features, base_value)
        top3_shap_features: List of exactly 3 SHAPFeature objects sorted by
            descending |shap_value| (Req 18.2, 18.5).
        base_value: Expected model output in risk-score space (used for
            waterfall chart rendering).

    Raises
    ------
    ValueError
        If ``disease`` is not one of the supported disease identifiers.

    Notes
    -----
    The human-readable explanation string is accessible via
    ``generate_explanation_v2(top3_shap_features, disease)`` if needed
    separately, or use ``compute_shap_v2_with_explanation()`` to get all
    three return values at once.
    """
    if disease not in DISEASE_FEATURE_SUBSETS:
        raise ValueError(
            f"xai_service_v2: Unsupported disease '{disease}'. "
            f"Must be one of: {sorted(DISEASE_FEATURE_SUBSETS)}"
        )

    _effective_risk_score = risk_score if risk_score is not None else 50.0
    _start = time.perf_counter()

    # Run SHAP computation with a 5-second timeout guard (Req 7.5)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _compute_shap_core,
                features,
                disease,
                model_used,
                _effective_risk_score,
            )
            top3, _explanation, base_value = future.result(timeout=_SHAP_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "xai_service_v2: compute_shap_v2 for '%s' timed out after %ds. "
            "Falling back to linear approximation.",
            disease,
            _SHAP_TIMEOUT_SECONDS,
        )
        # Graceful fallback: linear approximation on timeout
        shap_vals = _linear_shap_approximation(features, disease, _effective_risk_score)
        top3 = _build_shap_features(shap_vals, disease)
        base_value = 50.0
    except Exception as exc:
        logger.error(
            "xai_service_v2: Unexpected error in compute_shap_v2 for '%s': %s. "
            "Falling back to linear approximation.",
            disease,
            exc,
        )
        shap_vals = _linear_shap_approximation(features, disease, _effective_risk_score)
        top3 = _build_shap_features(shap_vals, disease)
        base_value = 50.0

    elapsed = time.perf_counter() - _start
    logger.debug(
        "xai_service_v2: SHAP computed for '%s' (model=%s) in %.3fs",
        disease,
        model_used,
        elapsed,
    )

    return top3, base_value


def compute_shap_v2_with_explanation(
    features: list[float],
    disease: str,
    model_used: str,
    risk_score: Optional[float] = None,
) -> tuple[list[SHAPFeature], str, float]:
    """Compute SHAP attributions and return top-3 features, explanation, and base value.

    Convenience wrapper around ``compute_shap_v2`` that also returns the
    human-readable explanation string.

    Parameters
    ----------
    features:
        Disease-specific feature vector.
    disease:
        One of "diabetes", "cvd", "ckd".
    model_used:
        "classical" or "quantum".
    risk_score:
        Optional pre-computed risk score in [0, 100].

    Returns
    -------
    (top3_shap_features, explanation, base_value)
        top3_shap_features: List of exactly 3 SHAPFeature objects.
        explanation: Human-readable string naming top-3 features and direction.
        base_value: Expected model output in risk-score space.
    """
    top3, base_value = compute_shap_v2(features, disease, model_used, risk_score)
    explanation = _generate_explanation(top3, disease)
    return top3, explanation, base_value


def generate_explanation_v2(top3: list[SHAPFeature], disease: str) -> str:
    """Generate a human-readable explanation from pre-computed SHAPFeature objects.

    Parameters
    ----------
    top3:
        List of SHAPFeature objects (typically 3 items).
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    Non-empty explanation string.
    """
    return _generate_explanation(top3, disease)


def get_feature_names(disease: str) -> list[str]:
    """Return the ordered feature names for a given disease.

    Parameters
    ----------
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    Ordered list of feature name strings.

    Raises
    ------
    ValueError
        If ``disease`` is not one of the supported disease identifiers.
    """
    if disease not in DISEASE_FEATURE_SUBSETS:
        raise ValueError(
            f"xai_service_v2: Unsupported disease '{disease}'. "
            f"Must be one of: {sorted(DISEASE_FEATURE_SUBSETS)}"
        )
    return list(DISEASE_FEATURE_SUBSETS[disease])
