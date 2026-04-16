"""Classical ML prediction service — RF + XGBoost ensemble.

Loads pre-trained models from data/models/ at startup.
Falls back to untrained sklearn models if files are not found.
Falls back to a deterministic mock if sklearn/xgboost are not installed.

Requirements: 6.1, 6.2, 6.3
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional sklearn / xgboost imports with mock fallback
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.dummy import DummyClassifier  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier  # type: ignore

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False

try:
    import joblib  # type: ignore

    _JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JOBLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISEASE_NAMES = ["diabetes", "cvd", "ckd"]
FEATURE_DIM = 14

# Path to pre-trained model files
_MODELS_DIR = Path(
    os.environ.get(
        "MODELS_DIR",
        str(Path(__file__).parent.parent.parent / "data" / "models"),
    )
)

# ---------------------------------------------------------------------------
# Mock fallback (used when sklearn/xgboost are not installed)
# ---------------------------------------------------------------------------


def _mock_predict(features: list[float]) -> dict[str, float]:
    """Deterministic mock prediction for testing without sklearn/xgboost."""
    x = np.array(features, dtype=float)
    scores: dict[str, float] = {}
    for i, name in enumerate(DISEASE_NAMES):
        # Deterministic value in [0, 100] based on feature sum
        val = float(
            np.clip(np.abs(np.cos(np.sum(x) * (i + 1) * 0.07)) * 100, 0.0, 100.0)
        )
        scores[name] = round(val, 4)
    return scores


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _make_untrained_rf() -> "RandomForestClassifier":
    """Return an RF classifier with random weights (no training data)."""
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    # Fit on minimal dummy data so predict_proba works
    X_dummy = np.zeros((4, FEATURE_DIM))
    y_dummy = [0, 1, 0, 1]
    clf.fit(X_dummy, y_dummy)
    return clf


def _make_untrained_xgb() -> "XGBClassifier":
    """Return an XGB classifier with random weights (no training data)."""
    clf = XGBClassifier(
        n_estimators=10,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    X_dummy = np.zeros((4, FEATURE_DIM))
    y_dummy = [0, 1, 0, 1]
    clf.fit(X_dummy, y_dummy)
    return clf


def _load_model(filename: str, make_fallback):
    """Load a model from disk; return fallback if file not found or load fails."""
    if not _JOBLIB_AVAILABLE:
        return None
    path = _MODELS_DIR / filename
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    return make_fallback()


def load_models() -> dict[str, dict]:
    """Load RF and XGBoost models for each disease.

    Returns a dict keyed by disease name, each containing "rf" and "xgb" models.
    Falls back to untrained sklearn models if files are not found.
    Returns empty dict if sklearn is not available (mock path will be used).
    """
    if not _SKLEARN_AVAILABLE:
        return {}

    models: dict[str, dict] = {}
    for disease in DISEASE_NAMES:
        rf = _load_model(f"rf_{disease}.joblib", _make_untrained_rf)
        xgb = _load_model(f"xgb_{disease}.joblib", _make_untrained_xgb) if _XGBOOST_AVAILABLE else None
        models[disease] = {"rf": rf, "xgb": xgb}
    return models


# ---------------------------------------------------------------------------
# Module-level model loading (once at import time)
# ---------------------------------------------------------------------------
_models: dict[str, dict] = load_models()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_classical(features: list[float]) -> dict[str, float]:
    """Run RF + XGBoost ensemble and return risk scores for Diabetes, CVD, CKD.

    Parameters
    ----------
    features:
        14-dimensional feature vector (must have exactly 14 elements).

    Returns
    -------
    dict with keys "diabetes", "cvd", "ckd" — each value in [0, 100].
    The ensemble averages RF and XGBoost probability predictions.
    """
    if len(features) != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {len(features)}")

    if not _SKLEARN_AVAILABLE or not _models:
        return _mock_predict(features)

    x = np.array(features, dtype=float).reshape(1, -1)
    scores: dict[str, float] = {}

    for disease in DISEASE_NAMES:
        disease_models = _models.get(disease, {})
        rf = disease_models.get("rf")
        xgb = disease_models.get("xgb")

        probs: list[float] = []

        if rf is not None:
            try:
                rf_prob = float(rf.predict_proba(x)[0][1])
                probs.append(rf_prob)
            except Exception:
                pass

        if xgb is not None:
            try:
                xgb_prob = float(xgb.predict_proba(x)[0][1])
                probs.append(xgb_prob)
            except Exception:
                pass

        if probs:
            avg_prob = sum(probs) / len(probs)
        else:
            # Fallback: deterministic mock for this disease
            avg_prob = float(
                np.clip(
                    np.abs(np.cos(np.sum(x) * (DISEASE_NAMES.index(disease) + 1) * 0.07)),
                    0.0,
                    1.0,
                )
            )

        score = float(np.clip(avg_prob * 100.0, 0.0, 100.0))
        scores[disease] = round(score, 4)

    return scores
