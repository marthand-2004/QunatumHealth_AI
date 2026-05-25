"""Classical ML prediction service v2 — disease-specific RF + XGBoost ensemble.

Loads disease-specific StandardScaler, RF, and XGBoost models at startup.
Accepts disease-specific feature vectors (6 dims for diabetes/cvd, 5 for ckd).
Falls back to untrained models if files are missing; logs descriptive errors.

Requirements: 2.1, 12.1, 12.3, 12.6
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional sklearn / xgboost / joblib imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

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

DISEASE_NAMES = ("diabetes", "cvd", "ckd")

# Disease-specific feature vector dimensions (must match training pipeline)
DISEASE_FEATURE_DIMS: dict[str, int] = {
    "diabetes": 6,
    "cvd": 6,
    "ckd": 5,
}

# Paths to model artifacts
_REPO_ROOT = Path(__file__).parent.parent.parent
_DATA_MODELS_DIR = Path(
    os.environ.get("MODELS_DIR", str(_REPO_ROOT / "data" / "models"))
)
_TRAINING_MODELS_DIR = _REPO_ROOT / "training" / "models"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback model factories
# ---------------------------------------------------------------------------


def _make_untrained_rf(n_features: int) -> "RandomForestClassifier":
    """Return an RF classifier fitted on minimal dummy data so predict_proba works."""
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.zeros((4, n_features))
    y_dummy = [0, 1, 0, 1]
    clf.fit(X_dummy, y_dummy)
    return clf


def _make_untrained_xgb(n_features: int) -> "XGBClassifier":
    """Return an XGB classifier fitted on minimal dummy data so predict_proba works."""
    clf = XGBClassifier(
        n_estimators=10,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    X_dummy = np.zeros((4, n_features))
    y_dummy = [0, 1, 0, 1]
    clf.fit(X_dummy, y_dummy)
    return clf


def _make_identity_scaler(n_features: int) -> "StandardScaler":
    """Return a StandardScaler fitted on zero-mean, unit-variance dummy data."""
    scaler = StandardScaler()
    X_dummy = np.random.default_rng(42).standard_normal((20, n_features))
    scaler.fit(X_dummy)
    return scaler


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_joblib(filename: str, search_dirs: list[Path]):
    """Try to load a joblib file from multiple directories; return None on failure."""
    if not _JOBLIB_AVAILABLE:
        return None
    for directory in search_dirs:
        path = directory / filename
        if path.exists():
            try:
                obj = joblib.load(path)
                logger.info("Loaded '%s' from %s", filename, path)
                return obj
            except Exception as exc:
                logger.error(
                    "Failed to load '%s' from %s: %s", filename, path, exc
                )
    return None


def _load_scaler(disease: str) -> Optional["StandardScaler"]:
    """Load the disease-specific StandardScaler; fall back to identity scaler."""
    filename = f"scaler_{disease}.joblib"
    obj = _load_joblib(filename, [_TRAINING_MODELS_DIR])
    if obj is not None:
        return obj

    n_features = DISEASE_FEATURE_DIMS[disease]
    logger.error(
        "classical_ml_v2: scaler file '%s' not found in %s. "
        "Falling back to an identity StandardScaler — predictions will be unreliable.",
        filename,
        _TRAINING_MODELS_DIR,
    )
    if _SKLEARN_AVAILABLE:
        return _make_identity_scaler(n_features)
    return None


def _load_rf(disease: str) -> Optional["RandomForestClassifier"]:
    """Load the disease-specific RF model; fall back to an untrained model."""
    filename = f"rf_{disease}.joblib"
    obj = _load_joblib(filename, [_DATA_MODELS_DIR, _TRAINING_MODELS_DIR])
    if obj is not None:
        return obj

    n_features = DISEASE_FEATURE_DIMS[disease]
    logger.error(
        "classical_ml_v2: RF model file '%s' not found in %s or %s. "
        "Falling back to an untrained RandomForestClassifier — predictions will be unreliable.",
        filename,
        _DATA_MODELS_DIR,
        _TRAINING_MODELS_DIR,
    )
    if _SKLEARN_AVAILABLE:
        return _make_untrained_rf(n_features)
    return None


def _load_xgb(disease: str) -> Optional["XGBClassifier"]:
    """Load the disease-specific XGBoost model; fall back to an untrained model."""
    filename = f"xgb_{disease}.joblib"
    obj = _load_joblib(filename, [_DATA_MODELS_DIR, _TRAINING_MODELS_DIR])
    if obj is not None:
        return obj

    n_features = DISEASE_FEATURE_DIMS[disease]
    logger.error(
        "classical_ml_v2: XGBoost model file '%s' not found in %s or %s. "
        "Falling back to an untrained XGBClassifier — predictions will be unreliable.",
        filename,
        _DATA_MODELS_DIR,
        _TRAINING_MODELS_DIR,
    )
    if _XGBOOST_AVAILABLE:
        return _make_untrained_xgb(n_features)
    return None


# ---------------------------------------------------------------------------
# Module-level loading (once at import time)
# ---------------------------------------------------------------------------

def _load_all() -> dict[str, dict]:
    """Load scalers, RF, and XGBoost models for all diseases at startup.

    Returns a dict keyed by disease name, each containing:
        "scaler": StandardScaler | None
        "rf":     RandomForestClassifier | None
        "xgb":    XGBClassifier | None
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning(
            "classical_ml_v2: scikit-learn is not installed. "
            "All predictions will use the deterministic mock fallback."
        )
        return {}

    result: dict[str, dict] = {}
    for disease in DISEASE_NAMES:
        result[disease] = {
            "scaler": _load_scaler(disease),
            "rf": _load_rf(disease),
            "xgb": _load_xgb(disease),
        }
    return result


_models: dict[str, dict] = _load_all()


# ---------------------------------------------------------------------------
# Mock fallback (used when sklearn is not installed)
# ---------------------------------------------------------------------------


def _mock_predict_v2(features: list[float], disease: str) -> tuple[float, float]:
    """Deterministic mock prediction for testing without sklearn/xgboost."""
    x = np.array(features, dtype=float)
    disease_idx = list(DISEASE_NAMES).index(disease)
    rf_prob = float(
        np.clip(np.abs(np.cos(np.sum(x) * (disease_idx + 1) * 0.07)), 0.0, 1.0)
    )
    xgb_prob = float(
        np.clip(np.abs(np.sin(np.sum(x) * (disease_idx + 1) * 0.05)), 0.0, 1.0)
    )
    return rf_prob, xgb_prob


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_classical_v2(
    features: list[float],
    disease: str,
    scaler: Optional["StandardScaler"] = None,
) -> tuple[float, float]:
    """Run the v2 RF + XGBoost ensemble for a single disease.

    Applies StandardScaler normalization before prediction. Uses the
    module-level loaded scaler unless an explicit ``scaler`` is provided
    (useful for testing or per-request override).

    Parameters
    ----------
    features:
        Disease-specific feature vector. Must have exactly the number of
        elements defined in ``DISEASE_FEATURE_DIMS`` for the given disease
        (6 for diabetes/cvd, 5 for ckd).
    disease:
        One of "diabetes", "cvd", "ckd".
    scaler:
        Optional pre-loaded StandardScaler. If None, the module-level
        scaler loaded at startup is used.

    Returns
    -------
    (rf_prob, xgb_prob)
        Tuple of probabilities in [0, 1]. Each value is the positive-class
        probability from the respective model. If a model is unavailable,
        its probability falls back to a deterministic mock value.

    Raises
    ------
    ValueError
        If ``disease`` is not one of the supported disease identifiers, or
        if the feature vector has the wrong number of elements.
    """
    if disease not in DISEASE_NAMES:
        raise ValueError(
            f"Unsupported disease '{disease}'. Must be one of: {DISEASE_NAMES}"
        )

    expected_dim = DISEASE_FEATURE_DIMS[disease]
    if len(features) != expected_dim:
        raise ValueError(
            f"Disease '{disease}' expects {expected_dim} features, "
            f"got {len(features)}."
        )

    if not _SKLEARN_AVAILABLE or not _models:
        return _mock_predict_v2(features, disease)

    disease_bundle = _models.get(disease, {})

    # Resolve scaler: prefer explicit argument, then module-level loaded scaler
    active_scaler: Optional["StandardScaler"] = scaler or disease_bundle.get("scaler")

    x = np.array(features, dtype=float).reshape(1, -1)

    # Apply StandardScaler normalization (Req 2.1)
    if active_scaler is not None:
        try:
            x = active_scaler.transform(x)
        except Exception as exc:
            logger.warning(
                "classical_ml_v2: StandardScaler.transform() failed for '%s': %s. "
                "Proceeding with unscaled features.",
                disease,
                exc,
            )

    # RF prediction
    rf_model = disease_bundle.get("rf")
    rf_prob: float
    if rf_model is not None:
        try:
            rf_prob = float(rf_model.predict_proba(x)[0][1])
        except Exception as exc:
            logger.error(
                "classical_ml_v2: RF predict_proba failed for '%s': %s. "
                "Using mock fallback.",
                disease,
                exc,
            )
            rf_prob, _ = _mock_predict_v2(features, disease)
    else:
        logger.warning(
            "classical_ml_v2: RF model unavailable for '%s'. Using mock fallback.",
            disease,
        )
        rf_prob, _ = _mock_predict_v2(features, disease)

    # XGBoost prediction
    xgb_model = disease_bundle.get("xgb")
    xgb_prob: float
    if xgb_model is not None and _XGBOOST_AVAILABLE:
        try:
            xgb_prob = float(xgb_model.predict_proba(x)[0][1])
        except Exception as exc:
            logger.error(
                "classical_ml_v2: XGBoost predict_proba failed for '%s': %s. "
                "Using mock fallback.",
                disease,
                exc,
            )
            _, xgb_prob = _mock_predict_v2(features, disease)
    else:
        if not _XGBOOST_AVAILABLE:
            logger.warning(
                "classical_ml_v2: xgboost is not installed. "
                "Using mock fallback for XGBoost prediction for '%s'.",
                disease,
            )
        else:
            logger.warning(
                "classical_ml_v2: XGBoost model unavailable for '%s'. Using mock fallback.",
                disease,
            )
        _, xgb_prob = _mock_predict_v2(features, disease)

    # Clamp to [0, 1] as a safety guard
    rf_prob = float(np.clip(rf_prob, 0.0, 1.0))
    xgb_prob = float(np.clip(xgb_prob, 0.0, 1.0))

    return rf_prob, xgb_prob


def get_loaded_models() -> dict[str, dict]:
    """Return the module-level loaded model bundle (for inspection/testing)."""
    return _models
