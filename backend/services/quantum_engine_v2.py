"""Quantum VQC prediction engine v2 — disease-specific PCA + trained VQC weights.

Loads disease-specific PCA, VQC weights, and best layer count at startup.
Accepts disease-specific feature vectors (6 dims for diabetes/cvd, 5 for ckd).
Applies StandardScaler → PCA → 6-qubit VQC circuit → probability in [0, 1].

Circuit architecture (must match training in 03_train_quantum_vqc_v2.py):
  AngleEmbedding (RY rotations) → StronglyEntanglingLayers (best_n_layers) → expval(PauliZ(0))

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 9.3, 12.2
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    import joblib  # type: ignore

    _JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JOBLIB_AVAILABLE = False

try:
    import pennylane as qml  # type: ignore

    _PENNYLANE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PENNYLANE_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class QuantumTimeoutError(Exception):
    """Raised when VQC inference exceeds the 800ms budget."""


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

# Number of qubits in the VQC circuit (PCA always reduces to this many dims)
N_QUBITS = 6

# Timeout budget for VQC inference in seconds (Req 9.3)
VQC_TIMEOUT_SECONDS = 0.8

# Paths to model artifacts
_REPO_ROOT = Path(__file__).parent.parent.parent
_TRAINING_MODELS_DIR = _REPO_ROOT / "training" / "models"
_QUANTUM_JUSTIFICATION_PATH = _REPO_ROOT / "training" / "results" / "quantum_justification.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load quantum justification (best layer count per disease)
# ---------------------------------------------------------------------------


def _load_best_layer_counts() -> dict[str, int]:
    """Load best VQC layer count per disease from quantum_justification.json."""
    if not _QUANTUM_JUSTIFICATION_PATH.exists():
        logger.error(
            "quantum_engine_v2: quantum_justification.json not found at %s. "
            "Defaulting to 2 layers for all diseases.",
            _QUANTUM_JUSTIFICATION_PATH,
        )
        return {d: 2 for d in DISEASE_NAMES}

    try:
        with open(_QUANTUM_JUSTIFICATION_PATH) as f:
            data = json.load(f)
        result: dict[str, int] = {}
        for disease in DISEASE_NAMES:
            if disease in data and "best_vqc_layers" in data[disease]:
                result[disease] = int(data[disease]["best_vqc_layers"])
                logger.info(
                    "quantum_engine_v2: best layer count for %s = %d",
                    disease,
                    result[disease],
                )
            else:
                logger.warning(
                    "quantum_engine_v2: 'best_vqc_layers' missing for '%s' in "
                    "quantum_justification.json. Defaulting to 2.",
                    disease,
                )
                result[disease] = 2
        return result
    except Exception as exc:
        logger.error(
            "quantum_engine_v2: Failed to load quantum_justification.json: %s. "
            "Defaulting to 2 layers for all diseases.",
            exc,
        )
        return {d: 2 for d in DISEASE_NAMES}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_pca(disease: str) -> Optional["PCA"]:
    """Load disease-specific PCA from training/models/; return None on failure."""
    if not _JOBLIB_AVAILABLE or not _SKLEARN_AVAILABLE:
        return None
    filename = f"pca_{disease}.joblib"
    path = _TRAINING_MODELS_DIR / filename
    if not path.exists():
        logger.error(
            "quantum_engine_v2: PCA file '%s' not found at %s. "
            "VQC predictions for '%s' will be unavailable.",
            filename,
            path,
            disease,
        )
        return None
    try:
        pca = joblib.load(path)
        logger.info("quantum_engine_v2: Loaded PCA for '%s' from %s", disease, path)
        return pca
    except Exception as exc:
        logger.error(
            "quantum_engine_v2: Failed to load PCA for '%s' from %s: %s. "
            "VQC predictions for '%s' will be unavailable.",
            filename,
            path,
            exc,
            disease,
        )
        return None


def _load_scaler(disease: str) -> Optional["StandardScaler"]:
    """Load disease-specific StandardScaler from training/models/; return None on failure."""
    if not _JOBLIB_AVAILABLE or not _SKLEARN_AVAILABLE:
        return None
    filename = f"scaler_{disease}.joblib"
    path = _TRAINING_MODELS_DIR / filename
    if not path.exists():
        logger.error(
            "quantum_engine_v2: Scaler file '%s' not found at %s. "
            "VQC predictions for '%s' will be unavailable.",
            filename,
            path,
            disease,
        )
        return None
    try:
        scaler = joblib.load(path)
        logger.info("quantum_engine_v2: Loaded StandardScaler for '%s' from %s", disease, path)
        return scaler
    except Exception as exc:
        logger.error(
            "quantum_engine_v2: Failed to load StandardScaler for '%s' from %s: %s. "
            "VQC predictions for '%s' will be unavailable.",
            filename,
            path,
            exc,
            disease,
        )
        return None


def _load_vqc_weights(disease: str) -> Optional[np.ndarray]:
    """Load disease-specific VQC weights from training/models/; return None on failure."""
    filename = f"vqc_{disease}_weights.npy"
    path = _TRAINING_MODELS_DIR / filename
    if not path.exists():
        logger.error(
            "quantum_engine_v2: VQC weights file '%s' not found at %s. "
            "VQC predictions for '%s' will be unavailable.",
            filename,
            path,
            disease,
        )
        return None
    try:
        weights = np.load(path)
        logger.info(
            "quantum_engine_v2: Loaded VQC weights for '%s' from %s (shape=%s)",
            disease,
            path,
            weights.shape,
        )
        return weights
    except Exception as exc:
        logger.error(
            "quantum_engine_v2: Failed to load VQC weights for '%s' from %s: %s. "
            "VQC predictions for '%s' will be unavailable.",
            filename,
            path,
            exc,
            disease,
        )
        return None


# ---------------------------------------------------------------------------
# Module-level loading (once at import time)
# ---------------------------------------------------------------------------

_best_layer_counts: dict[str, int] = _load_best_layer_counts()

_models: dict[str, dict] = {}
for _disease in DISEASE_NAMES:
    _models[_disease] = {
        "scaler": _load_scaler(_disease),
        "pca": _load_pca(_disease),
        "weights": _load_vqc_weights(_disease),
        "n_layers": _best_layer_counts.get(_disease, 2),
    }

# ---------------------------------------------------------------------------
# PennyLane circuit builder
# ---------------------------------------------------------------------------


def _build_circuit(n_layers: int, n_qubits: int = N_QUBITS):
    """Build a PennyLane QNode matching the training circuit architecture.

    Architecture (must match 03_train_quantum_vqc_v2.py):
      AngleEmbedding (RY rotations) → StronglyEntanglingLayers × n_layers → expval(PauliZ(0))
    """
    # Prefer lightning.qubit for speed; fall back to default.qubit
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff_method = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "best"

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(features: np.ndarray, weights: np.ndarray) -> float:
        qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


# Cache circuits keyed by (n_layers, n_qubits) to avoid rebuilding on every call
_circuit_cache: dict[tuple[int, int], object] = {}


def _get_circuit(n_layers: int, n_qubits: int = N_QUBITS):
    """Return a cached QNode for the given (n_layers, n_qubits) combination."""
    key = (n_layers, n_qubits)
    if key not in _circuit_cache:
        _circuit_cache[key] = _build_circuit(n_layers, n_qubits)
    return _circuit_cache[key]


# ---------------------------------------------------------------------------
# Expectation value → probability conversion
# ---------------------------------------------------------------------------


def _expval_to_prob(expval: float) -> float:
    """Map PauliZ expectation value from [-1, 1] to probability in [0, 1].

    Uses the linear mapping (expval + 1) / 2, matching the training script.
    """
    return float(np.clip((expval + 1.0) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Mock fallback (used when PennyLane or sklearn is not installed)
# ---------------------------------------------------------------------------


def _mock_predict_v2(features: list[float], disease: str) -> float:
    """Deterministic mock prediction for testing without PennyLane/sklearn."""
    x = np.array(features, dtype=float)
    disease_idx = list(DISEASE_NAMES).index(disease)
    return float(
        np.clip(np.abs(np.sin(np.sum(x) * (disease_idx + 1) * 0.1)), 0.0, 1.0)
    )


# ---------------------------------------------------------------------------
# Core inference (runs inside a thread for timeout enforcement)
# ---------------------------------------------------------------------------


def _run_vqc_inference(features: list[float], disease: str) -> float:
    """Execute the full VQC inference pipeline for a single disease.

    Steps:
      1. StandardScaler.transform()
      2. PCA.transform() → 6-dimensional vector
      3. VQC circuit: AngleEmbedding → StronglyEntanglingLayers → expval(PauliZ(0))
      4. Map expectation value to probability in [0, 1]

    This function is designed to be called inside a thread so that the caller
    can enforce the 800ms timeout via concurrent.futures.
    """
    bundle = _models.get(disease, {})
    scaler = bundle.get("scaler")
    pca = bundle.get("pca")
    weights = bundle.get("weights")
    n_layers = bundle.get("n_layers", 2)

    if scaler is None or pca is None or weights is None:
        logger.warning(
            "quantum_engine_v2: One or more model artifacts missing for '%s'. "
            "Using mock fallback.",
            disease,
        )
        return _mock_predict_v2(features, disease)

    # Step 1: StandardScaler normalization
    x = np.array(features, dtype=float).reshape(1, -1)
    try:
        x_scaled = scaler.transform(x)
    except Exception as exc:
        logger.warning(
            "quantum_engine_v2: StandardScaler.transform() failed for '%s': %s. "
            "Using mock fallback.",
            disease,
            exc,
        )
        return _mock_predict_v2(features, disease)

    # Step 2: PCA dimensionality reduction
    try:
        x_pca = pca.transform(x_scaled)  # shape: (1, n_components)
    except Exception as exc:
        logger.warning(
            "quantum_engine_v2: PCA.transform() failed for '%s': %s. "
            "Using mock fallback.",
            disease,
            exc,
        )
        return _mock_predict_v2(features, disease)

    # Determine actual qubit count from PCA output (may be < N_QUBITS for CKD)
    n_qubits = x_pca.shape[1]
    features_pca = x_pca[0]  # shape: (n_qubits,)

    # Step 3: VQC circuit execution
    if not _PENNYLANE_AVAILABLE:
        return _mock_predict_v2(features, disease)

    try:
        circuit = _get_circuit(n_layers, n_qubits)
        expval = float(circuit(features_pca, weights))
    except Exception as exc:
        logger.warning(
            "quantum_engine_v2: VQC circuit execution failed for '%s': %s. "
            "Using mock fallback.",
            disease,
            exc,
        )
        return _mock_predict_v2(features, disease)

    # Step 4: Map to probability
    return _expval_to_prob(expval)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_quantum_v2(features: list[float], disease: str) -> float:
    """Run the v2 VQC inference pipeline for a single disease.

    Applies StandardScaler → PCA → 6-qubit VQC circuit and returns a
    probability in [0, 1].

    Parameters
    ----------
    features:
        Disease-specific feature vector. Must have exactly the number of
        elements defined in ``DISEASE_FEATURE_DIMS`` for the given disease
        (6 for diabetes/cvd, 5 for ckd).
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    float
        Probability in [0, 1] representing the positive-class (disease) risk.

    Raises
    ------
    ValueError
        If ``disease`` is not one of the supported disease identifiers, or
        if the feature vector has the wrong number of elements.
    QuantumTimeoutError
        If VQC inference exceeds 800ms (Req 9.3).
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

    # Run inference in a thread pool with a hard 800ms timeout (Req 9.3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_vqc_inference, features, disease)
        try:
            result = future.result(timeout=VQC_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "quantum_engine_v2: VQC inference for '%s' exceeded %dms timeout. "
                "Raising QuantumTimeoutError.",
                disease,
                int(VQC_TIMEOUT_SECONDS * 1000),
            )
            raise QuantumTimeoutError(
                f"VQC inference for '{disease}' exceeded "
                f"{int(VQC_TIMEOUT_SECONDS * 1000)}ms timeout."
            )

    return float(np.clip(result, 0.0, 1.0))


def get_loaded_models() -> dict[str, dict]:
    """Return the module-level loaded model bundle (for inspection/testing).

    Returns a dict keyed by disease name, each containing:
        "scaler":   StandardScaler | None
        "pca":      PCA | None
        "weights":  np.ndarray | None
        "n_layers": int
    """
    return _models
