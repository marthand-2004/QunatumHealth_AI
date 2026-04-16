"""Quantum VQC prediction engine using PennyLane.

Builds a 6-qubit VQC circuit (features reduced via PCA from 14 → 6):
  AngleEmbedding → 3× StronglyEntanglingLayers → expectation values → [0, 100]

Requirements: 5.2, 5.3, 5.4, 5.6, 5.7
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# PennyLane import with mock fallback
# ---------------------------------------------------------------------------
try:
    import pennylane as qml  # type: ignore

    _PENNYLANE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PENNYLANE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Circuit constants
# ---------------------------------------------------------------------------
N_QUBITS = 6          # 14 features → 6 via PCA
N_LAYERS = 3          # StronglyEntanglingLayers depth
N_DISEASES = 3        # Diabetes, CVD, CKD
DISEASE_NAMES = ["diabetes", "cvd", "ckd"]

# ---------------------------------------------------------------------------
# PCA projection matrix (14 → 6) — random orthonormal basis (untrained)
# Seeded for reproducibility; will be replaced by trained weights later.
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(42)
_raw = _rng.standard_normal((14, N_QUBITS))
_pca_matrix, _, _ = np.linalg.svd(_raw, full_matrices=False)
# _pca_matrix shape: (14, 6)

# ---------------------------------------------------------------------------
# Random (untrained) VQC weights — shape: (N_DISEASES, N_LAYERS, N_QUBITS, 3)
# ---------------------------------------------------------------------------
_weight_rng = np.random.default_rng(7)
_vqc_weights: np.ndarray = _weight_rng.uniform(
    -np.pi, np.pi, size=(N_DISEASES, N_LAYERS, N_QUBITS, 3)
)


def _project_features(features: list[float]) -> np.ndarray:
    """Project 14-dim feature vector to 6-dim via PCA matrix."""
    x = np.array(features, dtype=float)
    return x @ _pca_matrix  # shape: (6,)


# ---------------------------------------------------------------------------
# PennyLane circuit (built once per disease index)
# ---------------------------------------------------------------------------
def _build_circuit(disease_idx: int):
    """Return a QNode for the given disease index."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(features: np.ndarray, weights: np.ndarray) -> float:
        qml.AngleEmbedding(features, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return circuit


# Build circuits lazily (only when PennyLane is available)
_circuits: list | None = None


def _get_circuits() -> list:
    global _circuits
    if _circuits is None:
        _circuits = [_build_circuit(i) for i in range(N_DISEASES)]
    return _circuits


# ---------------------------------------------------------------------------
# Mock fallback (used when PennyLane is not installed)
# ---------------------------------------------------------------------------
def _mock_predict(features: list[float]) -> dict[str, float]:
    """Deterministic mock prediction for testing without PennyLane."""
    x = np.array(features, dtype=float)
    scores: dict[str, float] = {}
    for i, name in enumerate(DISEASE_NAMES):
        # Simple deterministic hash-like value in [0, 100]
        val = float(np.clip(np.abs(np.sin(np.sum(x) * (i + 1) * 0.1)) * 100, 0.0, 100.0))
        scores[name] = round(val, 4)
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_quantum(features: list[float]) -> dict[str, float]:
    """Run VQC circuit and return risk scores for Diabetes, CVD, CKD.

    Parameters
    ----------
    features:
        14-dimensional feature vector (must have exactly 14 elements).

    Returns
    -------
    dict with keys "diabetes", "cvd", "ckd" — each value in [0, 100].
    """
    if len(features) != 14:
        raise ValueError(f"Expected 14 features, got {len(features)}")

    if not _PENNYLANE_AVAILABLE:
        return _mock_predict(features)

    projected = _project_features(features)
    circuits = _get_circuits()
    scores: dict[str, float] = {}

    for i, name in enumerate(DISEASE_NAMES):
        weights = _vqc_weights[i]  # shape: (N_LAYERS, N_QUBITS, 3)
        expval = float(circuits[i](projected, weights))
        # Map [-1, 1] → [0, 100]
        score = (expval + 1.0) * 50.0
        scores[name] = round(float(np.clip(score, 0.0, 100.0)), 4)

    return scores
