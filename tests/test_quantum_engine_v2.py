"""Unit tests for the quantum engine v2 (backend/services/quantum_engine_v2.py).

Covers:
- Disease-specific feature vector acceptance and dimension validation
- ValueError for unsupported disease names
- ValueError for wrong feature vector length
- PCA transformation path (mocked scaler + PCA)
- VQC circuit execution path (mocked PennyLane circuit)
- Timeout handling: QuantumTimeoutError raised when VQC exceeds 800ms
- Mock fallback when model artifacts are missing
- Output probability clamped to [0, 1]
- _expval_to_prob mapping correctness
- get_loaded_models() returns expected structure

Requirements: 3.1, 3.2, 3.3, 9.3
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.services.quantum_engine_v2 import (
    DISEASE_FEATURE_DIMS,
    DISEASE_NAMES,
    N_QUBITS,
    VQC_TIMEOUT_SECONDS,
    QuantumTimeoutError,
    _expval_to_prob,
    _mock_predict_v2,
    _run_vqc_inference,
    get_loaded_models,
    predict_quantum_v2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(disease: str) -> list[float]:
    """Return a valid feature vector of the correct length for the given disease."""
    dim = DISEASE_FEATURE_DIMS[disease]
    return [float(i + 1) for i in range(dim)]


def _make_mock_scaler(n_features: int) -> MagicMock:
    """Return a mock StandardScaler whose transform() returns a scaled array."""
    scaler = MagicMock()
    scaler.transform = MagicMock(
        side_effect=lambda x: np.array(x, dtype=float) * 0.5
    )
    return scaler


def _make_mock_pca(n_components: int = N_QUBITS) -> MagicMock:
    """Return a mock PCA whose transform() returns an n_components-dim array."""
    pca = MagicMock()
    pca.transform = MagicMock(
        side_effect=lambda x: np.ones((x.shape[0], n_components), dtype=float) * 0.1
    )
    return pca


def _make_mock_weights(n_layers: int, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Return a dummy weight array matching StronglyEntanglingLayers shape."""
    # StronglyEntanglingLayers weight shape: (n_layers, n_qubits, 3)
    return np.zeros((n_layers, n_qubits, 3), dtype=float)


# ---------------------------------------------------------------------------
# 1. Disease-specific feature vector acceptance (Req 3.1)
# ---------------------------------------------------------------------------

class TestFeatureVectorAcceptance:
    """Req 3.1 — each disease accepts only its correct-dimension feature vector."""

    @pytest.mark.parametrize("disease,expected_dim", [
        ("diabetes", 6),
        ("cvd", 6),
        ("ckd", 5),
    ])
    def test_correct_dimension_accepted(self, disease: str, expected_dim: int):
        """predict_quantum_v2 does not raise for a correctly-sized feature vector."""
        features = _make_features(disease)
        assert len(features) == expected_dim

        scaler = _make_mock_scaler(expected_dim)
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        bundle = {
            "scaler": scaler,
            "pca": pca,
            "weights": weights,
            "n_layers": 2,
        }
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=MagicMock(return_value=0.4),
        ):
            result = predict_quantum_v2(features, disease)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_wrong_dimension_raises_value_error(self, disease: str):
        """predict_quantum_v2 raises ValueError when feature vector has wrong length."""
        correct_dim = DISEASE_FEATURE_DIMS[disease]
        wrong_features = [1.0] * (correct_dim + 1)  # one extra element

        with pytest.raises(ValueError, match=str(correct_dim)):
            predict_quantum_v2(wrong_features, disease)

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_too_few_features_raises_value_error(self, disease: str):
        """predict_quantum_v2 raises ValueError when feature vector is too short."""
        correct_dim = DISEASE_FEATURE_DIMS[disease]
        short_features = [1.0] * max(1, correct_dim - 1)

        with pytest.raises(ValueError, match=str(correct_dim)):
            predict_quantum_v2(short_features, disease)

    def test_unsupported_disease_raises_value_error(self):
        """predict_quantum_v2 raises ValueError for an unknown disease name."""
        with pytest.raises(ValueError, match="Unsupported disease"):
            predict_quantum_v2([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "flu")

    def test_empty_disease_string_raises_value_error(self):
        """predict_quantum_v2 raises ValueError for an empty disease string."""
        with pytest.raises(ValueError, match="Unsupported disease"):
            predict_quantum_v2([1.0] * 6, "")

    def test_disease_names_constant_covers_all_three(self):
        """DISEASE_NAMES contains exactly diabetes, cvd, and ckd."""
        assert set(DISEASE_NAMES) == {"diabetes", "cvd", "ckd"}

    def test_disease_feature_dims_match_spec(self):
        """DISEASE_FEATURE_DIMS matches the spec: diabetes=6, cvd=6, ckd=5."""
        assert DISEASE_FEATURE_DIMS["diabetes"] == 6
        assert DISEASE_FEATURE_DIMS["cvd"] == 6
        assert DISEASE_FEATURE_DIMS["ckd"] == 5


# ---------------------------------------------------------------------------
# 2. PCA transformation (Req 3.1, 3.2)
# ---------------------------------------------------------------------------

class TestPCATransformation:
    """Req 3.1, 3.2 — StandardScaler.transform() then PCA.transform() are applied."""

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_scaler_transform_called_with_feature_array(self, disease: str):
        """StandardScaler.transform() is called with the input feature vector."""
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=MagicMock(return_value=0.0),
        ):
            _run_vqc_inference(features, disease)

        scaler.transform.assert_called_once()
        call_arg = scaler.transform.call_args[0][0]
        np.testing.assert_array_almost_equal(
            call_arg.flatten(), np.array(features, dtype=float)
        )

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_pca_transform_called_after_scaler(self, disease: str):
        """PCA.transform() is called with the scaler output."""
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=MagicMock(return_value=0.0),
        ):
            _run_vqc_inference(features, disease)

        pca.transform.assert_called_once()

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_pca_output_fed_to_circuit(self, disease: str):
        """The PCA-reduced vector (not the raw features) is passed to the VQC circuit."""
        features = _make_features(disease)
        pca_output = np.array([[0.11, 0.22, 0.33, 0.44, 0.55, 0.66]])
        scaler = _make_mock_scaler(len(features))
        pca = MagicMock()
        pca.transform = MagicMock(return_value=pca_output)
        weights = _make_mock_weights(n_layers=2)

        circuit_mock = MagicMock(return_value=0.5)
        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=circuit_mock,
        ):
            _run_vqc_inference(features, disease)

        circuit_mock.assert_called_once()
        called_features = circuit_mock.call_args[0][0]
        np.testing.assert_array_almost_equal(called_features, pca_output[0])

    def test_scaler_failure_falls_back_to_mock(self):
        """When StandardScaler.transform() raises, mock fallback is used."""
        disease = "diabetes"
        features = _make_features(disease)
        scaler = MagicMock()
        scaler.transform = MagicMock(side_effect=RuntimeError("scaler error"))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ):
            result = _run_vqc_inference(features, disease)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_pca_failure_falls_back_to_mock(self):
        """When PCA.transform() raises, mock fallback is used."""
        disease = "cvd"
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = MagicMock()
        pca.transform = MagicMock(side_effect=RuntimeError("pca error"))
        weights = _make_mock_weights(n_layers=2)

        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ):
            result = _run_vqc_inference(features, disease)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 3. VQC circuit execution (Req 3.2, 3.3)
# ---------------------------------------------------------------------------

class TestVQCCircuitExecution:
    """Req 3.2, 3.3 — VQC circuit is called with PCA output and trained weights."""

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_circuit_called_with_weights(self, disease: str):
        """The VQC circuit is invoked with the loaded weight array."""
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=3)

        circuit_mock = MagicMock(return_value=-0.2)
        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 3}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=circuit_mock,
        ):
            _run_vqc_inference(features, disease)

        circuit_mock.assert_called_once()
        _, call_kwargs = circuit_mock.call_args
        # weights are passed as second positional arg
        passed_weights = circuit_mock.call_args[0][1]
        np.testing.assert_array_equal(passed_weights, weights)

    @pytest.mark.parametrize("expval,expected_prob", [
        (-1.0, 0.0),   # minimum expval → probability 0
        (1.0, 1.0),    # maximum expval → probability 1
        (0.0, 0.5),    # neutral expval → probability 0.5
        (-0.6, 0.2),   # intermediate negative
        (0.6, 0.8),    # intermediate positive
    ])
    def test_expval_to_prob_mapping(self, expval: float, expected_prob: float):
        """_expval_to_prob maps PauliZ expectation value to [0,1] probability."""
        result = _expval_to_prob(expval)
        assert abs(result - expected_prob) < 1e-9, (
            f"expval={expval}: expected {expected_prob}, got {result}"
        )

    def test_expval_to_prob_clamps_above_one(self):
        """_expval_to_prob clamps values above 1.0 to 1.0."""
        assert _expval_to_prob(1.5) == 1.0

    def test_expval_to_prob_clamps_below_zero(self):
        """_expval_to_prob clamps values below 0.0 to 0.0."""
        assert _expval_to_prob(-1.5) == 0.0

    def test_circuit_failure_falls_back_to_mock(self):
        """When the VQC circuit raises, mock fallback is used and result is in [0,1]."""
        disease = "ckd"
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        circuit_mock = MagicMock(side_effect=RuntimeError("circuit error"))
        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=circuit_mock,
        ):
            result = _run_vqc_inference(features, disease)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("n_layers", [2, 3, 4])
    def test_get_circuit_called_with_correct_layer_count(self, n_layers: int):
        """_get_circuit is called with the disease-specific best layer count."""
        disease = "diabetes"
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=n_layers)

        circuit_mock = MagicMock(return_value=0.0)
        bundle = {
            "scaler": scaler,
            "pca": pca,
            "weights": weights,
            "n_layers": n_layers,
        }
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ) as _, patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=circuit_mock,
        ) as mock_get_circuit:
            _run_vqc_inference(features, disease)

        mock_get_circuit.assert_called_once_with(n_layers, N_QUBITS)

    def test_output_probability_in_unit_interval(self):
        """predict_quantum_v2 always returns a value in [0, 1]."""
        disease = "diabetes"
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        # Circuit returns an extreme expval that would map outside [0,1] without clamping
        circuit_mock = MagicMock(return_value=2.0)
        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=circuit_mock,
        ):
            result = predict_quantum_v2(features, disease)

        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 4. Timeout handling (Req 9.3)
# ---------------------------------------------------------------------------

class TestTimeoutHandling:
    """Req 9.3 — QuantumTimeoutError is raised when VQC inference exceeds 800ms."""

    def test_quantum_timeout_error_raised_on_slow_inference(self):
        """predict_quantum_v2 raises QuantumTimeoutError when inference is too slow."""
        disease = "diabetes"
        features = _make_features(disease)

        def _slow_inference(features, disease):
            # Sleep longer than the 800ms budget
            time.sleep(VQC_TIMEOUT_SECONDS + 0.5)
            return 0.5

        with patch(
            "backend.services.quantum_engine_v2._run_vqc_inference",
            side_effect=_slow_inference,
        ):
            with pytest.raises(QuantumTimeoutError):
                predict_quantum_v2(features, disease)

    def test_quantum_timeout_error_message_contains_disease(self):
        """QuantumTimeoutError message identifies the disease that timed out."""
        disease = "cvd"
        features = _make_features(disease)

        def _slow_inference(features, disease):
            time.sleep(VQC_TIMEOUT_SECONDS + 0.5)
            return 0.5

        with patch(
            "backend.services.quantum_engine_v2._run_vqc_inference",
            side_effect=_slow_inference,
        ):
            with pytest.raises(QuantumTimeoutError, match=disease):
                predict_quantum_v2(features, disease)

    def test_quantum_timeout_error_message_contains_ms_budget(self):
        """QuantumTimeoutError message includes the 800ms budget."""
        disease = "ckd"
        features = _make_features(disease)

        def _slow_inference(features, disease):
            time.sleep(VQC_TIMEOUT_SECONDS + 0.5)
            return 0.5

        with patch(
            "backend.services.quantum_engine_v2._run_vqc_inference",
            side_effect=_slow_inference,
        ):
            with pytest.raises(QuantumTimeoutError, match="800"):
                predict_quantum_v2(features, disease)

    def test_fast_inference_does_not_raise_timeout(self):
        """predict_quantum_v2 does not raise when inference completes within budget."""
        disease = "diabetes"
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        circuit_mock = MagicMock(return_value=0.3)
        bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ), patch(
            "backend.services.quantum_engine_v2._get_circuit",
            return_value=circuit_mock,
        ):
            # Should complete without raising
            result = predict_quantum_v2(features, disease)

        assert isinstance(result, float)

    def test_vqc_timeout_seconds_constant_is_0_8(self):
        """VQC_TIMEOUT_SECONDS is set to 0.8 (800ms) as per Req 9.3."""
        assert VQC_TIMEOUT_SECONDS == 0.8


# ---------------------------------------------------------------------------
# 5. Mock fallback when model artifacts are missing
# ---------------------------------------------------------------------------

class TestMockFallback:
    """When model artifacts are missing, _run_vqc_inference falls back to mock."""

    def test_missing_scaler_uses_mock_fallback(self):
        """When scaler is None, mock fallback is used and result is in [0, 1]."""
        disease = "diabetes"
        features = _make_features(disease)
        bundle = {
            "scaler": None,
            "pca": _make_mock_pca(N_QUBITS),
            "weights": _make_mock_weights(n_layers=2),
            "n_layers": 2,
        }
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ):
            result = _run_vqc_inference(features, disease)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_missing_pca_uses_mock_fallback(self):
        """When PCA is None, mock fallback is used and result is in [0, 1]."""
        disease = "cvd"
        features = _make_features(disease)
        bundle = {
            "scaler": _make_mock_scaler(len(features)),
            "pca": None,
            "weights": _make_mock_weights(n_layers=2),
            "n_layers": 2,
        }
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ):
            result = _run_vqc_inference(features, disease)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_missing_weights_uses_mock_fallback(self):
        """When weights are None, mock fallback is used and result is in [0, 1]."""
        disease = "ckd"
        features = _make_features(disease)
        bundle = {
            "scaler": _make_mock_scaler(len(features)),
            "pca": _make_mock_pca(N_QUBITS),
            "weights": None,
            "n_layers": 2,
        }
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ):
            result = _run_vqc_inference(features, disease)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_all_artifacts_missing_uses_mock_fallback(self):
        """When all artifacts are None, mock fallback is used for all diseases."""
        for disease in DISEASE_NAMES:
            features = _make_features(disease)
            bundle = {"scaler": None, "pca": None, "weights": None, "n_layers": 2}
            with patch(
                "backend.services.quantum_engine_v2._models",
                {disease: bundle},
            ):
                result = _run_vqc_inference(features, disease)

            assert isinstance(result, float), f"Expected float for {disease}"
            assert 0.0 <= result <= 1.0, f"Result out of range for {disease}: {result}"

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_mock_predict_v2_returns_float_in_unit_interval(self, disease: str):
        """_mock_predict_v2 always returns a float in [0, 1]."""
        features = _make_features(disease)
        result = _mock_predict_v2(features, disease)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_mock_predict_v2_is_deterministic(self, disease: str):
        """_mock_predict_v2 returns the same value for the same inputs."""
        features = _make_features(disease)
        result1 = _mock_predict_v2(features, disease)
        result2 = _mock_predict_v2(features, disease)
        assert result1 == result2

    def test_mock_predict_v2_differs_across_diseases(self):
        """_mock_predict_v2 produces different values for different diseases."""
        # Use the same feature values (padded to max dim) to isolate disease effect
        results = {}
        for disease in DISEASE_NAMES:
            features = _make_features(disease)
            results[disease] = _mock_predict_v2(features, disease)
        # At least two diseases should produce different mock values
        assert len(set(results.values())) > 1, (
            "Mock predictions should differ across diseases"
        )


# ---------------------------------------------------------------------------
# 6. Output probability range
# ---------------------------------------------------------------------------

class TestOutputProbabilityRange:
    """predict_quantum_v2 always returns a value in [0, 1]."""

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_result_in_unit_interval_with_valid_circuit(self, disease: str):
        """Result is in [0, 1] when circuit returns a valid expval."""
        features = _make_features(disease)
        scaler = _make_mock_scaler(len(features))
        pca = _make_mock_pca(N_QUBITS)
        weights = _make_mock_weights(n_layers=2)

        for expval in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            circuit_mock = MagicMock(return_value=expval)
            bundle = {"scaler": scaler, "pca": pca, "weights": weights, "n_layers": 2}
            with patch(
                "backend.services.quantum_engine_v2._models",
                {disease: bundle},
            ), patch(
                "backend.services.quantum_engine_v2._get_circuit",
                return_value=circuit_mock,
            ):
                result = predict_quantum_v2(features, disease)

            assert 0.0 <= result <= 1.0, (
                f"Result {result} out of [0,1] for disease={disease}, expval={expval}"
            )

    @pytest.mark.parametrize("disease", list(DISEASE_NAMES))
    def test_result_in_unit_interval_with_mock_fallback(self, disease: str):
        """Result is in [0, 1] even when mock fallback is used."""
        features = _make_features(disease)
        bundle = {"scaler": None, "pca": None, "weights": None, "n_layers": 2}
        with patch(
            "backend.services.quantum_engine_v2._models",
            {disease: bundle},
        ):
            result = predict_quantum_v2(features, disease)

        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 7. get_loaded_models() structure
# ---------------------------------------------------------------------------

class TestGetLoadedModels:
    """get_loaded_models() returns the expected per-disease bundle structure."""

    def test_returns_dict_with_all_diseases(self):
        """get_loaded_models() returns a dict keyed by all three disease names."""
        models = get_loaded_models()
        assert set(models.keys()) == set(DISEASE_NAMES)

    def test_each_bundle_has_required_keys(self):
        """Each disease bundle contains scaler, pca, weights, and n_layers keys."""
        models = get_loaded_models()
        required_keys = {"scaler", "pca", "weights", "n_layers"}
        for disease, bundle in models.items():
            assert required_keys.issubset(bundle.keys()), (
                f"Bundle for '{disease}' missing keys: "
                f"{required_keys - set(bundle.keys())}"
            )

    def test_n_layers_is_integer(self):
        """n_layers in each bundle is an integer."""
        models = get_loaded_models()
        for disease, bundle in models.items():
            assert isinstance(bundle["n_layers"], int), (
                f"n_layers for '{disease}' should be int, got {type(bundle['n_layers'])}"
            )

    def test_n_layers_in_valid_range(self):
        """n_layers in each bundle is between 2 and 4 (valid VQC layer counts)."""
        models = get_loaded_models()
        for disease, bundle in models.items():
            assert 2 <= bundle["n_layers"] <= 4, (
                f"n_layers for '{disease}' = {bundle['n_layers']}, expected 2–4"
            )
