"""Unit tests for the feature vector construction service.

Requirements: 5.1, 6.3, 2.5
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from datetime import datetime

from backend.models.prediction import FEATURE_NAMES, FEATURE_DIM, FeatureVector
from backend.services.feature_vector_service import (
    POPULATION_MEANS,
    SMOKING_ENCODING,
    _extract_lab_values,
    _encode_smoking,
    build_feature_array,
    build_feature_vector,
)


# ---------------------------------------------------------------------------
# _extract_lab_values
# ---------------------------------------------------------------------------

class TestExtractLabValues:
    def test_maps_known_lab_params(self):
        params = [
            {"name": "glucose", "value": 5.5},
            {"name": "hba1c", "value": 6.2},
            {"name": "creatinine", "value": 90.0},
        ]
        result = _extract_lab_values(params)
        assert result["glucose"] == 5.5
        assert result["hba1c"] == 6.2
        assert result["creatinine"] == 90.0

    def test_maps_total_cholesterol_to_cholesterol(self):
        params = [{"name": "total_cholesterol", "value": 4.8}]
        result = _extract_lab_values(params)
        assert result["cholesterol"] == 4.8

    def test_ignores_unknown_params(self):
        params = [{"name": "unknown_marker", "value": 99.0}]
        result = _extract_lab_values(params)
        assert "unknown_marker" not in result

    def test_first_occurrence_wins_for_duplicate_names(self):
        params = [
            {"name": "glucose", "value": 5.0},
            {"name": "glucose", "value": 9.0},
        ]
        result = _extract_lab_values(params)
        assert result["glucose"] == 5.0

    def test_empty_list_returns_empty_dict(self):
        assert _extract_lab_values([]) == {}


# ---------------------------------------------------------------------------
# _encode_smoking
# ---------------------------------------------------------------------------

class TestEncodeSmokingStatus:
    def test_never_encodes_to_zero(self):
        assert _encode_smoking("never") == 0.0

    def test_former_encodes_to_half(self):
        assert _encode_smoking("former") == 0.5

    def test_current_encodes_to_one(self):
        assert _encode_smoking("current") == 1.0

    def test_none_returns_population_mean(self):
        assert _encode_smoking(None) == POPULATION_MEANS["smoking_encoded"]

    def test_unknown_value_returns_population_mean(self):
        assert _encode_smoking("occasional") == POPULATION_MEANS["smoking_encoded"]


# ---------------------------------------------------------------------------
# build_feature_array
# ---------------------------------------------------------------------------

class TestBuildFeatureArray:
    def test_returns_exactly_14_features(self):
        features = build_feature_array({}, None)
        assert len(features) == FEATURE_DIM

    def test_feature_order_matches_feature_names(self):
        lab = {"glucose": 6.0, "hba1c": 6.5}
        lifestyle = {
            "bmi": 28.0,
            "exercise_frequency": 4,
            "sleep_hours": 7.5,
            "stress_level": 6,
            "smoking_status": "former",
        }
        features = build_feature_array(lab, lifestyle)
        assert len(features) == len(FEATURE_NAMES)
        # Verify specific positions
        assert features[FEATURE_NAMES.index("glucose")] == 6.0
        assert features[FEATURE_NAMES.index("hba1c")] == 6.5
        assert features[FEATURE_NAMES.index("bmi")] == 28.0
        assert features[FEATURE_NAMES.index("smoking_encoded")] == 0.5

    def test_missing_lab_values_use_population_means(self):
        features = build_feature_array({}, None)
        for i, name in enumerate(FEATURE_NAMES):
            assert features[i] == POPULATION_MEANS[name], (
                f"Feature '{name}' should default to {POPULATION_MEANS[name]}, got {features[i]}"
            )

    def test_lab_values_override_defaults(self):
        lab = {"glucose": 9.9, "creatinine": 150.0}
        features = build_feature_array(lab, None)
        assert features[FEATURE_NAMES.index("glucose")] == 9.9
        assert features[FEATURE_NAMES.index("creatinine")] == 150.0
        # Other features remain at population mean
        assert features[FEATURE_NAMES.index("hba1c")] == POPULATION_MEANS["hba1c"]

    def test_lifestyle_overrides_defaults(self):
        lifestyle = {
            "bmi": 30.5,
            "exercise_frequency": 2,
            "sleep_hours": 6.0,
            "stress_level": 8,
            "smoking_status": "current",
        }
        features = build_feature_array({}, lifestyle)
        assert features[FEATURE_NAMES.index("bmi")] == 30.5
        assert features[FEATURE_NAMES.index("exercise_frequency")] == 2.0
        assert features[FEATURE_NAMES.index("sleep_hours")] == 6.0
        assert features[FEATURE_NAMES.index("stress_level")] == 8.0
        assert features[FEATURE_NAMES.index("smoking_encoded")] == 1.0

    def test_no_lifestyle_uses_population_mean_for_lifestyle_features(self):
        features = build_feature_array({}, None)
        assert features[FEATURE_NAMES.index("bmi")] == POPULATION_MEANS["bmi"]
        assert features[FEATURE_NAMES.index("exercise_frequency")] == POPULATION_MEANS["exercise_frequency"]
        assert features[FEATURE_NAMES.index("sleep_hours")] == POPULATION_MEANS["sleep_hours"]
        assert features[FEATURE_NAMES.index("stress_level")] == POPULATION_MEANS["stress_level"]

    def test_all_features_are_floats(self):
        features = build_feature_array({"glucose": 5.5}, {"bmi": 25, "smoking_status": "never",
                                                            "exercise_frequency": 3, "sleep_hours": 7,
                                                            "stress_level": 5})
        assert all(isinstance(f, float) for f in features)

    def test_partial_lifestyle_partial_lab(self):
        """Only some fields provided — rest should be population means."""
        lab = {"hemoglobin": 13.5}
        lifestyle = {"bmi": 22.0, "smoking_status": "never"}
        features = build_feature_array(lab, lifestyle)
        assert features[FEATURE_NAMES.index("hemoglobin")] == 13.5
        assert features[FEATURE_NAMES.index("bmi")] == 22.0
        assert features[FEATURE_NAMES.index("smoking_encoded")] == 0.0
        # Unset lifestyle fields fall back to population mean
        assert features[FEATURE_NAMES.index("exercise_frequency")] == POPULATION_MEANS["exercise_frequency"]


# ---------------------------------------------------------------------------
# build_feature_vector (async, with mocked DB)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBuildFeatureVectorAsync:
    def _make_db(self, document: dict | None, lifestyle: dict | None):
        """Return a mock Motor DB with preset find_one responses."""
        db = MagicMock()

        docs_col = MagicMock()
        docs_col.find_one = AsyncMock(return_value=document)

        lifestyle_col = MagicMock()
        lifestyle_col.find_one = AsyncMock(return_value=lifestyle)

        fv_col = MagicMock()
        fv_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))

        def getitem(name):
            if name == "documents":
                return docs_col
            if name == "lifestyle_profiles":
                return lifestyle_col
            if name == "feature_vectors":
                return fv_col
            return MagicMock()

        db.__getitem__ = MagicMock(side_effect=getitem)
        return db

    async def test_returns_feature_vector_with_14_dims(self):
        user_oid = ObjectId()
        doc_oid = ObjectId()
        document = {
            "_id": doc_oid,
            "user_id": user_oid,
            "lab_parameters": [
                {"name": "glucose", "value": 5.5},
                {"name": "hba1c", "value": 5.8},
            ],
        }
        db = self._make_db(document, None)
        fv = await build_feature_vector(db, str(user_oid), str(doc_oid))
        assert isinstance(fv, FeatureVector)
        assert len(fv.features) == FEATURE_DIM

    async def test_feature_names_match_expected_order(self):
        user_oid = ObjectId()
        doc_oid = ObjectId()
        document = {"_id": doc_oid, "user_id": user_oid, "lab_parameters": []}
        db = self._make_db(document, None)
        fv = await build_feature_vector(db, str(user_oid), str(doc_oid))
        assert fv.feature_names == FEATURE_NAMES

    async def test_lab_values_reflected_in_features(self):
        user_oid = ObjectId()
        doc_oid = ObjectId()
        document = {
            "_id": doc_oid,
            "user_id": user_oid,
            "lab_parameters": [{"name": "glucose", "value": 7.2}],
        }
        db = self._make_db(document, None)
        fv = await build_feature_vector(db, str(user_oid), str(doc_oid))
        assert fv.features[FEATURE_NAMES.index("glucose")] == 7.2

    async def test_lifestyle_profile_used_when_present(self):
        user_oid = ObjectId()
        doc_oid = ObjectId()
        document = {"_id": doc_oid, "user_id": user_oid, "lab_parameters": []}
        lifestyle = {
            "user_id": user_oid,
            "bmi": 27.5,
            "smoking_status": "former",
            "exercise_frequency": 5,
            "sleep_hours": 8.0,
            "stress_level": 3,
        }
        db = self._make_db(document, lifestyle)
        fv = await build_feature_vector(db, str(user_oid), str(doc_oid))
        assert fv.features[FEATURE_NAMES.index("bmi")] == 27.5
        assert fv.features[FEATURE_NAMES.index("smoking_encoded")] == 0.5
        assert fv.features[FEATURE_NAMES.index("exercise_frequency")] == 5.0

    async def test_missing_document_raises_value_error(self):
        user_oid = ObjectId()
        doc_oid = ObjectId()
        db = self._make_db(None, None)
        with pytest.raises(ValueError, match="not found"):
            await build_feature_vector(db, str(user_oid), str(doc_oid))

    async def test_persists_to_feature_vectors_collection(self):
        user_oid = ObjectId()
        doc_oid = ObjectId()
        document = {"_id": doc_oid, "user_id": user_oid, "lab_parameters": []}
        db = self._make_db(document, None)
        await build_feature_vector(db, str(user_oid), str(doc_oid))
        # Verify insert_one was called on the feature_vectors collection
        db["feature_vectors"].insert_one.assert_called_once()

    async def test_imputation_for_all_missing_features(self):
        """When no lab params and no lifestyle, all features use population means."""
        user_oid = ObjectId()
        doc_oid = ObjectId()
        document = {"_id": doc_oid, "user_id": user_oid, "lab_parameters": []}
        db = self._make_db(document, None)
        fv = await build_feature_vector(db, str(user_oid), str(doc_oid))
        for i, name in enumerate(FEATURE_NAMES):
            assert fv.features[i] == POPULATION_MEANS[name], (
                f"Feature '{name}' should be imputed to {POPULATION_MEANS[name]}"
            )


# ---------------------------------------------------------------------------
# Property-Based Tests: Feature Mapper (disease-specific vectors)
# ---------------------------------------------------------------------------
# Validates: Requirements 1.7

from hypothesis import given, settings
from hypothesis import strategies as st
from backend.services.feature_mapper import FeatureMapper, DISEASE_FEATURE_SUBSETS

# Reasonable default medians for all features used across diseases
_DEFAULT_MEDIANS: dict[str, float] = {
    "glucose": 99.0,
    "hba1c": 5.7,
    "bmi": 27.5,
    "age": 50.0,
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "creatinine": 0.9,
    "hemoglobin": 13.5,
    "cholesterol": 200.0,
    "smoking_encoded": 0.2,
}

# Strategy: generate a dict with a subset of known lab/lifestyle feature keys
_ALL_FEATURE_KEYS = list(_DEFAULT_MEDIANS.keys())

_lab_values_strategy = st.dictionaries(
    keys=st.sampled_from(_ALL_FEATURE_KEYS),
    values=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=len(_ALL_FEATURE_KEYS),
)

_lifestyle_strategy = st.dictionaries(
    keys=st.sampled_from(_ALL_FEATURE_KEYS),
    values=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=len(_ALL_FEATURE_KEYS),
)


class TestFeatureMapperDimensionalityProperty:
    """Property 1: Disease Feature Vector Dimensionality.

    **Validates: Requirements 1.7**

    For all valid combinations of lab values and lifestyle profiles,
    the FeatureMapper SHALL produce a vector of exactly the correct
    length per disease:
      - diabetes: exactly 6 elements
      - cvd:      exactly 6 elements
      - ckd:      exactly 5 elements
    """

    @given(lab_values=_lab_values_strategy, lifestyle=_lifestyle_strategy)
    @settings(max_examples=200)
    def test_diabetes_vector_has_exactly_6_elements(
        self, lab_values: dict, lifestyle: dict
    ):
        """Diabetes feature vector is always exactly 6 elements."""
        mapper = FeatureMapper(medians=_DEFAULT_MEDIANS)
        result = mapper.map_features(lab_values, lifestyle, disease="diabetes")
        assert len(result.features) == 6, (
            f"Expected 6 features for diabetes, got {len(result.features)}"
        )

    @given(lab_values=_lab_values_strategy, lifestyle=_lifestyle_strategy)
    @settings(max_examples=200)
    def test_cvd_vector_has_exactly_6_elements(
        self, lab_values: dict, lifestyle: dict
    ):
        """CVD feature vector is always exactly 6 elements."""
        mapper = FeatureMapper(medians=_DEFAULT_MEDIANS)
        result = mapper.map_features(lab_values, lifestyle, disease="cvd")
        assert len(result.features) == 6, (
            f"Expected 6 features for cvd, got {len(result.features)}"
        )

    @given(lab_values=_lab_values_strategy, lifestyle=_lifestyle_strategy)
    @settings(max_examples=200)
    def test_ckd_vector_has_exactly_5_elements(
        self, lab_values: dict, lifestyle: dict
    ):
        """CKD feature vector is always exactly 5 elements."""
        mapper = FeatureMapper(medians=_DEFAULT_MEDIANS)
        result = mapper.map_features(lab_values, lifestyle, disease="ckd")
        assert len(result.features) == 5, (
            f"Expected 5 features for ckd, got {len(result.features)}"
        )

    @given(lab_values=_lab_values_strategy, lifestyle=_lifestyle_strategy)
    @settings(max_examples=100)
    def test_all_diseases_correct_length_simultaneously(
        self, lab_values: dict, lifestyle: dict
    ):
        """All three diseases produce correct-length vectors from the same inputs."""
        mapper = FeatureMapper(medians=_DEFAULT_MEDIANS)
        expected = {"diabetes": 6, "cvd": 6, "ckd": 5}
        for disease, expected_len in expected.items():
            result = mapper.map_features(lab_values, lifestyle, disease=disease)
            assert len(result.features) == expected_len, (
                f"Expected {expected_len} features for {disease}, "
                f"got {len(result.features)}"
            )

    def test_all_features_present_correct_length(self):
        """Edge case: all features provided — vector length still correct."""
        mapper = FeatureMapper(medians=_DEFAULT_MEDIANS)
        full_lab = dict(_DEFAULT_MEDIANS)
        for disease, expected_len in [("diabetes", 6), ("cvd", 6), ("ckd", 5)]:
            result = mapper.map_features(full_lab, {}, disease=disease)
            assert len(result.features) == expected_len

    def test_all_features_missing_correct_length(self):
        """Edge case: no features provided — imputation still yields correct length."""
        mapper = FeatureMapper(medians=_DEFAULT_MEDIANS)
        for disease, expected_len in [("diabetes", 6), ("cvd", 6), ("ckd", 5)]:
            result = mapper.map_features({}, {}, disease=disease)
            assert len(result.features) == expected_len

    def test_partial_features_correct_length(self):
        """Edge case: only some features provided — vector length still correct."""
        mapper = FeatureMapper(medians=_DEFAULT_MEDIANS)
        partial_lab = {"glucose": 5.5, "age": 45.0}
        partial_lifestyle = {"bmi": 24.0}
        for disease, expected_len in [("diabetes", 6), ("cvd", 6), ("ckd", 5)]:
            result = mapper.map_features(partial_lab, partial_lifestyle, disease=disease)
            assert len(result.features) == expected_len
