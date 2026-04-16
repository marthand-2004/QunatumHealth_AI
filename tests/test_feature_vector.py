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
