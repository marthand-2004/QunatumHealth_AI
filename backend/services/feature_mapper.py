"""Feature Mapper for QuantumHealthAI v2 Enhanced.

Maps OCR-extracted lab values and lifestyle data into disease-specific
feature vectors. Replaces the unified 14-dimensional vector with
disease-specific subsets.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Disease-specific feature subsets (ordered)
# ---------------------------------------------------------------------------

DISEASE_FEATURE_SUBSETS: dict[str, list[str]] = {
    "diabetes": ["glucose", "hba1c", "bmi", "age", "systolic_bp", "diastolic_bp"],  # 6 features
    "cvd":      ["age", "cholesterol", "systolic_bp", "diastolic_bp", "smoking_encoded", "bmi"],  # 6 features
    "ckd":      ["creatinine", "hemoglobin", "systolic_bp", "diastolic_bp", "age"],  # 5 features
}

_VALID_DISEASES = frozenset(DISEASE_FEATURE_SUBSETS.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureMapperResult:
    """Result of a disease-specific feature mapping operation.

    Attributes
    ----------
    features:
        Ordered list of float values matching the disease-specific subset.
    missing_flags:
        Dict mapping feature name → True if the value was imputed from median.
    disease:
        The disease identifier ("diabetes", "cvd", or "ckd").
    """
    features: list[float] = field(default_factory=list)
    missing_flags: dict[str, bool] = field(default_factory=dict)
    disease: str = ""


# ---------------------------------------------------------------------------
# FeatureMapper
# ---------------------------------------------------------------------------

class FeatureMapper:
    """Maps lab values and lifestyle data to disease-specific feature vectors.

    Missing features are imputed using the training-set median loaded from
    ``training/models/data_governance.json``.

    Parameters
    ----------
    medians:
        Pre-computed training-set medians keyed by feature name.
        Typically loaded from ``DataGovernanceModule.medians``.

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """

    def __init__(self, medians: dict[str, float]) -> None:
        self.medians = medians

    def map_features(
        self,
        lab_values: dict[str, float],
        lifestyle: dict,
        disease: str,
    ) -> FeatureMapperResult:
        """Extract a disease-specific ordered feature vector.

        For each feature in ``DISEASE_FEATURE_SUBSETS[disease]``:
        1. Look up in ``lab_values``.
        2. If not found, look up in ``lifestyle``.
        3. If still not found, impute with ``self.medians[feature]``
           (or 0.0 if the median is also missing) and set
           ``missing_flags[feature] = True``.

        Parameters
        ----------
        lab_values:
            Dict of lab feature name → numeric value (already unit-converted
            and cleaned by DataGovernanceModule).
        lifestyle:
            Dict of lifestyle feature name → value (e.g., bmi, smoking_encoded,
            exercise_frequency, sleep_hours, stress_level, age).
        disease:
            One of "diabetes", "cvd", "ckd".

        Returns
        -------
        FeatureMapperResult
            Contains the ordered feature list, missing_flags dict, and disease.

        Raises
        ------
        ValueError
            If ``disease`` is not one of the supported disease identifiers.
        """
        if disease not in _VALID_DISEASES:
            raise ValueError(
                f"Unsupported disease '{disease}'. "
                f"Must be one of: {sorted(_VALID_DISEASES)}"
            )

        subset = DISEASE_FEATURE_SUBSETS[disease]
        features: list[float] = []
        missing_flags: dict[str, bool] = {}

        for feature_name in subset:
            # Priority 1: lab_values
            if feature_name in lab_values:
                features.append(float(lab_values[feature_name]))
                missing_flags[feature_name] = False

            # Priority 2: lifestyle dict
            elif feature_name in lifestyle and lifestyle[feature_name] is not None:
                features.append(float(lifestyle[feature_name]))
                missing_flags[feature_name] = False

            # Priority 3: impute from median (or 0.0 as last resort)
            else:
                imputed = float(self.medians.get(feature_name, 0.0))
                features.append(imputed)
                missing_flags[feature_name] = True

        return FeatureMapperResult(
            features=features,
            missing_flags=missing_flags,
            disease=disease,
        )
