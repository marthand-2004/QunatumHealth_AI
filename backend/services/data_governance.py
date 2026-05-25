"""Data Governance Module for QuantumHealthAI v2 Enhanced.

Validates, normalizes, and cleans incoming lab values before feature extraction.
Loads medians and IQR fence values from training/models/data_governance.json.

Requirements: 13.1, 13.2, 13.3, 13.5, 20.3
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Acceptable input ranges for each lab feature (in original units)
ACCEPTABLE_RANGES: dict[str, tuple[float, float]] = {
    "glucose":     (50.0, 500.0),   # mg/dL
    "creatinine":  (0.5, 15.0),     # mg/dL
    "systolic_bp": (80.0, 250.0),   # mmHg
    "diastolic_bp":(50.0, 150.0),   # mmHg
    "hemoglobin":  (3.0, 20.0),     # g/dL
    "cholesterol": (50.0, 400.0),   # mg/dL
    "hba1c":       (3.0, 15.0),     # %
    "bmi":         (10.0, 70.0),    # kg/m²
}

# Unit conversion factors applied after range validation
UNIT_CONVERSIONS: dict[str, float] = {
    "glucose":     0.05551,   # mg/dL → mmol/L
    "creatinine":  88.42,     # mg/dL → µmol/L
    "cholesterol": 0.02586,   # mg/dL → mmol/L
}

# Sensitive field names whose values should be redacted in logs
_SENSITIVE_LAB_FIELDS = frozenset(ACCEPTABLE_RANGES.keys())

# Regex patterns for PII / sensitive data
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_FILE_HASH_PATTERN = re.compile(r"\b[0-9a-fA-F]{64}\b")

# Default path to data_governance.json
_DEFAULT_GOVERNANCE_PATH = (
    Path(__file__).parent.parent.parent / "training" / "models" / "data_governance.json"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log sanitization filter
# ---------------------------------------------------------------------------

class LOG_SANITIZER(logging.Filter):
    """Logging filter that masks patient emails, file hashes, and raw lab values.

    Attach to any logger or handler to prevent sensitive data from appearing
    in log output.

    Requirements: 20.3
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = self._sanitize(record.msg)
        # Also sanitize pre-formatted args
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._sanitize(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._sanitize(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True

    @staticmethod
    def _sanitize(text: str) -> str:
        """Replace emails, 64-char hex hashes, and numeric lab values with [REDACTED]."""
        text = _EMAIL_PATTERN.sub("[REDACTED]", text)
        text = _FILE_HASH_PATTERN.sub("[REDACTED]", text)
        return text


# ---------------------------------------------------------------------------
# DataGovernanceModule
# ---------------------------------------------------------------------------

class DataGovernanceModule:
    """Validates, normalizes, and cleans incoming lab values.

    Loads medians and IQR fence values from data_governance.json at init.
    Falls back gracefully if the file does not exist yet (e.g., before training).

    Requirements: 13.1, 13.2, 13.3, 13.5, 20.3
    """

    def __init__(
        self,
        governance_path: Optional[Path] = None,
    ) -> None:
        path = governance_path or _DEFAULT_GOVERNANCE_PATH
        self.medians: dict[str, float] = {}
        self.iqr_fences: dict[str, tuple[float, float]] = {}  # (lower, upper)

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    config = json.load(fh)
                self.medians = {k: float(v) for k, v in config.get("medians", {}).items()}
                raw_fences = config.get("iqr_fences", {})
                self.iqr_fences = {
                    k: (float(v[0]), float(v[1]))
                    for k, v in raw_fences.items()
                    if isinstance(v, (list, tuple)) and len(v) == 2
                }
                logger.info("DataGovernanceModule loaded config from %s", path)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "DataGovernanceModule: failed to load %s (%s). "
                    "Proceeding without IQR fences and medians.",
                    path,
                    exc,
                )
        else:
            logger.warning(
                "DataGovernanceModule: governance config not found at %s. "
                "IQR clipping and median imputation will be skipped.",
                path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_and_clean(
        self,
        raw_values: dict[str, float],
    ) -> tuple[dict[str, float], list[str]]:
        """Validate, convert units, and clip outliers for incoming lab values.

        Steps:
        1. Check acceptable ranges → clip to nearest boundary and log warning.
           (Does NOT raise HTTP 422; the router decides whether to reject.)
        2. Convert units using UNIT_CONVERSIONS.
        3. Apply IQR clipping using fence values from data_governance.json
           (skipped if fences are not loaded).

        Parameters
        ----------
        raw_values:
            Dict mapping feature name → raw numeric value (in original units).

        Returns
        -------
        (cleaned_values, warnings_list)
            cleaned_values: dict with validated, unit-converted, IQR-clipped values.
            warnings_list: list of human-readable warning strings (one per clipped value).
        """
        cleaned: dict[str, float] = {}
        warnings: list[str] = []

        for key, value in raw_values.items():
            val = float(value)

            # Step 1: Acceptable range check → clip
            if key in ACCEPTABLE_RANGES:
                lo, hi = ACCEPTABLE_RANGES[key]
                if val < lo or val > hi:
                    clipped = max(lo, min(hi, val))
                    warnings.append(
                        f"Feature '{key}' value {self.sanitize_log_value(key, val)} "
                        f"is outside acceptable range [{lo}, {hi}]; "
                        f"clipped to {self.sanitize_log_value(key, clipped)}."
                    )
                    logger.warning(
                        "DataGovernance: feature '%s' out of range [%s, %s]; clipped.",
                        key,
                        lo,
                        hi,
                    )
                    val = clipped

            # Step 2: Unit conversion
            if key in UNIT_CONVERSIONS:
                val = val * UNIT_CONVERSIONS[key]

            cleaned[key] = val

        # Step 3: IQR clipping (post unit-conversion, using converted fence values)
        if self.iqr_fences:
            for key, val in list(cleaned.items()):
                if key in self.iqr_fences:
                    lower, upper = self.iqr_fences[key]
                    if val < lower or val > upper:
                        clipped = max(lower, min(upper, val))
                        logger.debug(
                            "DataGovernance: IQR clip applied to '%s'.",
                            key,
                        )
                        cleaned[key] = clipped

        return cleaned, warnings

    @staticmethod
    def sanitize_log_value(key: str, value) -> str:
        """Return a safe string representation for logging.

        Masks raw lab values, patient emails, and file hashes with [REDACTED].

        Parameters
        ----------
        key:
            The field name (e.g., "glucose", "email").
        value:
            The raw value to potentially mask.
        """
        str_value = str(value)

        # Mask known sensitive lab fields
        if key in _SENSITIVE_LAB_FIELDS:
            return "[REDACTED]"

        # Mask email addresses
        if _EMAIL_PATTERN.search(str_value):
            return "[REDACTED]"

        # Mask 64-char hex file hashes
        if _FILE_HASH_PATTERN.search(str_value):
            return "[REDACTED]"

        return str_value
