"""Clinical Rule Engine for QuantumHealthAI v2 Enhanced.

Applies hard clinical threshold rules on top of model-predicted risk scores.
Each triggered rule adds 10 percentage points to the relevant disease risk
score, capped at 100.

Rules defined:
  - Diabetes: fasting glucose > 126 mg/dL
  - CKD:      creatinine > 1.3 mg/dL
  - CVD:      systolic BP > 140 mmHg OR diastolic BP > 90 mmHg

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# Each rule is a plain dict so it can be inspected, serialised, and extended
# without subclassing.  The ``condition`` callable receives the raw lab values
# dict (in original units, i.e. *before* unit conversion) and returns a bool.

CLINICAL_RULES: list[dict] = [
    {
        "disease": "diabetes",
        "condition": lambda vals: float(vals.get("glucose", 0)) > 126,
        "adjustment": 10,
        "description": "Fasting glucose > 126 mg/dL",
    },
    {
        "disease": "ckd",
        "condition": lambda vals: float(vals.get("creatinine", 0)) > 1.3,
        "adjustment": 10,
        "description": "Creatinine > 1.3 mg/dL",
    },
    {
        "disease": "cvd",
        "condition": (
            lambda vals: (
                float(vals.get("systolic_bp", 0)) > 140
                or float(vals.get("diastolic_bp", 0)) > 90
            )
        ),
        "adjustment": 10,
        "description": "Blood pressure > 140/90 mmHg",
    },
]


# ---------------------------------------------------------------------------
# apply_rules
# ---------------------------------------------------------------------------


def apply_rules(
    risk_scores: dict[str, float],
    raw_lab_values: dict[str, float],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Apply all clinical threshold rules to the model-predicted risk scores.

    Each rule is evaluated independently (Req 5.4).  When a rule's condition
    is met, its ``adjustment`` (10 percentage points) is added to the
    corresponding disease risk score and the rule description is appended to
    the triggered-rules list for that disease.  After all rules have been
    applied the score is capped at 100 (Req 5.5, 5.7).

    Parameters
    ----------
    risk_scores:
        Dict mapping disease name → Risk_Score in [0, 100] as produced by
        ``HybridFusion.fuse()``.  Only diseases present in this dict are
        processed; unknown diseases are passed through unchanged.
    raw_lab_values:
        Dict mapping feature name → raw numeric value in **original units**
        (mg/dL, mmHg, etc.).  Unit conversion must NOT have been applied yet
        so that the hard-coded clinical thresholds (126 mg/dL, 1.3 mg/dL,
        140/90 mmHg) are evaluated correctly.

    Returns
    -------
    adjusted_scores : dict[str, float]
        A new dict with the same keys as ``risk_scores`` but with rule
        adjustments applied and each value clamped to [0, 100].
    triggered_rules_per_disease : dict[str, list[str]]
        A dict mapping each disease name to a (possibly empty) list of
        human-readable rule description strings that were triggered.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
    """
    # Start with a copy so the caller's dict is not mutated
    adjusted_scores: dict[str, float] = {d: float(s) for d, s in risk_scores.items()}

    # Initialise triggered-rules lists for every disease in the input
    triggered_rules_per_disease: dict[str, list[str]] = {
        disease: [] for disease in risk_scores
    }

    for rule in CLINICAL_RULES:
        disease: str = rule["disease"]
        condition: Callable[[dict], bool] = rule["condition"]
        adjustment: float = float(rule["adjustment"])
        description: str = rule["description"]

        # Only apply the rule if the disease is present in the input scores
        if disease not in adjusted_scores:
            continue

        try:
            triggered = condition(raw_lab_values)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "clinical_rule_engine: Error evaluating rule '%s' for disease '%s': %s. "
                "Rule skipped.",
                description,
                disease,
                exc,
            )
            continue

        if triggered:
            adjusted_scores[disease] += adjustment
            triggered_rules_per_disease[disease].append(description)
            logger.info(
                "clinical_rule_engine: Rule triggered for '%s' — '%s'. "
                "Score adjusted by +%s.",
                disease,
                description,
                adjustment,
            )

    # Cap each score at 100 (Req 5.5, 5.7)
    for disease in adjusted_scores:
        if adjusted_scores[disease] > 100.0:
            logger.debug(
                "clinical_rule_engine: Risk score for '%s' capped at 100 "
                "(was %.2f after rule application).",
                disease,
                adjusted_scores[disease],
            )
            adjusted_scores[disease] = 100.0

        # Ensure lower bound as well (defensive)
        if adjusted_scores[disease] < 0.0:
            adjusted_scores[disease] = 0.0

    return adjusted_scores, triggered_rules_per_disease
