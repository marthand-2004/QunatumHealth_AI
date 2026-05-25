"""Hybrid Fusion service — AUC-weighted combination of RF, XGBoost, and VQC predictions.

Loads per-disease fusion weights from ``training/models/fusion_weights_v2.json``
at startup. Applies AUC-weighted fusion to produce a Risk_Score in [0, 100].

When the VQC model is unavailable (quantum_prob is None), the VQC weight is
redistributed proportionally to the remaining classical models (RF and XGBoost).

Falls back to equal weights (1/3 each, or 1/2 each if VQC unavailable) when
the weights file is missing or malformed.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISEASE_NAMES = ("diabetes", "cvd", "ckd")

_REPO_ROOT = Path(__file__).parent.parent.parent
_FUSION_WEIGHTS_PATH = _REPO_ROOT / "training" / "models" / "fusion_weights_v2.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------


def _load_fusion_weights(path: Path) -> dict[str, dict[str, float]]:
    """Load per-disease fusion weights from the JSON file.

    The JSON format produced by ``04_hybrid_fusion_v2.py`` is:
    ::

        {
          "diabetes": {"weights": {"rf": 0.35, "xgb": 0.40, "vqc": 0.25}, ...},
          "cvd":      {"weights": {"rf": ..., "xgb": ..., "vqc": ...}, ...},
          "ckd":      {"weights": {"rf": ..., "xgb": ..., "vqc": ...}, ...}
        }

    Returns a simplified dict keyed by disease name, each value being a dict
    with keys "rf", "xgb", "vqc" mapping to their respective weights.

    Falls back to equal weights (1/3 each) for any missing or malformed entry.
    """
    equal_weights: dict[str, float] = {"rf": 1.0 / 3, "xgb": 1.0 / 3, "vqc": 1.0 / 3}

    if not path.exists():
        logger.error(
            "hybrid_fusion: fusion_weights_v2.json not found at %s. "
            "Falling back to equal weights (1/3 each) for all diseases.",
            path,
        )
        return {disease: dict(equal_weights) for disease in DISEASE_NAMES}

    try:
        with open(path) as f:
            raw: dict = json.load(f)
    except Exception as exc:
        logger.error(
            "hybrid_fusion: Failed to parse fusion_weights_v2.json at %s: %s. "
            "Falling back to equal weights (1/3 each) for all diseases.",
            path,
            exc,
        )
        return {disease: dict(equal_weights) for disease in DISEASE_NAMES}

    result: dict[str, dict[str, float]] = {}
    for disease in DISEASE_NAMES:
        if disease not in raw:
            logger.warning(
                "hybrid_fusion: Disease '%s' not found in fusion_weights_v2.json. "
                "Using equal weights (1/3 each).",
                disease,
            )
            result[disease] = dict(equal_weights)
            continue

        entry = raw[disease]

        # Support both flat format {"rf": w, "xgb": w, "vqc": w}
        # and nested format {"weights": {"rf": w, "xgb": w, "vqc": w}, ...}
        if "weights" in entry and isinstance(entry["weights"], dict):
            weights_raw = entry["weights"]
        elif all(k in entry for k in ("rf", "xgb", "vqc")):
            weights_raw = entry
        else:
            logger.warning(
                "hybrid_fusion: Unexpected format for disease '%s' in "
                "fusion_weights_v2.json. Using equal weights (1/3 each).",
                disease,
            )
            result[disease] = dict(equal_weights)
            continue

        try:
            w_rf = float(weights_raw["rf"])
            w_xgb = float(weights_raw["xgb"])
            w_vqc = float(weights_raw["vqc"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "hybrid_fusion: Could not parse weights for disease '%s': %s. "
                "Using equal weights (1/3 each).",
                disease,
                exc,
            )
            result[disease] = dict(equal_weights)
            continue

        result[disease] = {"rf": w_rf, "xgb": w_xgb, "vqc": w_vqc}
        logger.info(
            "hybrid_fusion: Loaded weights for '%s': rf=%.4f, xgb=%.4f, vqc=%.4f",
            disease,
            w_rf,
            w_xgb,
            w_vqc,
        )

    return result


# ---------------------------------------------------------------------------
# HybridFusion class
# ---------------------------------------------------------------------------


class HybridFusion:
    """AUC-weighted fusion of RF, XGBoost, and VQC model probabilities.

    Parameters
    ----------
    fusion_weights:
        Per-disease weight dict, keyed by disease name. Each value is a dict
        with keys "rf", "xgb", "vqc" mapping to their respective AUC-derived
        weights. Typically loaded from ``training/models/fusion_weights_v2.json``.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """

    def __init__(self, fusion_weights: dict[str, dict[str, float]]) -> None:
        self._weights = fusion_weights

    def fuse(
        self,
        classical_rf_prob: float,
        classical_xgb_prob: float,
        quantum_prob: Optional[float],
        disease: str,
    ) -> float:
        """Compute the AUC-weighted Risk_Score for a single disease.

        Formula (Req 4.2):
            risk_score = sum(weight_i * prob_i) * 100

        When ``quantum_prob`` is None (VQC unavailable, Req 4.4), the VQC
        weight is redistributed proportionally to RF and XGBoost:
            w_rf_adj  = w_rf  / (w_rf + w_xgb)
            w_xgb_adj = w_xgb / (w_rf + w_xgb)

        Falls back to equal weights if the disease is not found in the loaded
        weights dict.

        Parameters
        ----------
        classical_rf_prob:
            RF positive-class probability in [0, 1].
        classical_xgb_prob:
            XGBoost positive-class probability in [0, 1].
        quantum_prob:
            VQC positive-class probability in [0, 1], or None if unavailable.
        disease:
            One of "diabetes", "cvd", "ckd".

        Returns
        -------
        float
            Risk_Score in the closed interval [0, 100] (Req 4.5).
        """
        # Retrieve per-disease weights; fall back to equal weights if missing
        if disease in self._weights:
            w = self._weights[disease]
            w_rf = w.get("rf", 1.0 / 3)
            w_xgb = w.get("xgb", 1.0 / 3)
            w_vqc = w.get("vqc", 1.0 / 3)
        else:
            logger.warning(
                "hybrid_fusion: Disease '%s' not found in loaded weights. "
                "Using equal weights (1/3 each).",
                disease,
            )
            w_rf = w_xgb = w_vqc = 1.0 / 3

        if quantum_prob is None:
            # VQC unavailable — redistribute VQC weight proportionally (Req 4.4)
            classical_total = w_rf + w_xgb
            if classical_total <= 0.0:
                # Degenerate case: both classical weights are zero → equal split
                w_rf_adj = 0.5
                w_xgb_adj = 0.5
            else:
                w_rf_adj = w_rf / classical_total
                w_xgb_adj = w_xgb / classical_total

            risk_score = (w_rf_adj * classical_rf_prob + w_xgb_adj * classical_xgb_prob) * 100.0
        else:
            # All three models available (Req 4.2)
            risk_score = (
                w_rf * classical_rf_prob
                + w_xgb * classical_xgb_prob
                + w_vqc * quantum_prob
            ) * 100.0

        # Clamp to [0, 100] (Req 4.5)
        return float(max(0.0, min(100.0, risk_score)))


# ---------------------------------------------------------------------------
# Module-level singleton (loaded once at import time)
# ---------------------------------------------------------------------------

_fusion_weights: dict[str, dict[str, float]] = _load_fusion_weights(_FUSION_WEIGHTS_PATH)

hybrid_fusion = HybridFusion(fusion_weights=_fusion_weights)
