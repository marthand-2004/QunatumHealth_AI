"""Research and admin endpoints for QuantumHealthAI v2 Enhanced.

Provides:
  - GET /api/research/quantum-justification   (Admin only)
  - GET /api/admin/latency-stats              (Admin only)
  - GET /api/research/evaluation-plots/{disease}/{model_type}  (Admin/Doctor)

Requirements: 6.6, 17.5, 21.5
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from backend.core.deps import require_role

logger = logging.getLogger(__name__)

router = APIRouter()
admin_router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_TRAINING_RESULTS_DIR = _REPO_ROOT / "training" / "results"
_LATENCY_LOG_PATH = _REPO_ROOT / "data" / "logs" / "latency.jsonl"

# Maximum number of latency records to consider for stats (Req 21.5)
_LATENCY_WINDOW = 1000

# Valid diseases and model types for evaluation plot serving (Req 6.6)
_VALID_DISEASES = frozenset({"diabetes", "cvd", "ckd"})
_VALID_MODEL_TYPES = frozenset({"rf", "xgb", "vqc"})

# Plot subdirectory mapping: model_type → subdirectory name
_PLOT_SUBDIRS = {
    "confusion_matrix": "confusion_matrices",
    "roc_curve": "roc_curves",
    "calibration_curve": "calibration_curves",
    "shap": "shap_plots",
    "training_curve": "training_curves",
}

# Role dependencies
_admin_only = require_role(["admin"])
_admin_or_doctor = require_role(["admin", "doctor"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json_file(path: Path, label: str) -> dict:
    """Load a JSON file and return its contents as a dict.

    Raises HTTP 404 if the file does not exist, HTTP 500 on parse error.
    """
    if not path.exists():
        logger.warning("research: %s not found at %s", label, path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found. Run the training pipeline to generate it.",
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("research: Failed to parse %s at %s: %s", label, path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse {label}: {exc}",
        )


def _read_last_n_latency_records(n: int) -> list[dict]:
    """Read the last *n* records from the JSONL latency log.

    Returns an empty list if the file does not exist or is empty.
    """
    if not _LATENCY_LOG_PATH.exists():
        return []

    records: list[dict] = []
    try:
        with open(_LATENCY_LOG_PATH, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        # Take the last n lines
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("research: Skipping malformed JSONL line in latency log.")
    except OSError as exc:
        logger.error("research: Failed to read latency log at %s: %s", _LATENCY_LOG_PATH, exc)

    return records


def _compute_percentile(sorted_values: list[float], pct: float) -> float:
    """Compute the *pct*-th percentile of a pre-sorted list using linear interpolation.

    Parameters
    ----------
    sorted_values:
        Ascending-sorted list of numeric values.
    pct:
        Percentile in [0, 100].

    Returns
    -------
    float
        Interpolated percentile value, or 0.0 for an empty list.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    # Linear interpolation (same as numpy.percentile default)
    index = (pct / 100.0) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_values[-1]
    frac = index - lower
    return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])


def _compute_stage_stats(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean, p50, p95, p99 latency per stage from a list of latency records.

    Each record is expected to have a ``latency`` sub-dict with stage keys
    matching the ``LatencyBreakdown`` field names (e.g. ``classical_ml_ms``).

    Parameters
    ----------
    records:
        List of JSONL records from ``data/logs/latency.jsonl``.

    Returns
    -------
    dict
        Mapping of stage_name → {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "count"}.
    """
    # Collect per-stage values
    stage_values: dict[str, list[float]] = {}

    for record in records:
        latency = record.get("latency", {})
        if not isinstance(latency, dict):
            continue
        for key, value in latency.items():
            if not key.endswith("_ms"):
                continue
            if value is None:
                continue
            try:
                ms = float(value)
            except (TypeError, ValueError):
                continue
            stage_values.setdefault(key, []).append(ms)

    # Compute statistics per stage
    result: dict[str, dict[str, float]] = {}
    for stage, values in stage_values.items():
        if not values:
            continue
        sorted_vals = sorted(values)
        result[stage] = {
            "mean_ms": round(statistics.mean(values), 3),
            "p50_ms": round(_compute_percentile(sorted_vals, 50), 3),
            "p95_ms": round(_compute_percentile(sorted_vals, 95), 3),
            "p99_ms": round(_compute_percentile(sorted_vals, 99), 3),
            "count": len(values),
        }

    return result


# ---------------------------------------------------------------------------
# GET /api/research/quantum-justification  (Admin only)
# ---------------------------------------------------------------------------


@router.get(
    "/quantum-justification",
    summary="Quantum justification report (Admin only)",
    description=(
        "Returns the quantum justification JSON report comparing VQC vs classical "
        "model AUC across layer counts per disease. Requires Admin role."
    ),
    response_model=dict,
    status_code=status.HTTP_200_OK,
)
async def get_quantum_justification(
    _current_user: dict = Depends(_admin_only),
) -> dict:
    """Serve ``training/results/quantum_justification.json``.

    Requirements: 17.5
    """
    path = _TRAINING_RESULTS_DIR / "quantum_justification.json"
    data = _load_json_file(path, "quantum_justification.json")
    return data


# ---------------------------------------------------------------------------
# GET /api/admin/latency-stats  (Admin only)
# ---------------------------------------------------------------------------


@admin_router.get(
    "/latency-stats",
    summary="Inference latency statistics (Admin only)",
    description=(
        "Computes mean, p50, p95, and p99 latency per pipeline stage from the "
        f"last {_LATENCY_WINDOW} records in data/logs/latency.jsonl. Requires Admin role."
    ),
    response_model=dict,
    status_code=status.HTTP_200_OK,
)
async def get_latency_stats(
    _current_user: dict = Depends(_admin_only),
) -> dict:
    """Compute and return per-stage latency statistics.

    Requirements: 21.5
    """
    records = _read_last_n_latency_records(_LATENCY_WINDOW)

    if not records:
        return {
            "record_count": 0,
            "window": _LATENCY_WINDOW,
            "message": "No latency records found. Run predictions to populate the log.",
            "stages": {},
        }

    stage_stats = _compute_stage_stats(records)

    return {
        "record_count": len(records),
        "window": _LATENCY_WINDOW,
        "stages": stage_stats,
    }


# ---------------------------------------------------------------------------
# GET /api/research/evaluation-plots/{disease}/{model_type}  (Admin/Doctor)
# ---------------------------------------------------------------------------

# Supported plot types and their file naming conventions
_PLOT_TYPE_CONFIG: dict[str, dict] = {
    "confusion_matrix": {
        "subdir": "confusion_matrices",
        "filename_template": "confusion_matrix_{disease}_{model_type}.png",
    },
    "roc_curve": {
        "subdir": "roc_curves",
        "filename_template": "roc_curve_{disease}_{model_type}.png",
    },
    "calibration_curve": {
        "subdir": "calibration_curves",
        "filename_template": "calibration_curve_{disease}_{model_type}.png",
    },
    "shap": {
        "subdir": "shap_plots",
        "filename_template": "shap_summary_{disease}.png",
    },
    "training_curve": {
        "subdir": "training_curves",
        "filename_template": "vqc_{disease}_loss.png",
    },
}


@router.get(
    "/evaluation-plots/{disease}/{model_type}",
    summary="Serve evaluation plot PNG (Admin/Doctor)",
    description=(
        "Serves a PNG evaluation plot for the given disease and model type. "
        "Supported model types: rf, xgb, vqc. "
        "Use the 'plot_type' query parameter to select the plot category "
        "(confusion_matrix, roc_curve, calibration_curve, shap, training_curve). "
        "Requires Admin or Doctor role."
    ),
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
)
async def get_evaluation_plot(
    disease: str,
    model_type: str,
    plot_type: str = "roc_curve",
    _current_user: dict = Depends(_admin_or_doctor),
) -> FileResponse:
    """Serve a PNG evaluation plot from ``training/results/``.

    Requirements: 6.6
    """
    # Validate disease
    if disease not in _VALID_DISEASES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid disease '{disease}'. Must be one of: {sorted(_VALID_DISEASES)}.",
        )

    # Validate model_type
    if model_type not in _VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid model_type '{model_type}'. Must be one of: {sorted(_VALID_MODEL_TYPES)}.",
        )

    # Validate plot_type
    if plot_type not in _PLOT_TYPE_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid plot_type '{plot_type}'. "
                f"Must be one of: {sorted(_PLOT_TYPE_CONFIG.keys())}."
            ),
        )

    config = _PLOT_TYPE_CONFIG[plot_type]
    subdir: str = config["subdir"]
    filename: str = config["filename_template"].format(
        disease=disease,
        model_type=model_type,
    )

    plot_path = _TRAINING_RESULTS_DIR / subdir / filename

    if not plot_path.exists():
        logger.warning(
            "research: Evaluation plot not found: %s (disease=%s, model_type=%s, plot_type=%s)",
            plot_path,
            disease,
            model_type,
            plot_type,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Plot '{plot_type}' for disease='{disease}', model_type='{model_type}' "
                f"not found at '{plot_path.relative_to(_REPO_ROOT)}'. "
                "Run the training pipeline to generate evaluation plots."
            ),
        )

    return FileResponse(
        path=str(plot_path),
        media_type="image/png",
        filename=filename,
    )
