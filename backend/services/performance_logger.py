"""Performance Logger — per-stage inference latency tracking and persistence.

Records wall-clock duration of each inference pipeline stage in milliseconds,
builds a ``LatencyBreakdown`` model, and appends sanitized records to
``data/logs/latency.jsonl`` for offline analysis.

Usage
-----
::

    logger_inst = PerformanceLogger()

    with logger_inst.stage("feature_extraction"):
        features = feature_mapper.map_features(...)

    with logger_inst.stage("classical_ml"):
        rf_prob, xgb_prob = predict_classical_v2(...)

    breakdown = logger_inst.get_latency_breakdown()
    await logger_inst.persist({"disease": "diabetes", "latency": breakdown.model_dump()})

Requirements: 9.2, 20.3, 21.1, 21.2, 21.4
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from backend.models.prediction_v2 import LatencyBreakdown

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_LOG_PATH = _REPO_ROOT / "data" / "logs" / "latency.jsonl"

# Stage name → LatencyBreakdown field name mapping
_STAGE_FIELD_MAP: dict[str, str] = {
    "feature_extraction":   "feature_extraction_ms",
    "scaler_normalization": "scaler_normalization_ms",
    "classical_ml":         "classical_ml_ms",
    "vqc":                  "vqc_ms",
    "hybrid_fusion":        "hybrid_fusion_ms",
    "clinical_rules":       "clinical_rules_ms",
    "xai_layer":            "xai_layer_ms",
    "ocr":                  "ocr_ms",
}

# Regex patterns for PII / sensitive data (mirrors data_governance.py)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_FILE_HASH_PATTERN = re.compile(r"\b[0-9a-fA-F]{64}\b")

# Lab feature names whose numeric values must be redacted from logs (Req 20.3)
_SENSITIVE_LAB_FIELDS = frozenset({
    "glucose", "creatinine", "systolic_bp", "diastolic_bp",
    "hemoglobin", "cholesterol", "hba1c", "bmi",
})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log sanitization helpers
# ---------------------------------------------------------------------------

def _sanitize_string(text: str) -> str:
    """Replace emails and 64-char hex hashes with ``[REDACTED]``.

    Parameters
    ----------
    text:
        Raw string that may contain PII or sensitive identifiers.

    Returns
    -------
    str
        Sanitized string safe for log output.
    """
    text = _EMAIL_PATTERN.sub("[REDACTED]", text)
    text = _FILE_HASH_PATTERN.sub("[REDACTED]", text)
    return text


def _sanitize_record(record: Any) -> Any:
    """Recursively sanitize a log record dict/list/scalar.

    - Strings: redact emails and file hashes.
    - Dicts: redact values for known sensitive lab field keys; recurse into
      all other values.
    - Lists/tuples: recurse into each element.
    - Numerics: pass through unchanged (numeric values are only redacted when
      the parent dict key is a known sensitive lab field).

    Parameters
    ----------
    record:
        Arbitrary JSON-serialisable value.

    Returns
    -------
    Any
        Sanitized copy of the input.
    """
    if isinstance(record, dict):
        sanitized: dict[str, Any] = {}
        for key, value in record.items():
            if key in _SENSITIVE_LAB_FIELDS:
                # Redact raw lab values regardless of type (Req 20.3)
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str):
                sanitized[key] = _sanitize_string(value)
            else:
                sanitized[key] = _sanitize_record(value)
        return sanitized
    elif isinstance(record, (list, tuple)):
        return [_sanitize_record(item) for item in record]
    elif isinstance(record, str):
        return _sanitize_string(record)
    else:
        # int, float, bool, None — safe to pass through
        return record


# ---------------------------------------------------------------------------
# PerformanceLogger
# ---------------------------------------------------------------------------

class PerformanceLogger:
    """Records per-stage wall-clock latency for the v2 inference pipeline.

    Each call to :meth:`stage` returns a context manager that measures the
    elapsed time of the enclosed block and stores it under the given stage
    name.  After all stages complete, call :meth:`get_latency_breakdown` to
    obtain a :class:`~backend.models.prediction_v2.LatencyBreakdown` model,
    and :meth:`persist` to append a sanitized JSONL record to disk.

    Parameters
    ----------
    log_path:
        Path to the JSONL file where latency records are appended.
        Defaults to ``data/logs/latency.jsonl``.

    Requirements: 9.2, 21.1, 21.2, 21.4
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self._log_path: Path = log_path or _DEFAULT_LOG_PATH
        self._timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Stage context manager — Req 21.1
    # ------------------------------------------------------------------

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager that records the wall-clock duration of a stage.

        The measured duration (in milliseconds) is stored under ``name`` and
        is later used by :meth:`get_latency_breakdown`.

        Parameters
        ----------
        name:
            Stage identifier.  Recognised names (mapped to
            ``LatencyBreakdown`` fields):
            ``"feature_extraction"``, ``"scaler_normalization"``,
            ``"classical_ml"``, ``"vqc"``, ``"hybrid_fusion"``,
            ``"clinical_rules"``, ``"xai_layer"``, ``"ocr"``.
            Unknown names are stored but not mapped to a named field.

        Example
        -------
        ::

            perf = PerformanceLogger()
            with perf.stage("classical_ml"):
                rf_prob, xgb_prob = predict_classical_v2(features, disease)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._timings[name] = elapsed_ms
            logger.debug(
                "PerformanceLogger: stage '%s' completed in %.2f ms",
                name,
                elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Latency breakdown — Req 21.2
    # ------------------------------------------------------------------

    def get_latency_breakdown(self) -> LatencyBreakdown:
        """Build a :class:`LatencyBreakdown` from the recorded stage timings.

        The ``total_ms`` field is computed as the sum of all non-OCR stage
        durations (``ocr_ms`` is excluded from the latency budget per
        Req 9.1 and 9.4).

        Returns
        -------
        LatencyBreakdown
            Pydantic model with per-stage durations in milliseconds.
        """
        # Map recorded timings to LatencyBreakdown field names
        field_values: dict[str, float] = {}
        for stage_name, duration_ms in self._timings.items():
            field_name = _STAGE_FIELD_MAP.get(stage_name)
            if field_name:
                field_values[field_name] = duration_ms
            else:
                logger.debug(
                    "PerformanceLogger: unknown stage '%s' (%.2f ms) — "
                    "not mapped to a LatencyBreakdown field.",
                    stage_name,
                    duration_ms,
                )

        # total_ms = sum of all non-OCR stages (Req 9.1)
        non_ocr_stages = {
            k: v for k, v in self._timings.items() if k != "ocr"
        }
        total_ms = sum(non_ocr_stages.values())

        return LatencyBreakdown(
            ocr_ms=field_values.get("ocr_ms"),
            feature_extraction_ms=field_values.get("feature_extraction_ms", 0.0),
            scaler_normalization_ms=field_values.get("scaler_normalization_ms", 0.0),
            classical_ml_ms=field_values.get("classical_ml_ms", 0.0),
            vqc_ms=field_values.get("vqc_ms", 0.0),
            hybrid_fusion_ms=field_values.get("hybrid_fusion_ms", 0.0),
            clinical_rules_ms=field_values.get("clinical_rules_ms", 0.0),
            xai_layer_ms=field_values.get("xai_layer_ms", 0.0),
            total_ms=total_ms,
        )

    # ------------------------------------------------------------------
    # Persistence — Req 21.4
    # ------------------------------------------------------------------

    async def persist(self, record: dict[str, Any]) -> None:
        """Append a sanitized latency record to the JSONL log file.

        The record is sanitized before writing to remove patient emails,
        file hashes, and raw lab values (Req 20.3).  The log directory is
        created automatically if it does not exist.

        The write is performed in a thread-pool executor to avoid blocking
        the event loop.

        Parameters
        ----------
        record:
            Arbitrary JSON-serialisable dict to append.  Typically contains
            the disease name, timestamp, and the ``LatencyBreakdown`` dict.

        Example
        -------
        ::

            breakdown = perf.get_latency_breakdown()
            await perf.persist({
                "disease": "diabetes",
                "latency": breakdown.model_dump(),
            })
        """
        sanitized = _sanitize_record(record)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_record, sanitized)

    def _write_record(self, record: dict[str, Any]) -> None:
        """Synchronous helper: create log directory and append JSONL line.

        Separated from :meth:`persist` so it can be run in a thread-pool
        executor without blocking the async event loop.

        Parameters
        ----------
        record:
            Pre-sanitized dict to serialise as a single JSONL line.
        """
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, default=str) + "\n"
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
            logger.debug(
                "PerformanceLogger: appended record to %s",
                self._log_path,
            )
        except OSError as exc:
            logger.error(
                "PerformanceLogger: failed to write latency record to %s: %s",
                self._log_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded stage timings.

        Useful when reusing a single ``PerformanceLogger`` instance across
        multiple requests (e.g., in tests).
        """
        self._timings.clear()

    @property
    def timings(self) -> dict[str, float]:
        """Read-only view of the recorded stage timings (ms).

        Returns
        -------
        dict[str, float]
            Mapping of stage name → elapsed milliseconds.
        """
        return dict(self._timings)
