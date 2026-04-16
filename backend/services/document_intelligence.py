"""Document Intelligence service — lab parameter parsing, unit normalization, and abnormal flagging.

Requirements: 4.1, 4.2, 4.3, 4.4
"""
from __future__ import annotations

import re
from typing import Optional

from backend.models.document import LabParameter

# ---------------------------------------------------------------------------
# Unit conversion lookup table
# Each entry: (conversion_factor, target_si_unit)
# converted_value = raw_value * conversion_factor
# ---------------------------------------------------------------------------
_UNIT_CONVERSIONS: dict[str, dict[str, tuple[float, str]]] = {
    "glucose": {
        "mg/dl": (0.05551, "mmol/L"),
        "mg/dL": (0.05551, "mmol/L"),
        "mmol/l": (1.0, "mmol/L"),
        "mmol/L": (1.0, "mmol/L"),
    },
    "hba1c": {
        "%": (1.0, "%"),
        "percent": (1.0, "%"),
        "mmol/mol": (0.1, "%"),  # approximate IFCC → NGSP
    },
    "creatinine": {
        "mg/dl": (88.42, "µmol/L"),
        "mg/dL": (88.42, "µmol/L"),
        "umol/l": (1.0, "µmol/L"),
        "µmol/l": (1.0, "µmol/L"),
        "µmol/L": (1.0, "µmol/L"),
        "umol/L": (1.0, "µmol/L"),
    },
    "total_cholesterol": {
        "mg/dl": (0.02586, "mmol/L"),
        "mg/dL": (0.02586, "mmol/L"),
        "mmol/l": (1.0, "mmol/L"),
        "mmol/L": (1.0, "mmol/L"),
    },
    "cholesterol": {
        "mg/dl": (0.02586, "mmol/L"),
        "mg/dL": (0.02586, "mmol/L"),
        "mmol/l": (1.0, "mmol/L"),
        "mmol/L": (1.0, "mmol/L"),
    },
    "triglycerides": {
        "mg/dl": (0.01129, "mmol/L"),
        "mg/dL": (0.01129, "mmol/L"),
        "mmol/l": (1.0, "mmol/L"),
        "mmol/L": (1.0, "mmol/L"),
    },
    "hdl": {
        "mg/dl": (0.02586, "mmol/L"),
        "mg/dL": (0.02586, "mmol/L"),
        "mmol/l": (1.0, "mmol/L"),
        "mmol/L": (1.0, "mmol/L"),
    },
    "ldl": {
        "mg/dl": (0.02586, "mmol/L"),
        "mg/dL": (0.02586, "mmol/L"),
        "mmol/l": (1.0, "mmol/L"),
        "mmol/L": (1.0, "mmol/L"),
    },
    "hemoglobin": {
        "g/dl": (1.0, "g/dL"),
        "g/dL": (1.0, "g/dL"),
        "mmol/l": (16.115, "g/dL"),
        "mmol/L": (16.115, "g/dL"),
    },
    "bun": {
        "mg/dl": (0.3570, "mmol/L"),
        "mg/dL": (0.3570, "mmol/L"),
        "mmol/l": (1.0, "mmol/L"),
        "mmol/L": (1.0, "mmol/L"),
    },
    # CBC — pass-through (already in correct units)
    "rbc_count":      {"million/cmm": (1.0, "million/µL"), "million/µl": (1.0, "million/µL"), "million/ul": (1.0, "million/µL"), "10^6/µl": (1.0, "million/µL")},
    "wbc_count":      {"/cmm": (1.0, "/µL"), "/µl": (1.0, "/µL"), "/ul": (1.0, "/µL"), "cells/µl": (1.0, "/µL"), "10^3/µl": (1000.0, "/µL")},
    "platelet_count": {"/cmm": (1.0, "/µL"), "/µl": (1.0, "/µL"), "10^3/µl": (1000.0, "/µL"), "lakhs/µl": (100000.0, "/µL")},
    "hematocrit":     {"%": (1.0, "%")},
    "mcv":            {"fl": (1.0, "fL"), "fL": (1.0, "fL")},
    "mch":            {"pg": (1.0, "pg")},
    "mchc":           {"g/dl": (1.0, "g/dL"), "g/dL": (1.0, "g/dL")},
    "rdw":            {"%": (1.0, "%")},
    "neutrophils":    {"%": (1.0, "%")},
    "lymphocytes":    {"%": (1.0, "%")},
    "eosinophils":    {"%": (1.0, "%")},
    "monocytes":      {"%": (1.0, "%")},
    "basophils":      {"%": (1.0, "%")},
}

# ---------------------------------------------------------------------------
# Clinical reference ranges (in SI / normalized units)
# Tuple: (low, high) — None means no lower/upper bound
# ---------------------------------------------------------------------------
_REFERENCE_RANGES: dict[str, tuple[Optional[float], Optional[float]]] = {
    "glucose":           (3.9, 6.1),      # mmol/L
    "hba1c":             (4.0, 5.6),      # %
    "creatinine":        (53.0, 106.0),   # µmol/L
    "total_cholesterol": (None, 5.2),     # mmol/L
    "cholesterol":       (None, 5.2),
    "triglycerides":     (None, 1.7),     # mmol/L
    "hdl":               (1.0, None),     # mmol/L
    "ldl":               (None, 3.4),     # mmol/L
    "hemoglobin":        (12.0, 17.0),    # g/dL
    "bun":               (2.5, 7.1),      # mmol/L
    # CBC
    "rbc_count":         (4.5, 5.5),      # million/µL
    "wbc_count":         (4000, 10000),   # /µL
    "platelet_count":    (150000, 410000),# /µL
    "hematocrit":        (40.0, 50.0),    # %
    "mcv":               (83.0, 101.0),   # fL
    "mch":               (27.0, 32.0),    # pg
    "mchc":              (31.5, 36.0),    # g/dL
    "rdw":               (11.6, 14.0),    # %
    "neutrophils":       (40.0, 80.0),    # %
    "lymphocytes":       (20.0, 40.0),    # %
    "eosinophils":       (1.0, 6.0),      # %
    "monocytes":         (2.0, 10.0),     # %
    "basophils":         (0.0, 1.0),      # %
}

# ---------------------------------------------------------------------------
# Canonical name aliases — maps OCR variants to internal canonical names
# ---------------------------------------------------------------------------
_NAME_ALIASES: dict[str, str] = {
    # glucose
    "glucose": "glucose",
    "blood glucose": "glucose",
    "fasting glucose": "glucose",
    "fasting blood glucose": "glucose",
    "fbs": "glucose",
    "rbs": "glucose",
    "random blood sugar": "glucose",
    "plasma glucose": "glucose",
    "blood sugar": "glucose",
    # hba1c
    "hba1c": "hba1c",
    "hb a1c": "hba1c",
    "hemoglobin a1c": "hba1c",
    "haemoglobin a1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "glycosylated hemoglobin": "hba1c",
    "a1c": "hba1c",
    "glycohemoglobin": "hba1c",
    # creatinine
    "creatinine": "creatinine",
    "serum creatinine": "creatinine",
    "creat": "creatinine",
    "s. creatinine": "creatinine",
    # cholesterol
    "total cholesterol": "total_cholesterol",
    "cholesterol": "total_cholesterol",
    "t. cholesterol": "total_cholesterol",
    "tc": "total_cholesterol",
    "serum cholesterol": "total_cholesterol",
    "chol": "total_cholesterol",
    # triglycerides
    "triglycerides": "triglycerides",
    "triglyceride": "triglycerides",
    "tg": "triglycerides",
    "trigs": "triglycerides",
    "serum triglycerides": "triglycerides",
    # hdl
    "hdl": "hdl",
    "hdl cholesterol": "hdl",
    "hdl-c": "hdl",
    "high density lipoprotein": "hdl",
    "hdl-cholesterol": "hdl",
    # ldl
    "ldl": "ldl",
    "ldl cholesterol": "ldl",
    "ldl-c": "ldl",
    "low density lipoprotein": "ldl",
    "ldl-cholesterol": "ldl",
    # hemoglobin
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "hgb level": "hemoglobin",
    # bun
    "bun": "bun",
    "blood urea nitrogen": "bun",
    "urea nitrogen": "bun",
    "urea": "bun",
    "blood urea": "bun",
    "serum urea": "bun",
    # ── CBC parameters ────────────────────────────────────────────────────────
    # RBC
    "rbc": "rbc_count",
    "rbc count": "rbc_count",
    "red blood cell count": "rbc_count",
    "red blood cells": "rbc_count",
    "red cell count": "rbc_count",
    "erythrocyte count": "rbc_count",
    # WBC
    "wbc": "wbc_count",
    "wbc count": "wbc_count",
    "white blood cell count": "wbc_count",
    "white blood cells": "wbc_count",
    "total leucocyte count": "wbc_count",
    "total leukocyte count": "wbc_count",
    "tlc": "wbc_count",
    "leucocyte count": "wbc_count",
    "leukocyte count": "wbc_count",
    # Platelets
    "platelets": "platelet_count",
    "platelet count": "platelet_count",
    "plt": "platelet_count",
    "thrombocyte count": "platelet_count",
    "platelet": "platelet_count",
    # Hematocrit
    "hematocrit": "hematocrit",
    "haematocrit": "hematocrit",
    "hct": "hematocrit",
    "pcv": "hematocrit",
    "packed cell volume": "hematocrit",
    # MCV
    "mcv": "mcv",
    "mean corpuscular volume": "mcv",
    "mean cell volume": "mcv",
    # MCH
    "mch": "mch",
    "mean corpuscular hemoglobin": "mch",
    "mean cell hemoglobin": "mch",
    # MCHC
    "mchc": "mchc",
    "mean corpuscular hemoglobin concentration": "mchc",
    "mean cell hemoglobin concentration": "mchc",
    # RDW
    "rdw": "rdw",
    "rdw-cv": "rdw",
    "rdw cv": "rdw",
    "red cell distribution width": "rdw",
    "rdw-sd": "rdw_sd",
    # Neutrophils
    "neutrophils": "neutrophils",
    "neutrophil": "neutrophils",
    "neut": "neutrophils",
    "polymorphs": "neutrophils",
    "segmented neutrophils": "neutrophils",
    # Lymphocytes
    "lymphocytes": "lymphocytes",
    "lymphocyte": "lymphocytes",
    "lymph": "lymphocytes",
    # Eosinophils
    "eosinophils": "eosinophils",
    "eosinophil": "eosinophils",
    "eos": "eosinophils",
    # Monocytes
    "monocytes": "monocytes",
    "monocyte": "monocytes",
    "mono": "monocytes",
    # Basophils
    "basophils": "basophils",
    "basophil": "basophils",
    "baso": "basophils",
    # Blood pressure
    "systolic bp": "systolic_bp",
    "systolic blood pressure": "systolic_bp",
    "sbp": "systolic_bp",
    "diastolic bp": "diastolic_bp",
    "diastolic blood pressure": "diastolic_bp",
    "dbp": "diastolic_bp",
}

# ---------------------------------------------------------------------------
# Regex patterns for extracting lab values from free-form OCR text
# ---------------------------------------------------------------------------
# Matches: "Glucose: 5.4 mmol/L" or "HbA1c: 6.2 %" etc.
# Requires an explicit separator (: or =) to avoid matching digits embedded
# in parameter names (e.g. "HbA1c" must not match "1" as the value).
_LAB_LINE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 \-\.]*?)"    # parameter name
    r"\s*[:=]\s*"                                # required separator
    r"(?P<value>\d+(?:\.\d+)?)"                  # numeric value
    r"\s*"
    r"(?P<unit>[A-Za-zµ%][A-Za-z0-9µ%/]*)?",    # optional unit
    re.IGNORECASE,
)

# Matches tabular rows: columns separated by 2+ spaces or tabs
_TABLE_ROW_PATTERN = re.compile(r"[ \t]{2,}|\t")


def _canonicalize_name(raw_name: str) -> Optional[str]:
    """Map a raw OCR name to a canonical parameter name, or None if unknown."""
    key = raw_name.strip().lower()
    return _NAME_ALIASES.get(key)


def normalize_unit(value: float, from_unit: str, param_name: str) -> tuple[float, str]:
    """Convert *value* from *from_unit* to the SI unit for *param_name*.

    Returns (converted_value, si_unit).  If no conversion is defined the
    original value and unit are returned unchanged.

    Requirements: 4.3
    """
    canonical = _canonicalize_name(param_name) or param_name.lower().strip()
    conversions = _UNIT_CONVERSIONS.get(canonical, {})

    # Try exact match first, then case-insensitive
    factor_unit = conversions.get(from_unit)
    if factor_unit is None:
        for k, v in conversions.items():
            if k.lower() == from_unit.lower():
                factor_unit = v
                break

    if factor_unit is None:
        return value, from_unit

    factor, si_unit = factor_unit
    return round(value * factor, 6), si_unit


def flag_abnormal(param: LabParameter) -> LabParameter:
    """Return a copy of *param* with *is_abnormal* set based on clinical reference ranges.

    Requirements: 4.4
    """
    canonical = _canonicalize_name(param.name) or param.name.lower().strip()
    ref = _REFERENCE_RANGES.get(canonical)
    if ref is None:
        # Unknown parameter — cannot flag
        return param.model_copy(update={"is_abnormal": False})

    low, high = ref
    abnormal = False
    if low is not None and param.value < low:
        abnormal = True
    if high is not None and param.value > high:
        abnormal = True

    return param.model_copy(update={
        "is_abnormal": abnormal,
        "reference_range": (
            low if low is not None else 0.0,
            high if high is not None else float("inf"),
        ),
    })


def _parse_table_structure(text: str) -> list[LabParameter]:
    """Detect tabular structures in OCR output and extract lab parameters.

    Looks for lines that appear to be table rows (multiple columns separated
    by whitespace) and tries to map the first column as the parameter name
    and the second numeric column as the value.

    Requirements: 4.2
    """
    params: list[LabParameter] = []
    lines = text.splitlines()

    # Heuristic: a table row has ≥2 whitespace-separated columns
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Split on 2+ spaces or tabs
        cols = [c.strip() for c in _TABLE_ROW_PATTERN.split(stripped) if c.strip()]
        if len(cols) < 2:
            continue

        name_col = cols[0]
        canonical = _canonicalize_name(name_col)
        if canonical is None:
            continue

        # Find the first numeric column after the name; also check the
        # immediately following column for a unit string.
        # Handles formats:
        #   TestName    Value    Unit           (standard)
        #   TestName    Value    RefRange    Unit  (Flabs style — unit is last)
        value: Optional[float] = None
        unit_str = ""
        raw_text = stripped

        remaining_cols = cols[1:]

        # Check if last column looks like a unit (Flabs: Name Value RefRange Unit)
        last_col = remaining_cols[-1] if remaining_cols else ""
        last_is_unit = bool(re.match(r"^[A-Za-zµ%][A-Za-z0-9µ%/\-\.]*$", last_col)) and not re.search(r"\d", last_col)

        for idx, col in enumerate(remaining_cols):
            m = re.match(r"^(\d+(?:\.\d+)?)\s*([A-Za-zµ%][A-Za-z0-9µ%/]*)?$", col)
            if m:
                value = float(m.group(1))
                unit_str = (m.group(2) or "").strip()
                if not unit_str:
                    # Check next column for unit
                    if idx + 1 < len(remaining_cols):
                        next_col = remaining_cols[idx + 1]
                        if re.match(r"^[A-Za-zµ%][A-Za-z0-9µ%/]*$", next_col) and not re.search(r"\d", next_col):
                            unit_str = next_col
                    # If still no unit and last col looks like unit, use it
                    if not unit_str and last_is_unit and last_col != col:
                        unit_str = last_col
                break

        if value is None:
            continue

        norm_value, norm_unit = normalize_unit(value, unit_str, canonical)
        ref = _REFERENCE_RANGES.get(canonical, (0.0, float("inf")))
        low = ref[0] if ref[0] is not None else 0.0
        high = ref[1] if ref[1] is not None else float("inf")

        param = LabParameter(
            name=canonical,
            value=norm_value,
            unit=norm_unit,
            reference_range=(low, high),
            is_abnormal=False,
            raw_text=raw_text,
        )
        params.append(flag_abnormal(param))

    return params


def parse_lab_parameters(text: str) -> list[LabParameter]:
    """Parse OCR text into structured LabParameter objects using regex extraction.

    Combines free-form line parsing with tabular structure detection.
    Deduplicates by canonical parameter name (last occurrence wins for tables,
    first for free-form lines).

    Requirements: 4.1, 4.2
    """
    seen: dict[str, LabParameter] = {}

    # --- Pass 1: free-form regex extraction ---
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        m = _LAB_LINE_PATTERN.search(stripped)
        if not m:
            continue

        raw_name = m.group("name").strip()
        canonical = _canonicalize_name(raw_name)
        if canonical is None:
            continue

        try:
            value = float(m.group("value"))
        except (TypeError, ValueError):
            continue

        unit_str = (m.group("unit") or "").strip()
        norm_value, norm_unit = normalize_unit(value, unit_str, canonical)

        ref = _REFERENCE_RANGES.get(canonical, (0.0, float("inf")))
        low = ref[0] if ref[0] is not None else 0.0
        high = ref[1] if ref[1] is not None else float("inf")

        param = LabParameter(
            name=canonical,
            value=norm_value,
            unit=norm_unit,
            reference_range=(low, high),
            is_abnormal=False,
            raw_text=stripped,
        )
        param = flag_abnormal(param)

        if canonical not in seen:
            seen[canonical] = param

    # --- Pass 2: tabular structure detection (may override free-form) ---
    table_params = _parse_table_structure(text)
    for param in table_params:
        seen[param.name] = param  # table results take precedence

    return list(seen.values())


def extract_lab_parameters(ocr_text: str) -> list[LabParameter]:
    """Full pipeline: parse OCR text → normalize units → flag abnormal values.

    This is the primary entry point for the Document Intelligence component.

    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    params = parse_lab_parameters(ocr_text)
    # flag_abnormal is already applied inside parse_lab_parameters,
    # but we re-apply here to ensure the pipeline is explicit and testable
    # independently of the parser internals.
    return [flag_abnormal(p) for p in params]
