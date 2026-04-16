"""Unit tests for document_intelligence service.

Requirements: 4.1, 4.2, 4.3, 4.4
"""
import pytest

from backend.models.document import LabParameter
from backend.services.document_intelligence import (
    extract_lab_parameters,
    flag_abnormal,
    normalize_unit,
    parse_lab_parameters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(name: str, value: float, unit: str, ref_low: float = 0.0, ref_high: float = 100.0) -> LabParameter:
    return LabParameter(
        name=name,
        value=value,
        unit=unit,
        reference_range=(ref_low, ref_high),
        is_abnormal=False,
        raw_text=f"{name} {value} {unit}",
    )


# ---------------------------------------------------------------------------
# normalize_unit tests
# ---------------------------------------------------------------------------

class TestNormalizeUnit:
    def test_glucose_mg_dl_to_mmol_l(self):
        value, unit = normalize_unit(90.0, "mg/dL", "glucose")
        assert unit == "mmol/L"
        assert abs(value - 90.0 * 0.05551) < 1e-4

    def test_glucose_already_mmol_l(self):
        value, unit = normalize_unit(5.0, "mmol/L", "glucose")
        assert unit == "mmol/L"
        assert value == pytest.approx(5.0)

    def test_creatinine_mg_dl_to_umol_l(self):
        value, unit = normalize_unit(1.0, "mg/dL", "creatinine")
        assert unit == "µmol/L"
        assert abs(value - 88.42) < 0.01

    def test_cholesterol_mg_dl_to_mmol_l(self):
        value, unit = normalize_unit(200.0, "mg/dL", "total_cholesterol")
        assert unit == "mmol/L"
        assert abs(value - 200.0 * 0.02586) < 1e-4

    def test_triglycerides_mg_dl_to_mmol_l(self):
        value, unit = normalize_unit(150.0, "mg/dL", "triglycerides")
        assert unit == "mmol/L"
        assert abs(value - 150.0 * 0.01129) < 1e-4

    def test_hdl_mg_dl_to_mmol_l(self):
        value, unit = normalize_unit(50.0, "mg/dL", "hdl")
        assert unit == "mmol/L"
        assert abs(value - 50.0 * 0.02586) < 1e-4

    def test_ldl_mg_dl_to_mmol_l(self):
        value, unit = normalize_unit(100.0, "mg/dL", "ldl")
        assert unit == "mmol/L"
        assert abs(value - 100.0 * 0.02586) < 1e-4

    def test_hemoglobin_g_dl_unchanged(self):
        value, unit = normalize_unit(14.0, "g/dL", "hemoglobin")
        assert unit == "g/dL"
        assert value == pytest.approx(14.0)

    def test_bun_mg_dl_to_mmol_l(self):
        value, unit = normalize_unit(15.0, "mg/dL", "bun")
        assert unit == "mmol/L"
        assert abs(value - 15.0 * 0.3570) < 1e-4

    def test_hba1c_percent_unchanged(self):
        value, unit = normalize_unit(5.5, "%", "hba1c")
        assert unit == "%"
        assert value == pytest.approx(5.5)

    def test_unknown_unit_passthrough(self):
        value, unit = normalize_unit(42.0, "xyz", "glucose")
        assert value == pytest.approx(42.0)
        assert unit == "xyz"

    def test_unknown_param_passthrough(self):
        value, unit = normalize_unit(10.0, "mg/dL", "unknown_param")
        assert value == pytest.approx(10.0)
        assert unit == "mg/dL"

    def test_case_insensitive_unit(self):
        # "mg/dl" lowercase should still convert
        value, unit = normalize_unit(90.0, "mg/dl", "glucose")
        assert unit == "mmol/L"

    def test_alias_name_resolution(self):
        # "blood glucose" should resolve to "glucose" conversions
        value, unit = normalize_unit(90.0, "mg/dL", "blood glucose")
        assert unit == "mmol/L"


# ---------------------------------------------------------------------------
# flag_abnormal tests
# ---------------------------------------------------------------------------

class TestFlagAbnormal:
    def test_normal_glucose(self):
        param = _make_param("glucose", 5.0, "mmol/L", 3.9, 6.1)
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_high_glucose(self):
        param = _make_param("glucose", 7.5, "mmol/L", 3.9, 6.1)
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_low_glucose(self):
        param = _make_param("glucose", 3.0, "mmol/L", 3.9, 6.1)
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_normal_hba1c(self):
        param = _make_param("hba1c", 5.2, "%", 4.0, 5.6)
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_high_hba1c(self):
        param = _make_param("hba1c", 7.0, "%", 4.0, 5.6)
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_normal_creatinine(self):
        param = _make_param("creatinine", 80.0, "µmol/L", 53.0, 106.0)
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_high_creatinine(self):
        param = _make_param("creatinine", 120.0, "µmol/L", 53.0, 106.0)
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_high_cholesterol(self):
        # total_cholesterol has only upper bound (<5.2)
        param = _make_param("total_cholesterol", 6.0, "mmol/L")
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_normal_cholesterol(self):
        param = _make_param("total_cholesterol", 4.5, "mmol/L")
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_low_hdl_is_abnormal(self):
        # HDL has only lower bound (>1.0)
        param = _make_param("hdl", 0.8, "mmol/L")
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_normal_hdl(self):
        param = _make_param("hdl", 1.5, "mmol/L")
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_high_triglycerides(self):
        param = _make_param("triglycerides", 2.0, "mmol/L")
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_normal_hemoglobin(self):
        param = _make_param("hemoglobin", 14.0, "g/dL", 12.0, 17.0)
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_low_hemoglobin(self):
        param = _make_param("hemoglobin", 10.0, "g/dL", 12.0, 17.0)
        result = flag_abnormal(param)
        assert result.is_abnormal is True

    def test_unknown_param_not_flagged(self):
        param = _make_param("unknown_lab", 999.0, "units")
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_boundary_value_not_abnormal(self):
        # Exactly at the boundary should NOT be flagged
        param = _make_param("glucose", 6.1, "mmol/L", 3.9, 6.1)
        result = flag_abnormal(param)
        assert result.is_abnormal is False

    def test_flag_abnormal_does_not_mutate_original(self):
        param = _make_param("glucose", 7.5, "mmol/L", 3.9, 6.1)
        original_flag = param.is_abnormal
        flag_abnormal(param)
        assert param.is_abnormal == original_flag  # original unchanged


# ---------------------------------------------------------------------------
# parse_lab_parameters tests
# ---------------------------------------------------------------------------

class TestParseLabParameters:
    def test_parse_simple_glucose_line(self):
        text = "Glucose: 5.4 mmol/L"
        params = parse_lab_parameters(text)
        names = [p.name for p in params]
        assert "glucose" in names
        g = next(p for p in params if p.name == "glucose")
        assert g.value == pytest.approx(5.4)
        assert g.unit == "mmol/L"

    def test_parse_glucose_mg_dl_converts(self):
        text = "Glucose: 90 mg/dL"
        params = parse_lab_parameters(text)
        g = next((p for p in params if p.name == "glucose"), None)
        assert g is not None
        assert g.unit == "mmol/L"
        assert abs(g.value - 90 * 0.05551) < 1e-3

    def test_parse_hba1c(self):
        text = "HbA1c: 6.2 %"
        params = parse_lab_parameters(text)
        h = next((p for p in params if p.name == "hba1c"), None)
        assert h is not None
        assert h.value == pytest.approx(6.2)
        assert h.is_abnormal is True  # 6.2 > 5.6

    def test_parse_multiple_params(self):
        text = (
            "Glucose: 5.4 mmol/L\n"
            "HbA1c: 5.2 %\n"
            "Creatinine: 80 µmol/L\n"
        )
        params = parse_lab_parameters(text)
        names = {p.name for p in params}
        assert "glucose" in names
        assert "hba1c" in names
        assert "creatinine" in names

    def test_parse_empty_text(self):
        assert parse_lab_parameters("") == []

    def test_parse_text_with_no_lab_values(self):
        text = "Patient Name: John Doe\nDate: 2024-01-01\nDoctor: Dr. Smith"
        params = parse_lab_parameters(text)
        assert params == []

    def test_parse_alias_fbs(self):
        text = "FBS: 5.0 mmol/L"
        params = parse_lab_parameters(text)
        names = [p.name for p in params]
        assert "glucose" in names

    def test_parse_alias_haemoglobin(self):
        text = "Haemoglobin: 13.5 g/dL"
        params = parse_lab_parameters(text)
        names = [p.name for p in params]
        assert "hemoglobin" in names

    def test_parse_tabular_structure(self):
        text = (
            "Test Name          Value    Unit\n"
            "Glucose            5.4      mmol/L\n"
            "Creatinine         88       µmol/L\n"
        )
        params = parse_lab_parameters(text)
        names = {p.name for p in params}
        assert "glucose" in names
        assert "creatinine" in names

    def test_parse_tabular_mg_dl_converts(self):
        text = "Glucose            90       mg/dL\n"
        params = parse_lab_parameters(text)
        g = next((p for p in params if p.name == "glucose"), None)
        assert g is not None
        assert g.unit == "mmol/L"

    def test_abnormal_flag_set_on_high_value(self):
        text = "Glucose: 8.0 mmol/L"
        params = parse_lab_parameters(text)
        g = next(p for p in params if p.name == "glucose")
        assert g.is_abnormal is True

    def test_normal_flag_on_normal_value(self):
        text = "Glucose: 5.0 mmol/L"
        params = parse_lab_parameters(text)
        g = next(p for p in params if p.name == "glucose")
        assert g.is_abnormal is False

    def test_raw_text_preserved(self):
        text = "Glucose: 5.4 mmol/L"
        params = parse_lab_parameters(text)
        g = next(p for p in params if p.name == "glucose")
        assert "5.4" in g.raw_text

    def test_deduplication_keeps_one_entry_per_param(self):
        text = "Glucose: 5.4 mmol/L\nGlucose: 6.0 mmol/L"
        params = parse_lab_parameters(text)
        glucose_params = [p for p in params if p.name == "glucose"]
        assert len(glucose_params) == 1


# ---------------------------------------------------------------------------
# extract_lab_parameters (full pipeline) tests
# ---------------------------------------------------------------------------

class TestExtractLabParameters:
    def test_full_pipeline_returns_lab_params(self):
        text = (
            "Glucose: 90 mg/dL\n"
            "HbA1c: 7.5 %\n"
            "Creatinine: 1.2 mg/dL\n"
            "Total Cholesterol: 220 mg/dL\n"
            "Triglycerides: 180 mg/dL\n"
            "HDL: 45 mg/dL\n"
            "LDL: 130 mg/dL\n"
            "Hemoglobin: 13.5 g/dL\n"
            "BUN: 18 mg/dL\n"
        )
        params = extract_lab_parameters(text)
        names = {p.name for p in params}
        assert "glucose" in names
        assert "hba1c" in names
        assert "creatinine" in names
        assert "total_cholesterol" in names
        assert "triglycerides" in names
        assert "hdl" in names
        assert "ldl" in names
        assert "hemoglobin" in names
        assert "bun" in names

    def test_full_pipeline_units_normalized(self):
        text = "Glucose: 90 mg/dL"
        params = extract_lab_parameters(text)
        g = next(p for p in params if p.name == "glucose")
        assert g.unit == "mmol/L"

    def test_full_pipeline_abnormal_flags(self):
        text = "HbA1c: 8.0 %"
        params = extract_lab_parameters(text)
        h = next(p for p in params if p.name == "hba1c")
        assert h.is_abnormal is True

    def test_full_pipeline_empty_text(self):
        assert extract_lab_parameters("") == []

    def test_full_pipeline_all_params_have_reference_range(self):
        text = (
            "Glucose: 5.0 mmol/L\n"
            "HbA1c: 5.0 %\n"
        )
        params = extract_lab_parameters(text)
        for p in params:
            assert len(p.reference_range) == 2

    def test_full_pipeline_real_report_format(self):
        """Simulate a realistic OCR output from a blood panel report."""
        text = """
LABORATORY REPORT
Patient: John Doe
Date: 2024-01-15

BIOCHEMISTRY
Fasting Glucose    5.6    mmol/L    (3.9 - 6.1)
HbA1c              5.4    %         (4.0 - 5.6)
Creatinine         95     umol/L    (53 - 106)
Total Cholesterol  4.8    mmol/L    (<5.2)
Triglycerides      1.5    mmol/L    (<1.7)
HDL                1.2    mmol/L    (>1.0)
LDL                3.0    mmol/L    (<3.4)
Hemoglobin         14.5   g/dL      (12 - 17)
BUN                5.0    mmol/L    (2.5 - 7.1)
"""
        params = extract_lab_parameters(text)
        names = {p.name for p in params}
        # At least some key parameters should be extracted
        assert len(params) >= 3
        # All extracted params should have valid values
        for p in params:
            assert p.value > 0
            assert p.unit != ""
