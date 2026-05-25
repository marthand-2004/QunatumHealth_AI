"""End-to-end integration test for the v2 prediction pipeline and what-if analyzer.

Runs the full inference pipeline directly (no HTTP server needed) and validates
all PredictionResponse fields, clinical rule triggering, JSON round-trip, and
what-if analysis core logic.

Tasks: 24.3 (end-to-end prediction) and 24.4 (what-if analysis)
Requirements: 9.1, 9.4, 10.1, 24.1, 8.1, 8.2, 8.3, 8.6
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure workspace root is on sys.path so `backend` package is importable
# when running as: python tests/test_e2e_prediction_v2.py
_WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TEST_LAB_VALUES = {
    "glucose": 145.0,        # mg/dL — elevated (triggers diabetes rule)
    "hba1c": 7.2,            # %
    "bmi": 28.5,             # kg/m²
    "age": 55.0,             # years
    "systolic_bp": 145.0,    # mmHg — elevated (triggers CVD rule)
    "diastolic_bp": 92.0,    # mmHg — elevated
    "cholesterol": 220.0,    # mg/dL
    "creatinine": 1.5,       # mg/dL — elevated (triggers CKD rule)
    "hemoglobin": 13.5,      # g/dL
    "smoking_encoded": 1.0,  # smoker
    "exercise_frequency": 2.0,
}

TEST_LIFESTYLE = {
    "bmi": 28.5,
    "age": 55.0,
    "smoking_encoded": 1.0,
    "exercise_frequency": 2.0,
}

DISEASES = ("diabetes", "cvd", "ckd")

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def check(label: str, condition: bool, detail: str = "") -> bool:
    """Record a single assertion and return whether it passed."""
    _results.append((label, condition, detail))
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _import_services():
    """Import all required backend services; return False on ImportError."""
    try:
        from backend.services.data_governance import DataGovernanceModule
        from backend.services.feature_mapper import FeatureMapper
        from backend.services.classical_ml_v2 import predict_classical_v2
        from backend.services.quantum_engine_v2 import predict_quantum_v2, QuantumTimeoutError
        from backend.services.hybrid_fusion import hybrid_fusion
        from backend.services.clinical_rule_engine import apply_rules
        from backend.services.xai_service_v2 import compute_shap_v2_with_explanation
        from backend.services.limitations_module import get_disclaimer, get_limitations
        from backend.models.prediction_v2 import PredictionResponse, LatencyBreakdown
        return {
            "DataGovernanceModule": DataGovernanceModule,
            "FeatureMapper": FeatureMapper,
            "predict_classical_v2": predict_classical_v2,
            "predict_quantum_v2": predict_quantum_v2,
            "QuantumTimeoutError": QuantumTimeoutError,
            "hybrid_fusion": hybrid_fusion,
            "apply_rules": apply_rules,
            "compute_shap_v2_with_explanation": compute_shap_v2_with_explanation,
            "get_disclaimer": get_disclaimer,
            "get_limitations": get_limitations,
            "PredictionResponse": PredictionResponse,
            "LatencyBreakdown": LatencyBreakdown,
        }
    except ImportError as exc:
        print(f"\n[ERROR] Import failed: {exc}")
        print("Make sure you are running from the workspace root: python tests/test_e2e_prediction_v2.py")
        return None


def _import_what_if():
    """Import what-if analyzer internals; return False on ImportError."""
    try:
        from backend.services.what_if_analyzer import (
            _compute_what_if_sync,
            _build_summary,
            WhatIfRequest,
            WhatIfResponse,
        )
        return {
            "_compute_what_if_sync": _compute_what_if_sync,
            "_build_summary": _build_summary,
            "WhatIfRequest": WhatIfRequest,
            "WhatIfResponse": WhatIfResponse,
        }
    except ImportError as exc:
        print(f"\n[ERROR] What-if import failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Task 24.3 — End-to-End Prediction Test
# ---------------------------------------------------------------------------

def run_e2e_prediction_test(svc: dict) -> list[Any]:
    """Run the full v2 inference pipeline and return PredictionResponse objects."""
    section("TASK 24.3 — End-to-End Prediction Pipeline")

    DataGovernanceModule = svc["DataGovernanceModule"]
    FeatureMapper = svc["FeatureMapper"]
    predict_classical_v2 = svc["predict_classical_v2"]
    predict_quantum_v2 = svc["predict_quantum_v2"]
    QuantumTimeoutError = svc["QuantumTimeoutError"]
    hybrid_fusion = svc["hybrid_fusion"]
    apply_rules = svc["apply_rules"]
    compute_shap_v2_with_explanation = svc["compute_shap_v2_with_explanation"]
    get_disclaimer = svc["get_disclaimer"]
    get_limitations = svc["get_limitations"]
    PredictionResponse = svc["PredictionResponse"]
    LatencyBreakdown = svc["LatencyBreakdown"]

    pipeline_start = time.perf_counter()

    # ── Stage 1: DataGovernanceModule.validate_and_clean() ──────────────────
    t0 = time.perf_counter()
    governance = DataGovernanceModule()
    cleaned_lab_values, warnings = governance.validate_and_clean(TEST_LAB_VALUES)
    feature_extraction_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  [Stage] DataGovernance: {feature_extraction_ms:.1f}ms  "
          f"(warnings: {len(warnings)})")

    # ── Stage 2: FeatureMapper.map_features() per disease ───────────────────
    feature_mapper = FeatureMapper(medians=governance.medians)
    mapper_results: dict = {}
    for disease in DISEASES:
        t0 = time.perf_counter()
        result = feature_mapper.map_features(
            lab_values=cleaned_lab_values,
            lifestyle=TEST_LIFESTYLE,
            disease=disease,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        mapper_results[disease] = result
        print(f"  [Stage] FeatureMapper({disease}): {elapsed:.1f}ms  "
              f"features={len(result.features)}")

    # ── Stage 3: predict_classical_v2() per disease ─────────────────────────
    classical_results: dict = {}
    classical_ms_total = 0.0
    for disease in DISEASES:
        t0 = time.perf_counter()
        rf_prob, xgb_prob = predict_classical_v2(
            mapper_results[disease].features, disease
        )
        elapsed = (time.perf_counter() - t0) * 1000
        classical_ms_total += elapsed
        classical_results[disease] = (rf_prob, xgb_prob)
        print(f"  [Stage] ClassicalML({disease}): {elapsed:.1f}ms  "
              f"rf={rf_prob:.3f} xgb={xgb_prob:.3f}")

    # ── Stage 4: predict_quantum_v2() per disease (with timeout handling) ───
    quantum_results: dict = {}
    quantum_ms_total = 0.0
    for disease in DISEASES:
        t0 = time.perf_counter()
        vqc_prob = None
        try:
            vqc_prob = predict_quantum_v2(mapper_results[disease].features, disease)
        except QuantumTimeoutError:
            print(f"  [Stage] QuantumVQC({disease}): TIMEOUT — using classical-only fusion")
        except Exception as exc:
            print(f"  [Stage] QuantumVQC({disease}): ERROR ({exc}) — using classical-only fusion")
        elapsed = (time.perf_counter() - t0) * 1000
        quantum_ms_total += elapsed
        quantum_results[disease] = vqc_prob
        if vqc_prob is not None:
            print(f"  [Stage] QuantumVQC({disease}): {elapsed:.1f}ms  vqc={vqc_prob:.3f}")

    # ── Stage 5: hybrid_fusion.fuse() per disease ───────────────────────────
    raw_risk_scores: dict = {}
    fusion_ms_total = 0.0
    for disease in DISEASES:
        t0 = time.perf_counter()
        rf_prob, xgb_prob = classical_results[disease]
        vqc_prob = quantum_results[disease]
        risk_score = hybrid_fusion.fuse(
            classical_rf_prob=rf_prob,
            classical_xgb_prob=xgb_prob,
            quantum_prob=vqc_prob,
            disease=disease,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        fusion_ms_total += elapsed
        raw_risk_scores[disease] = risk_score
        print(f"  [Stage] HybridFusion({disease}): {elapsed:.1f}ms  "
              f"risk_score={risk_score:.2f}")

    # ── Stage 6: apply_rules() across all diseases ──────────────────────────
    t0 = time.perf_counter()
    adjusted_scores, triggered_rules_per_disease = apply_rules(
        risk_scores=raw_risk_scores,
        raw_lab_values=TEST_LAB_VALUES,  # use original units for rule evaluation
    )
    clinical_rules_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  [Stage] ClinicalRules: {clinical_rules_ms:.1f}ms")
    for disease in DISEASES:
        rules = triggered_rules_per_disease.get(disease, [])
        print(f"    {disease}: triggered={rules}")

    # ── Stage 7: compute_shap_v2_with_explanation() per disease ─────────────
    shap_results: dict = {}
    xai_ms_total = 0.0
    for disease in DISEASES:
        t0 = time.perf_counter()
        shap_features, explanation, base_value = compute_shap_v2_with_explanation(
            features=mapper_results[disease].features,
            disease=disease,
            model_used="rf",
            risk_score=adjusted_scores[disease],
        )
        elapsed = (time.perf_counter() - t0) * 1000
        xai_ms_total += elapsed
        shap_results[disease] = (shap_features, explanation, base_value)
        print(f"  [Stage] XAI({disease}): {elapsed:.1f}ms  "
              f"top_feature={shap_features[0].feature_name if shap_features else 'N/A'}")

    # ── Stage 8: Build PredictionResponse per disease ───────────────────────
    disclaimer = get_disclaimer()
    limitations = get_limitations()
    responses: list = []

    for disease in DISEASES:
        rf_prob, xgb_prob = classical_results[disease]
        vqc_prob = quantum_results[disease]
        probs = [rf_prob, xgb_prob]
        if vqc_prob is not None:
            probs.append(vqc_prob)
        avg_prob = sum(probs) / len(probs)
        confidence = float(min(1.0, max(0.0, abs(avg_prob - 0.5) * 2.0)))

        shap_features, explanation, base_value = shap_results[disease]

        latency = LatencyBreakdown(
            feature_extraction_ms=feature_extraction_ms,
            classical_ml_ms=classical_ms_total / len(DISEASES),
            vqc_ms=quantum_ms_total / len(DISEASES),
            hybrid_fusion_ms=fusion_ms_total / len(DISEASES),
            clinical_rules_ms=clinical_rules_ms,
            xai_layer_ms=xai_ms_total / len(DISEASES),
            total_ms=(time.perf_counter() - pipeline_start) * 1000,
        )

        response = PredictionResponse(
            disease=disease,
            risk_score=adjusted_scores[disease],
            confidence=confidence,
            explanation=explanation,
            shap_features=shap_features,
            triggered_rules=triggered_rules_per_disease.get(disease, []),
            missing_flags=mapper_results[disease].missing_flags,
            disclaimer=disclaimer,
            limitations=limitations,
        )
        responses.append(response)

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    print(f"\n  Total pipeline latency (excl. OCR): {total_ms:.1f}ms")

    # ── Assertions ───────────────────────────────────────────────────────────
    print("\n--- Assertions ---")

    # 1. Exactly 3 PredictionResponse objects
    check("Returns exactly 3 PredictionResponse objects", len(responses) == 3,
          f"got {len(responses)}")

    # 2. Each disease is in the expected set
    diseases_found = {r.disease for r in responses}
    check("Each response has disease in {diabetes, cvd, ckd}",
          diseases_found == {"diabetes", "cvd", "ckd"},
          f"got {diseases_found}")

    for r in responses:
        d = r.disease

        # 3. risk_score in [0, 100]
        check(f"[{d}] risk_score in [0, 100]",
              0 <= r.risk_score <= 100,
              f"got {r.risk_score}")

        # 4. confidence in [0, 1]
        check(f"[{d}] confidence in [0, 1]",
              0 <= r.confidence <= 1,
              f"got {r.confidence}")

        # 5. risk_level in expected set
        check(f"[{d}] risk_level in {{Low, Moderate, High, Critical}}",
              r.risk_level in {"Low", "Moderate", "High", "Critical"},
              f"got {r.risk_level!r}")

        # 6. explanation is non-empty string
        check(f"[{d}] explanation is non-empty string",
              isinstance(r.explanation, str) and len(r.explanation.strip()) > 0,
              f"got {r.explanation!r}")

        # 7. shap_features has exactly 3 items
        check(f"[{d}] shap_features has exactly 3 items",
              len(r.shap_features) == 3,
              f"got {len(r.shap_features)}")

        # 8. Each shap_features item has required fields
        for i, sf in enumerate(r.shap_features):
            check(f"[{d}] shap_features[{i}] has feature_name",
                  isinstance(sf.feature_name, str) and sf.feature_name)
            check(f"[{d}] shap_features[{i}] has shap_value (float)",
                  isinstance(sf.shap_value, float))
            check(f"[{d}] shap_features[{i}] direction in {{increases risk, decreases risk}}",
                  sf.direction in {"increases risk", "decreases risk"},
                  f"got {sf.direction!r}")

        # 9. triggered_rules is a list
        check(f"[{d}] triggered_rules is a list",
              isinstance(r.triggered_rules, list),
              f"got {type(r.triggered_rules)}")

        # 10. threshold_used in [0, 1]
        check(f"[{d}] threshold_used in [0, 1]",
              0 <= r.threshold_used <= 1,
              f"got {r.threshold_used}")

        # 11. missing_flags is a dict
        check(f"[{d}] missing_flags is a dict",
              isinstance(r.missing_flags, dict),
              f"got {type(r.missing_flags)}")

        # 12. robustness_score in [0, 1]
        check(f"[{d}] robustness_score in [0, 1]",
              0 <= r.robustness_score <= 1,
              f"got {r.robustness_score}")

        # 13. disclaimer is non-empty string
        check(f"[{d}] disclaimer is non-empty string",
              isinstance(r.disclaimer, str) and len(r.disclaimer.strip()) > 0,
              f"got {r.disclaimer!r}")

        # 14. limitations has exactly 3 items
        check(f"[{d}] limitations has exactly 3 items",
              len(r.limitations) == 3,
              f"got {len(r.limitations)}")

    # 15. Total latency check (warning only, not a hard failure)
    if total_ms > 1000:
        print(f"\n  [WARNING] Total latency {total_ms:.1f}ms exceeds 1000ms budget")
    check("Total latency (excl. OCR) < 1000ms",
          total_ms < 1000,
          f"{total_ms:.1f}ms")

    # 16. Clinical rules triggered as expected
    diabetes_resp = next(r for r in responses if r.disease == "diabetes")
    cvd_resp = next(r for r in responses if r.disease == "cvd")
    ckd_resp = next(r for r in responses if r.disease == "ckd")

    check("Diabetes: 'Fasting glucose > 126 mg/dL' rule triggered",
          "Fasting glucose > 126 mg/dL" in diabetes_resp.triggered_rules,
          f"triggered={diabetes_resp.triggered_rules}")

    check("CVD: 'Blood pressure > 140/90 mmHg' rule triggered",
          "Blood pressure > 140/90 mmHg" in cvd_resp.triggered_rules,
          f"triggered={cvd_resp.triggered_rules}")

    check("CKD: 'Creatinine > 1.3 mg/dL' rule triggered",
          "Creatinine > 1.3 mg/dL" in ckd_resp.triggered_rules,
          f"triggered={ckd_resp.triggered_rules}")

    # 17. JSON round-trip
    print("\n  [JSON round-trip check]")
    for r in responses:
        try:
            json_str = r.model_dump_json()
            data = json.loads(json_str)
            r2 = PredictionResponse(**data)
            check(f"[{r.disease}] JSON round-trip: risk_score preserved",
                  abs(r.risk_score - r2.risk_score) < 0.001,
                  f"{r.risk_score} vs {r2.risk_score}")
            check(f"[{r.disease}] JSON round-trip: disease preserved",
                  r.disease == r2.disease)
            check(f"[{r.disease}] JSON round-trip: risk_level preserved",
                  r.risk_level == r2.risk_level)
            check(f"[{r.disease}] JSON round-trip: shap_features count preserved",
                  len(r.shap_features) == len(r2.shap_features))
        except Exception as exc:
            check(f"[{r.disease}] JSON round-trip succeeded", False, str(exc))

    return responses


# ---------------------------------------------------------------------------
# Task 24.4 — What-If Analysis Test
# ---------------------------------------------------------------------------

def run_what_if_test(wi: dict) -> None:
    """Test what-if analyzer core logic without MongoDB."""
    section("TASK 24.4 — What-If Analysis (no DB)")

    _compute_what_if_sync = wi["_compute_what_if_sync"]
    _build_summary = wi["_build_summary"]
    WhatIfRequest = wi["WhatIfRequest"]
    WhatIfResponse = wi["WhatIfResponse"]

    # ── _build_summary tests ─────────────────────────────────────────────────
    print("\n  [_build_summary tests]")

    s1 = _build_summary("diabetes", 72.0, 58.0, -14.0)
    check("_build_summary(diabetes, 72→58, -14) contains 'reduced'",
          "reduced" in s1.lower(), f"got: {s1!r}")
    check("_build_summary(diabetes, 72→58, -14) contains '72'",
          "72" in s1, f"got: {s1!r}")
    check("_build_summary(diabetes, 72→58, -14) contains '58'",
          "58" in s1, f"got: {s1!r}")
    check("_build_summary(diabetes, 72→58, -14) contains '14'",
          "14" in s1, f"got: {s1!r}")

    s2 = _build_summary("cvd", 58.0, 72.0, 14.0)
    check("_build_summary(cvd, 58→72, +14) contains 'increased'",
          "increased" in s2.lower(), f"got: {s2!r}")

    s3 = _build_summary("ckd", 65.0, 65.0, 0.0)
    check("_build_summary(ckd, 65→65, 0) contains 'No change'",
          "No change" in s3, f"got: {s3!r}")

    # ── Direct _compute_what_if_sync test ────────────────────────────────────
    print("\n  [_compute_what_if_sync tests]")

    # Build a feature vector from TEST_LAB_VALUES (all known features)
    all_feature_names = list(TEST_LAB_VALUES.keys())
    all_feature_values = [TEST_LAB_VALUES[k] for k in all_feature_names]

    modified_inputs = {
        "bmi": 22.0,             # healthier BMI
        "exercise_frequency": 5.0,  # more exercise
    }

    t0 = time.perf_counter()
    try:
        response = _compute_what_if_sync(
            feature_names=all_feature_names,
            features=all_feature_values,
            modified_inputs=modified_inputs,
            disease="diabetes",
        )
        elapsed = (time.perf_counter() - t0)
        elapsed_ms = elapsed * 1000

        print(f"  [Stage] _compute_what_if_sync(diabetes): {elapsed_ms:.1f}ms")
        print(f"    original_risk_score={response.original_risk_score}")
        print(f"    new_risk_score={response.new_risk_score}")
        print(f"    delta={response.delta}")
        print(f"    summary={response.summary!r}")

        # Assertions
        check("WhatIfResponse has original_risk_score",
              hasattr(response, "original_risk_score") and isinstance(response.original_risk_score, float))
        check("WhatIfResponse has new_risk_score",
              hasattr(response, "new_risk_score") and isinstance(response.new_risk_score, float))
        check("WhatIfResponse has delta",
              hasattr(response, "delta") and isinstance(response.delta, float))
        check("WhatIfResponse has summary",
              hasattr(response, "summary") and isinstance(response.summary, str))

        check("original_risk_score in [0, 100]",
              0 <= response.original_risk_score <= 100,
              f"got {response.original_risk_score}")
        check("new_risk_score in [0, 100]",
              0 <= response.new_risk_score <= 100,
              f"got {response.new_risk_score}")

        # delta = new - original (within 0.01 tolerance due to rounding)
        expected_delta = response.new_risk_score - response.original_risk_score
        check("delta = new_risk_score - original_risk_score (within 0.01)",
              abs(response.delta - expected_delta) <= 0.01,
              f"delta={response.delta}, expected={expected_delta:.4f}")

        check("summary is non-empty",
              isinstance(response.summary, str) and len(response.summary.strip()) > 0,
              f"got {response.summary!r}")

        check("_compute_what_if_sync completes within 3 seconds",
              elapsed < 3.0,
              f"{elapsed:.2f}s")

    except Exception as exc:
        check("_compute_what_if_sync executed without exception", False, str(exc))
        print(f"  [ERROR] {exc}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary() -> int:
    """Print final pass/fail summary and return exit code (0=all pass, 1=any fail)."""
    section("SUMMARY")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)

    print(f"\n  Total: {total}  |  PASS: {passed}  |  FAIL: {failed}")

    if failed:
        print("\n  Failed assertions:")
        for label, ok, detail in _results:
            if not ok:
                suffix = f"  ({detail})" if detail else ""
                print(f"    [FAIL] {label}{suffix}")

    print()
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nQuantumHealthAI v2 — End-to-End Integration Test")
    print("=" * 60)

    # Import services
    svc = _import_services()
    if svc is None:
        return 1

    wi = _import_what_if()
    if wi is None:
        return 1

    # Task 24.3
    try:
        run_e2e_prediction_test(svc)
    except Exception as exc:
        print(f"\n[FATAL] End-to-end prediction test crashed: {exc}")
        import traceback
        traceback.print_exc()
        check("End-to-end prediction test completed without fatal error", False, str(exc))

    # Task 24.4
    try:
        run_what_if_test(wi)
    except Exception as exc:
        print(f"\n[FATAL] What-if test crashed: {exc}")
        import traceback
        traceback.print_exc()
        check("What-if test completed without fatal error", False, str(exc))

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
