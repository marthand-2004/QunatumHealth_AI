"""
Script 04 (v2): Compute AUC-weighted hybrid fusion weights.

v2 changes vs 04_hybrid_fusion.py:
    - Loads per-model AUC values from evaluation_report.json (produced by
      05_evaluate_v2.py) or falls back to quantum_justification.json (produced
      by 03_train_quantum_vqc_v2.py) when the evaluation report is absent.
    - Computes AUC-weighted fusion weights per disease:
          weight_i = AUC_i / sum(AUC_j)   for RF, XGBoost, VQC
    - Saves fusion_weights_v2.json to training/models/ with per-disease weights.
    - No grid search — weights are derived directly from model performance.

Artifacts saved:
    training/models/fusion_weights_v2.json   — per-disease AUC-weighted weights

Requirements: 4.1, 4.2, 4.3
"""
import json
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

EVAL_REPORT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.json")
QUANTUM_JUSTIFICATION_PATH = os.path.join(RESULTS_DIR, "quantum_justification.json")
FUSION_WEIGHTS_PATH = os.path.join(TRAINING_MODELS_DIR, "fusion_weights_v2.json")

DISEASES = ["diabetes", "cvd", "ckd"]

# ── AUC loading ───────────────────────────────────────────────────────────────


def _load_aucs_from_eval_report(eval_report: dict, disease: str) -> dict[str, float]:
    """
    Extract per-model AUC values from evaluation_report.json for a disease.

    Expected structure (produced by 05_evaluate_v2.py):
        {
          "diabetes": {
            "rf":  {"test_auc": 0.87, "mean_auc": 0.86, ...},
            "xgb": {"test_auc": 0.89, "mean_auc": 0.88, ...},
            "vqc": {"test_auc": 0.84, ...}
          },
          ...
        }

    Falls back to mean_auc when test_auc is absent.
    """
    d = eval_report.get(disease, {})
    aucs: dict[str, float] = {}

    for model in ("rf", "xgb", "vqc"):
        entry = d.get(model, {})
        if isinstance(entry, dict):
            auc = entry.get("test_auc") or entry.get("mean_auc")
            if auc is not None:
                aucs[model] = float(auc)

    return aucs


def _load_aucs_from_quantum_justification(
    justification: dict, disease: str
) -> dict[str, float]:
    """
    Extract per-model AUC values from quantum_justification.json for a disease.

    Expected structure (produced by 03_train_quantum_vqc_v2.py):
        {
          "diabetes": {
            "rf_auc": 0.87,
            "xgb_auc": 0.89,
            "best_vqc_auc": 0.84,
            ...
          },
          ...
        }
    """
    d = justification.get(disease, {})
    aucs: dict[str, float] = {}

    if "rf_auc" in d:
        aucs["rf"] = float(d["rf_auc"])
    if "xgb_auc" in d:
        aucs["xgb"] = float(d["xgb_auc"])
    # Use best_vqc_auc (test-set AUC of the best VQC variant)
    if "best_vqc_auc" in d:
        aucs["vqc"] = float(d["best_vqc_auc"])

    return aucs


def load_auc_values() -> dict[str, dict[str, float]]:
    """
    Load per-model AUC values for all diseases.

    Priority:
      1. evaluation_report.json  (most accurate — includes test-set AUC)
      2. quantum_justification.json  (fallback — produced by script 03)

    Returns:
        {
          "diabetes": {"rf": 0.87, "xgb": 0.89, "vqc": 0.84},
          "cvd":      {"rf": 0.82, "xgb": 0.85, "vqc": 0.80},
          "ckd":      {"rf": 0.91, "xgb": 0.93, "vqc": 0.88},
        }
    """
    # ── Attempt 1: evaluation_report.json ────────────────────────────────────
    if os.path.exists(EVAL_REPORT_PATH):
        print(f"  Loading AUC values from: {EVAL_REPORT_PATH}")
        with open(EVAL_REPORT_PATH) as f:
            eval_report = json.load(f)

        all_aucs: dict[str, dict[str, float]] = {}
        for disease in DISEASES:
            aucs = _load_aucs_from_eval_report(eval_report, disease)
            if aucs:
                all_aucs[disease] = aucs
                print(
                    f"  {disease.upper()}: "
                    + ", ".join(f"{m}={v:.4f}" for m, v in aucs.items())
                )
            else:
                print(f"  [WARN] No AUC data for {disease} in evaluation_report.json")

        if all_aucs:
            return all_aucs

        print("  [WARN] evaluation_report.json contained no usable AUC data.")

    else:
        print(
            f"  [WARN] {EVAL_REPORT_PATH} not found. "
            "Run 05_evaluate_v2.py for the most accurate AUC values."
        )

    # ── Attempt 2: quantum_justification.json ─────────────────────────────────
    if os.path.exists(QUANTUM_JUSTIFICATION_PATH):
        print(f"  Falling back to: {QUANTUM_JUSTIFICATION_PATH}")
        with open(QUANTUM_JUSTIFICATION_PATH) as f:
            justification = json.load(f)

        all_aucs = {}
        for disease in DISEASES:
            aucs = _load_aucs_from_quantum_justification(justification, disease)
            if aucs:
                all_aucs[disease] = aucs
                print(
                    f"  {disease.upper()}: "
                    + ", ".join(f"{m}={v:.4f}" for m, v in aucs.items())
                )
            else:
                print(
                    f"  [WARN] No AUC data for {disease} in quantum_justification.json"
                )

        if all_aucs:
            return all_aucs

        print("  [WARN] quantum_justification.json contained no usable AUC data.")

    # ── No AUC data available ─────────────────────────────────────────────────
    print(
        "\n  [ERROR] No AUC source files found. "
        "Run the following scripts first:\n"
        "    python training/scripts/02_train_classical_models_v2.py\n"
        "    python training/scripts/03_train_quantum_vqc_v2.py\n"
        "    python training/scripts/05_evaluate_v2.py  (recommended)\n"
    )
    sys.exit(1)


# ── Weight computation ────────────────────────────────────────────────────────


def compute_fusion_weights(aucs: dict[str, float]) -> dict[str, float]:
    """
    Compute AUC-weighted fusion weights for a single disease.

    Formula (Req 4.1):
        weight_i = AUC_i / sum(AUC_j)   for each model i in {rf, xgb, vqc}

    When a model's AUC is absent (e.g. VQC not trained), it is excluded and
    the remaining weights are renormalized so they still sum to 1.0.

    Returns:
        {"rf": 0.35, "xgb": 0.40, "vqc": 0.25}  (example)
    """
    if not aucs:
        raise ValueError("Cannot compute fusion weights: no AUC values provided.")

    total_auc = sum(aucs.values())
    if total_auc <= 0.0:
        raise ValueError(
            f"Cannot compute fusion weights: total AUC is {total_auc:.4f} "
            "(all models reporting 0 AUC — check that models were trained correctly)."
        )

    weights = {model: auc / total_auc for model, auc in aucs.items()}

    # Verify weights sum to 1.0 within floating-point tolerance (Property 8)
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, (
        f"Fusion weights sum to {weight_sum:.10f}, expected 1.0 ± 1e-6"
    )

    return weights


def compute_all_fusion_weights(
    all_aucs: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Compute AUC-weighted fusion weights for all diseases.

    Returns:
        {
          "diabetes": {"rf": 0.35, "xgb": 0.40, "vqc": 0.25},
          "cvd":      {"rf": 0.33, "xgb": 0.38, "vqc": 0.29},
          "ckd":      {"rf": 0.34, "xgb": 0.37, "vqc": 0.29},
        }
    """
    fusion_weights: dict[str, dict[str, float]] = {}

    for disease in DISEASES:
        if disease not in all_aucs:
            print(f"  [SKIP] {disease.upper()}: no AUC data available")
            continue

        aucs = all_aucs[disease]
        weights = compute_fusion_weights(aucs)
        fusion_weights[disease] = weights

        weight_sum = sum(weights.values())
        print(
            f"  {disease.upper()}: "
            + ", ".join(f"{m}={w:.6f}" for m, w in weights.items())
            + f"  (sum={weight_sum:.8f})"
        )

    return fusion_weights


# ── Persistence ───────────────────────────────────────────────────────────────


def save_fusion_weights(
    fusion_weights: dict[str, dict[str, float]],
    all_aucs: dict[str, dict[str, float]],
) -> None:
    """
    Save fusion_weights_v2.json to training/models/.

    The file includes both the computed weights and the source AUC values
    for traceability (Req 4.3).

    Structure:
        {
          "diabetes": {
            "weights": {"rf": 0.35, "xgb": 0.40, "vqc": 0.25},
            "source_aucs": {"rf": 0.87, "xgb": 0.89, "vqc": 0.84}
          },
          ...
        }
    """
    os.makedirs(TRAINING_MODELS_DIR, exist_ok=True)

    output: dict = {}
    for disease, weights in fusion_weights.items():
        output[disease] = {
            "weights": {m: round(w, 8) for m, w in weights.items()},
            "source_aucs": {
                m: round(v, 4) for m, v in all_aucs.get(disease, {}).items()
            },
        }

    with open(FUSION_WEIGHTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  [SAVED] {FUSION_WEIGHTS_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 70)
    print("QuantumHealthAI v2 — AUC-Weighted Hybrid Fusion")
    print("=" * 70)
    print()
    print("Formula: weight_i = AUC_i / sum(AUC_j)  for RF, XGBoost, VQC")
    print()

    # ── Load AUC values ───────────────────────────────────────────────────────
    print("Loading AUC values...")
    all_aucs = load_auc_values()

    # ── Compute weights ───────────────────────────────────────────────────────
    print("\nComputing AUC-weighted fusion weights...")
    fusion_weights = compute_all_fusion_weights(all_aucs)

    if not fusion_weights:
        print("\n[ERROR] No fusion weights computed. Exiting.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_fusion_weights(fusion_weights, all_aucs)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Hybrid Fusion v2 Complete")
    print("=" * 70)
    print("\nFusion weights summary:")
    for disease, weights in fusion_weights.items():
        parts = ", ".join(f"{m}={w:.4f}" for m, w in weights.items())
        print(f"  {disease.upper()}: {parts}")

    print(f"\nArtifact saved:")
    print(f"  Fusion weights → {FUSION_WEIGHTS_PATH}")
    print("\nNext step: python training/scripts/05_evaluate_v2.py")
