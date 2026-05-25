"""
Script 06 (v2): Robustness testing — evaluate model stability under input perturbations.

For each disease, Gaussian noise is added to the scaled test-set features at two
noise levels (±5% and ±10% of each feature's standard deviation).  The hybrid
fusion pipeline is re-run for each noise level and the resulting risk-score
changes are measured.

Metrics computed per disease per noise level:
    - mean_risk_change   : mean absolute change in Risk_Score vs unperturbed
    - std_risk_change    : std dev of the absolute change
    - pct_changed_risk_level : % of samples whose risk-level category changed

Per-disease summary:
    robustness_score = max(0.0, min(1.0, 1 − (mean_risk_change_at_5pct / 100)))

Artifacts saved:
    training/results/robustness_report.json

Requirements: 19.1, 19.2, 19.3, 19.4
"""
import json
import os
import random
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "processed")
TRAINING_MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

FUSION_WEIGHTS_PATH = os.path.join(TRAINING_MODELS_DIR, "fusion_weights_v2.json")
ROBUSTNESS_REPORT_PATH = os.path.join(RESULTS_DIR, "robustness_report.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Disease-specific feature subsets (must match scripts 02 and 03) ──────────
DISEASE_FEATURE_SUBSETS: dict[str, list[str]] = {
    "diabetes": ["glucose", "hba1c", "bmi", "age", "systolic_bp", "diastolic_bp"],
    "cvd":      ["age", "cholesterol", "systolic_bp", "diastolic_bp", "smoking_encoded", "bmi"],
    "ckd":      ["creatinine", "hemoglobin", "systolic_bp", "diastolic_bp", "age"],
}

DISEASES = ["diabetes", "cvd", "ckd"]

# Noise levels: fraction of each feature's std dev
NOISE_LEVELS = [0.05, 0.10]

# Risk-level thresholds (from design document)
# Low: 0–30, Moderate: 31–60, High: 61–80, Critical: 81–100
def risk_level(score: float) -> str:
    """Map a risk score in [0, 100] to a risk-level category string."""
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Moderate"
    if score <= 80:
        return "High"
    return "Critical"


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_test_set(disease: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the aligned CSV for a disease and return the held-out test split
    (X_test, y_test) using the same stratified 80/20 split with seed=42 as
    the training scripts.
    """
    path = os.path.join(PROCESSED_DIR, f"{disease}_aligned.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 01_preprocess_datasets.py first."
        )

    df = pd.read_csv(path)
    features = DISEASE_FEATURE_SUBSETS[disease]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset for '{disease}' is missing columns: {missing}. "
            "Re-run 01_preprocess_datasets.py."
        )

    X = df[features].values.astype(np.float64)
    y = df["target"].values.astype(int)

    # Same split as scripts 02 and 03 — stratified 80/20, seed=42
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"  Loaded test set: {len(X_test)} samples, {X_test.shape[1]} features: {features}")
    return X_test, y_test


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(disease: str) -> dict:
    """
    Load RF, XGBoost, and (optionally) VQC models plus the disease-specific
    StandardScaler for a given disease.

    Returns a dict with keys: "rf", "xgb", "scaler", and optionally "vqc_weights",
    "pca", "vqc_layers".
    """
    models: dict = {}

    # ── StandardScaler ────────────────────────────────────────────────────────
    scaler_path = os.path.join(TRAINING_MODELS_DIR, f"scaler_{disease}.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}. Run 02_train_classical_models_v2.py first."
        )
    models["scaler"] = joblib.load(scaler_path)
    print(f"  Loaded scaler: {scaler_path}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_path = os.path.join(DATA_MODELS_DIR, f"rf_{disease}.joblib")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(
            f"RF model not found: {rf_path}. Run 02_train_classical_models_v2.py first."
        )
    models["rf"] = joblib.load(rf_path)
    print(f"  Loaded RF model: {rf_path}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_path = os.path.join(DATA_MODELS_DIR, f"xgb_{disease}.joblib")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(
            f"XGBoost model not found: {xgb_path}. Run 02_train_classical_models_v2.py first."
        )
    models["xgb"] = joblib.load(xgb_path)
    print(f"  Loaded XGBoost model: {xgb_path}")

    # ── VQC (optional) ────────────────────────────────────────────────────────
    vqc_weights_path = os.path.join(TRAINING_MODELS_DIR, f"vqc_{disease}_weights.npy")
    pca_path = os.path.join(TRAINING_MODELS_DIR, f"pca_{disease}.joblib")

    if os.path.exists(vqc_weights_path) and os.path.exists(pca_path):
        try:
            import pennylane as qml
            from pennylane import numpy as pnp

            models["vqc_weights"] = pnp.array(
                np.load(vqc_weights_path), requires_grad=False
            )
            models["pca"] = joblib.load(pca_path)

            # Determine best layer count from quantum_justification.json
            justification_path = os.path.join(RESULTS_DIR, "quantum_justification.json")
            best_layers = 2  # default
            if os.path.exists(justification_path):
                with open(justification_path) as f:
                    justification = json.load(f)
                best_layers = int(
                    justification.get(disease, {}).get("best_vqc_layers", 2)
                )
            models["vqc_layers"] = best_layers
            models["pennylane_available"] = True
            print(
                f"  Loaded VQC weights ({best_layers} layers): {vqc_weights_path}"
            )
        except ImportError:
            print(f"  [WARN] PennyLane not installed — VQC will be skipped for {disease}.")
    else:
        print(f"  [INFO] VQC artifacts not found for {disease} — classical-only fusion.")

    return models


# ── Fusion weights loading ────────────────────────────────────────────────────

def load_fusion_weights() -> dict[str, dict[str, float]]:
    """
    Load AUC-weighted fusion weights from fusion_weights_v2.json.

    Falls back to equal weights (rf=0.5, xgb=0.5) per disease when the file
    is absent.

    Returns:
        {"diabetes": {"rf": 0.35, "xgb": 0.40, "vqc": 0.25}, ...}
    """
    if not os.path.exists(FUSION_WEIGHTS_PATH):
        print(
            f"  [WARN] {FUSION_WEIGHTS_PATH} not found. "
            "Using equal weights (rf=0.5, xgb=0.5) for all diseases. "
            "Run 04_hybrid_fusion_v2.py for accurate weights."
        )
        return {d: {"rf": 0.5, "xgb": 0.5} for d in DISEASES}

    with open(FUSION_WEIGHTS_PATH) as f:
        raw = json.load(f)

    # The file stores {"diabetes": {"weights": {...}, "source_aucs": {...}}, ...}
    # or {"diabetes": {"rf": ..., "xgb": ..., "vqc": ...}, ...}
    weights: dict[str, dict[str, float]] = {}
    for disease in DISEASES:
        entry = raw.get(disease, {})
        if "weights" in entry:
            weights[disease] = {k: float(v) for k, v in entry["weights"].items()}
        elif "rf" in entry or "xgb" in entry:
            weights[disease] = {k: float(v) for k, v in entry.items()}
        else:
            print(f"  [WARN] No fusion weights for {disease}. Using equal weights.")
            weights[disease] = {"rf": 0.5, "xgb": 0.5}

    print(f"  Loaded fusion weights from {FUSION_WEIGHTS_PATH}")
    for disease, w in weights.items():
        parts = ", ".join(f"{m}={v:.4f}" for m, v in w.items())
        print(f"    {disease.upper()}: {parts}")

    return weights


# ── Hybrid fusion ─────────────────────────────────────────────────────────────

def _build_vqc_circuit(n_layers: int, n_qubits: int = 6):
    """Build a PennyLane QNode matching the circuit in 03_train_quantum_vqc_v2.py."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(features, weights):
        qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def _expval_to_prob(expval: float) -> float:
    """Convert PauliZ expectation value in [-1, 1] to probability in [0, 1]."""
    return (float(expval) + 1.0) / 2.0


def compute_risk_scores(
    X_scaled: np.ndarray,
    models: dict,
    fusion_weights: dict[str, float],
) -> np.ndarray:
    """
    Compute hybrid-fusion risk scores for a batch of already-scaled feature
    vectors.

    The noise is applied to X_scaled *before* calling this function, so this
    function always receives the (possibly perturbed) scaled features.

    Hybrid fusion formula:
        risk_score = sum(weight_i * prob_i * 100)

    When VQC is unavailable, its weight is redistributed proportionally to RF
    and XGBoost.

    Returns:
        np.ndarray of shape (n_samples,) with risk scores in [0, 100].
    """
    rf_probs = models["rf"].predict_proba(X_scaled)[:, 1]   # shape (n,)
    xgb_probs = models["xgb"].predict_proba(X_scaled)[:, 1]  # shape (n,)

    w_rf = fusion_weights.get("rf", 0.5)
    w_xgb = fusion_weights.get("xgb", 0.5)
    w_vqc = fusion_weights.get("vqc", 0.0)

    vqc_probs: np.ndarray | None = None

    # ── VQC predictions (optional) ────────────────────────────────────────────
    if (
        w_vqc > 0.0
        and models.get("pennylane_available")
        and "vqc_weights" in models
        and "pca" in models
    ):
        try:
            from pennylane import numpy as pnp

            pca = models["pca"]
            vqc_weights = models["vqc_weights"]
            n_layers = models.get("vqc_layers", 2)
            circuit = _build_vqc_circuit(n_layers)

            X_pca = pca.transform(X_scaled)
            vqc_probs = np.array(
                [
                    _expval_to_prob(
                        circuit(pnp.array(x, requires_grad=False), vqc_weights)
                    )
                    for x in X_pca
                ]
            )
        except Exception as exc:
            print(f"    [WARN] VQC inference failed: {exc}. Falling back to classical-only.")
            vqc_probs = None

    # ── Redistribute VQC weight when VQC is unavailable ──────────────────────
    if vqc_probs is None and w_vqc > 0.0:
        classical_total = w_rf + w_xgb
        if classical_total > 0.0:
            w_rf = w_rf + w_vqc * (w_rf / classical_total)
            w_xgb = w_xgb + w_vqc * (w_xgb / classical_total)
        else:
            w_rf = 0.5
            w_xgb = 0.5
        w_vqc = 0.0

    # ── Compute weighted risk score ───────────────────────────────────────────
    if vqc_probs is not None:
        risk_scores = (w_rf * rf_probs + w_xgb * xgb_probs + w_vqc * vqc_probs) * 100.0
    else:
        risk_scores = (w_rf * rf_probs + w_xgb * xgb_probs) * 100.0

    # Clamp to [0, 100]
    return np.clip(risk_scores, 0.0, 100.0)


# ── Robustness evaluation for one disease ────────────────────────────────────

def evaluate_robustness(
    disease: str,
    models: dict,
    fusion_weights: dict[str, float],
) -> dict:
    """
    Evaluate robustness for a single disease.

    Steps:
      1. Load test set and scale with the disease-specific StandardScaler.
      2. Compute baseline (unperturbed) risk scores.
      3. For each noise level in NOISE_LEVELS:
         a. Compute per-feature std dev of the scaled test features.
         b. Add Gaussian noise: noisy = scaled + N(0, noise_level * std_per_feature).
         c. Recompute risk scores.
         d. Measure mean absolute change and % of samples that changed risk level.
      4. Compute robustness_score from the 5% noise level.

    Returns a dict matching the output JSON structure.
    """
    print(f"\n{'='*70}")
    print(f"Robustness evaluation: {disease.upper()}")
    print(f"{'='*70}")

    # ── Load test set ─────────────────────────────────────────────────────────
    X_test, _ = load_test_set(disease)

    # ── Scale test features (transform only — never fit_transform) ────────────
    scaler = models["scaler"]
    X_scaled = scaler.transform(X_test)
    print(
        f"  [DATA LEAKAGE CHECK] Test set ({len(X_test)} samples) scaled using "
        f"transform() only — no fitting on test data."
    )

    # ── Baseline predictions ──────────────────────────────────────────────────
    print(f"\n  Computing baseline (unperturbed) predictions...")
    baseline_scores = compute_risk_scores(X_scaled, models, fusion_weights)
    baseline_levels = np.array([risk_level(s) for s in baseline_scores])
    print(
        f"  Baseline — mean risk score: {baseline_scores.mean():.2f}, "
        f"std: {baseline_scores.std():.2f}"
    )

    # ── Per-feature std dev of the scaled test set ────────────────────────────
    # Using the std dev of the scaled test features so that noise is meaningful
    # in the normalized space (as specified in the task requirements).
    feature_std = X_scaled.std(axis=0)  # shape (n_features,)
    print(
        f"  Feature std devs (scaled): "
        + ", ".join(f"{v:.4f}" for v in feature_std)
    )

    # ── Noise-level loop ──────────────────────────────────────────────────────
    noise_results: dict[str, dict] = {}
    mean_risk_change_at_5pct: float | None = None

    for noise_level in NOISE_LEVELS:
        label = f"noise_{int(noise_level * 100)}pct"
        print(f"\n  Noise level: {noise_level * 100:.0f}% of feature std dev")

        # Reset numpy seed for reproducibility at each noise level
        np.random.seed(SEED)

        # Add Gaussian noise to scaled features
        noise = np.random.normal(
            loc=0.0,
            scale=noise_level * feature_std,  # broadcast over samples
            size=X_scaled.shape,
        )
        X_noisy = X_scaled + noise

        # Recompute predictions with noisy features
        noisy_scores = compute_risk_scores(X_noisy, models, fusion_weights)
        noisy_levels = np.array([risk_level(s) for s in noisy_scores])

        # Mean absolute change in risk score
        abs_changes = np.abs(noisy_scores - baseline_scores)
        mean_change = float(abs_changes.mean())
        std_change = float(abs_changes.std())

        # Percentage of samples that changed risk level category
        changed_mask = noisy_levels != baseline_levels
        pct_changed = float(changed_mask.mean() * 100.0)

        print(
            f"    Mean absolute risk change: {mean_change:.4f} | "
            f"Std: {std_change:.4f} | "
            f"% changed risk level: {pct_changed:.2f}%"
        )

        noise_results[label] = {
            "mean_risk_change": round(mean_change, 4),
            "std_risk_change": round(std_change, 4),
            "pct_changed_risk_level": round(pct_changed, 4),
        }

        if noise_level == 0.05:
            mean_risk_change_at_5pct = mean_change

    # ── Robustness score ──────────────────────────────────────────────────────
    assert mean_risk_change_at_5pct is not None, (
        "5% noise level must be in NOISE_LEVELS to compute robustness_score."
    )
    robustness_score = max(0.0, min(1.0, 1.0 - (mean_risk_change_at_5pct / 100.0)))
    print(f"\n  Robustness score: {robustness_score:.4f}")

    result = {
        "robustness_score": round(robustness_score, 4),
    }
    result.update(noise_results)
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("QuantumHealthAI v2 — Robustness Testing")
    print("=" * 70)
    print(f"Random seeds: numpy={SEED}, random={SEED}")
    print(f"Noise levels: {[f'{int(n*100)}%' for n in NOISE_LEVELS]} of feature std dev")
    print(f"Noise applied to: scaled features (after StandardScaler.transform)")
    print()

    # ── Load fusion weights ───────────────────────────────────────────────────
    print("Loading fusion weights...")
    all_fusion_weights = load_fusion_weights()
    print()

    # ── Evaluate each disease ─────────────────────────────────────────────────
    report: dict[str, dict] = {}

    for disease in DISEASES:
        print(f"\nLoading models for {disease.upper()}...")
        try:
            models = load_models(disease)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {disease}: {exc}")
            continue
        except Exception as exc:
            print(f"  [FAIL] {disease} model loading failed: {exc}")
            import traceback
            traceback.print_exc()
            continue

        fusion_weights = all_fusion_weights.get(disease, {"rf": 0.5, "xgb": 0.5})

        try:
            result = evaluate_robustness(disease, models, fusion_weights)
            report[disease] = result
        except Exception as exc:
            print(f"\n  [FAIL] Robustness evaluation for {disease} failed: {exc}")
            import traceback
            traceback.print_exc()

    # ── Save report ───────────────────────────────────────────────────────────
    if not report:
        print("\n[ERROR] No robustness results computed. Exiting.")
        sys.exit(1)

    with open(ROBUSTNESS_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {ROBUSTNESS_REPORT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Robustness Testing Complete")
    print("=" * 70)
    print("\nSummary:")
    for disease, result in report.items():
        print(f"\n  {disease.upper()}:")
        print(f"    Robustness score: {result['robustness_score']:.4f}")
        for label in [f"noise_{int(n*100)}pct" for n in NOISE_LEVELS]:
            if label in result:
                nr = result[label]
                print(
                    f"    {label}: mean_risk_change={nr['mean_risk_change']:.4f}, "
                    f"std={nr['std_risk_change']:.4f}, "
                    f"pct_changed_level={nr['pct_changed_risk_level']:.2f}%"
                )

    print(f"\nArtifact saved:")
    print(f"  Robustness report → {ROBUSTNESS_REPORT_PATH}")


if __name__ == "__main__":
    main()
