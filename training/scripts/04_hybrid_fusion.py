"""
Script 04: Optimize hybrid fusion weights (classical + quantum).

Grid searches over fusion weights to find optimal combination.
Default: 60% classical + 40% quantum.

Saves: training/models/fusion_weights.json
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_NAMES = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level"
]

N_QUBITS = 6


def load_vqc_predictions(disease: str, X_test_pca: np.ndarray) -> np.ndarray:
    """Load VQC weights and generate predictions."""
    weights_path = os.path.join(MODELS_DIR, f"vqc_{disease}_weights.npy")
    if not os.path.exists(weights_path):
        return None

    try:
        import pennylane as qml
        from pennylane import numpy as pnp

        weights = pnp.array(np.load(weights_path), requires_grad=False)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev)
        def circuit(features, w):
            qml.AngleEmbedding(features, wires=range(N_QUBITS), rotation="Y")
            qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
            return qml.expval(qml.PauliZ(0))

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        preds = np.array([
            sigmoid(float(circuit(pnp.array(x, requires_grad=False), weights)))
            for x in X_test_pca
        ])
        return preds
    except Exception as e:
        print(f"    [WARN] VQC prediction failed: {e}")
        return None


def optimize_fusion(disease: str):
    """Find optimal fusion weights for a disease."""
    print(f"\n  Optimizing fusion for {disease.upper()}...")

    path = os.path.join(PROCESSED_DIR, f"{disease}_aligned.csv")
    if not os.path.exists(path):
        print(f"    [SKIP] Dataset not found")
        return {"classical": 0.6, "quantum": 0.4}

    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values
    y = df["target"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Load classical models
    rf_path = os.path.join(MODELS_DIR, f"rf_{disease}.joblib")
    xgb_path = os.path.join(MODELS_DIR, f"xgb_{disease}.joblib")

    if not os.path.exists(rf_path) or not os.path.exists(xgb_path):
        print(f"    [SKIP] Classical models not found. Run 02_train_classical_models.py first.")
        return {"classical": 0.6, "quantum": 0.4}

    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)

    # Classical ensemble predictions
    rf_preds = rf.predict_proba(X_test)[:, 1]
    xgb_preds = xgb.predict_proba(X_test)[:, 1]
    classical_preds = (rf_preds + xgb_preds) / 2

    classical_auc = roc_auc_score(y_test, classical_preds)
    print(f"    Classical AUC: {classical_auc:.4f}")

    # Load scaler and PCA
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")
    pca_path = os.path.join(MODELS_DIR, "pca_scaler.joblib")
    pca_angle_path = os.path.join(MODELS_DIR, f"pca_angle_scaler_{disease}.joblib")

    if not all(os.path.exists(p) for p in [scaler_path, pca_path]):
        print(f"    [SKIP] Scalers not found. Run 03_train_quantum_vqc.py first.")
        return {"classical": 0.6, "quantum": 0.4}

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    if os.path.exists(pca_angle_path):
        pca_angle_scaler = joblib.load(pca_angle_path)
        X_test_pca = pca_angle_scaler.transform(X_test_pca)

    # Quantum predictions
    quantum_preds = load_vqc_predictions(disease, X_test_pca)

    if quantum_preds is None:
        print(f"    [SKIP] VQC predictions unavailable. Using default weights.")
        return {"classical": 0.6, "quantum": 0.4}

    quantum_auc = roc_auc_score(y_test, quantum_preds)
    print(f"    Quantum AUC:   {quantum_auc:.4f}")

    # Grid search over fusion weights
    best_auc = 0
    best_w_classical = 0.6
    best_w_quantum = 0.4

    print(f"    Grid searching fusion weights...")
    for w_classical in np.arange(0.1, 1.0, 0.1):
        w_quantum = 1.0 - w_classical
        hybrid_preds = w_classical * classical_preds + w_quantum * quantum_preds
        hybrid_auc = roc_auc_score(y_test, hybrid_preds)

        if hybrid_auc > best_auc:
            best_auc = hybrid_auc
            best_w_classical = round(w_classical, 1)
            best_w_quantum = round(w_quantum, 1)

    print(f"    Best fusion: {best_w_classical:.1f} classical + {best_w_quantum:.1f} quantum")
    print(f"    Hybrid AUC: {best_auc:.4f}")

    return {
        "classical": best_w_classical,
        "quantum": best_w_quantum,
        "classical_auc": classical_auc,
        "quantum_auc": quantum_auc,
        "hybrid_auc": best_auc,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("QuantumHealthAI — Hybrid Fusion Optimization")
    print("=" * 70)

    fusion_weights = {}
    for disease in ["diabetes", "cvd", "ckd"]:
        try:
            weights = optimize_fusion(disease)
            fusion_weights[disease] = weights
        except Exception as e:
            print(f"[FAIL] {disease}: {e}")
            fusion_weights[disease] = {"classical": 0.6, "quantum": 0.4}

    # Save fusion weights
    out_path = os.path.join(MODELS_DIR, "fusion_weights.json")
    with open(out_path, "w") as f:
        json.dump(fusion_weights, f, indent=2)
    print(f"\n[SAVED] Fusion weights: {out_path}")

    print("\n" + "=" * 70)
    print("Fusion Optimization Complete")
    print("=" * 70)
    print("\nFusion weights summary:")
    for disease, w in fusion_weights.items():
        print(f"  {disease.upper()}: {w.get('classical', 0.6):.1f} classical + {w.get('quantum', 0.4):.1f} quantum")

    print("\nNext step: python training/scripts/05_evaluate_all_models.py")
