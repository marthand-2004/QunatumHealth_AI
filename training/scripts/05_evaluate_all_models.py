"""
Script 05: Comprehensive evaluation of all models.

Generates:
    - results/performance_table.csv
    - results/confusion_matrices/*.png
    - results/shap_plots/*.png
    - results/calibration_curves/*.png
    - results/latency_benchmark.csv
    - Prints full comparison table to stdout
"""
import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.calibration import calibration_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Install with: pip install shap")

SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

for d in ["confusion_matrices", "shap_plots", "calibration_curves", "roc_curves"]:
    os.makedirs(os.path.join(RESULTS_DIR, d), exist_ok=True)

FEATURE_NAMES = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level"
]

FEATURE_LABELS = {
    "glucose": "Blood Glucose",
    "hba1c": "HbA1c",
    "creatinine": "Creatinine",
    "cholesterol": "Cholesterol",
    "triglycerides": "Triglycerides",
    "hemoglobin": "Hemoglobin",
    "bmi": "BMI",
    "age": "Age",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "smoking_encoded": "Smoking",
    "exercise_frequency": "Exercise",
    "sleep_hours": "Sleep Hours",
    "stress_level": "Stress Level",
}

N_QUBITS = 6


def load_test_data(disease: str):
    path = os.path.join(PROCESSED_DIR, f"{disease}_aligned.csv")
    if not os.path.exists(path):
        return None, None, None, None
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_classical_preds(disease: str, X_test: np.ndarray):
    rf = joblib.load(os.path.join(MODELS_DIR, f"rf_{disease}.joblib"))
    xgb = joblib.load(os.path.join(MODELS_DIR, f"xgb_{disease}.joblib"))
    rf_p = rf.predict_proba(X_test)[:, 1]
    xgb_p = xgb.predict_proba(X_test)[:, 1]
    return (rf_p + xgb_p) / 2, rf, xgb


def get_quantum_preds(disease: str, X_test: np.ndarray):
    try:
        import pennylane as qml
        from pennylane import numpy as pnp

        scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.joblib"))
        pca = joblib.load(os.path.join(MODELS_DIR, "pca_scaler.joblib"))
        weights = pnp.array(np.load(os.path.join(MODELS_DIR, f"vqc_{disease}_weights.npy")), requires_grad=False)

        X_scaled = scaler.transform(X_test)
        X_pca = pca.transform(X_scaled)

        pca_angle_path = os.path.join(MODELS_DIR, f"pca_angle_scaler_{disease}.joblib")
        if os.path.exists(pca_angle_path):
            X_pca = joblib.load(pca_angle_path).transform(X_pca)

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
            for x in X_pca
        ])
        return preds
    except Exception as e:
        print(f"    [WARN] Quantum preds failed: {e}")
        return None


def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, preds_dict: dict, title: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Classical": "blue", "Quantum": "red", "Hybrid": "green"}
    for name, y_proba in preds_dict.items():
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, "gray"),
                label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_shap(model, X_test: np.ndarray, disease: str, model_name: str):
    if not SHAP_AVAILABLE:
        return
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
            show=False, max_display=10
        )
        plt.title(f"SHAP Feature Importance — {disease.upper()} ({model_name})")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "shap_plots", f"shap_{disease}_{model_name.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    [SAVED] SHAP plot: {path}")

        # Print top 3 features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
        print(f"    Top 3 features for {disease} ({model_name}):")
        for i, idx in enumerate(top3_idx, 1):
            print(f"      {i}. {FEATURE_LABELS.get(FEATURE_NAMES[idx], FEATURE_NAMES[idx])}: {mean_abs_shap[idx]:.4f}")
    except Exception as e:
        print(f"    [WARN] SHAP failed: {e}")


def plot_calibration(y_true, preds_dict: dict, title: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    colors = {"Classical": "blue", "Quantum": "red", "Hybrid": "green"}
    for name, y_proba in preds_dict.items():
        if y_proba is None:
            continue
        try:
            fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
            ax.plot(mean_pred, fraction_pos, "s-", color=colors.get(name, "gray"),
                    label=name, linewidth=2)
        except Exception:
            pass
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def benchmark_latency(disease: str, X_test: np.ndarray, n_runs: int = 100):
    """Benchmark inference latency."""
    results = {}

    # Classical
    try:
        rf = joblib.load(os.path.join(MODELS_DIR, f"rf_{disease}.joblib"))
        xgb = joblib.load(os.path.join(MODELS_DIR, f"xgb_{disease}.joblib"))
        sample = X_test[:1]
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            rf.predict_proba(sample)
            xgb.predict_proba(sample)
            times.append(time.perf_counter() - t0)
        results["classical_ms"] = np.mean(times) * 1000
        results["classical_std_ms"] = np.std(times) * 1000
    except Exception:
        results["classical_ms"] = None

    return results


def evaluate_disease(disease: str):
    """Full evaluation for one disease."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {disease.upper()}")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test = load_test_data(disease)
    if X_test is None:
        print(f"  [SKIP] Dataset not found")
        return {}

    all_metrics = {}

    # Classical
    try:
        classical_preds, rf, xgb = get_classical_preds(disease, X_test)
        m = compute_metrics(y_test, classical_preds)
        all_metrics["classical"] = m
        print(f"\n  Classical (RF+XGB ensemble):")
        for k, v in m.items():
            print(f"    {k:12s}: {v:.4f}")

        # Confusion matrix
        y_pred_classical = (classical_preds >= 0.5).astype(int)
        plot_confusion_matrix(
            y_test, y_pred_classical,
            f"Classical — {disease.upper()}",
            os.path.join(RESULTS_DIR, "confusion_matrices", f"cm_classical_{disease}.png")
        )

        # SHAP
        print(f"\n  SHAP analysis (Random Forest):")
        plot_shap(rf, X_test, disease, "RF")

    except Exception as e:
        print(f"  [FAIL] Classical evaluation: {e}")

    # Quantum
    quantum_preds = get_quantum_preds(disease, X_test)
    if quantum_preds is not None:
        m = compute_metrics(y_test, quantum_preds)
        all_metrics["quantum"] = m
        print(f"\n  Quantum (VQC):")
        for k, v in m.items():
            print(f"    {k:12s}: {v:.4f}")

    # Hybrid
    if quantum_preds is not None and "classical" in all_metrics:
        fusion_path = os.path.join(MODELS_DIR, "fusion_weights.json")
        if os.path.exists(fusion_path):
            with open(fusion_path) as f:
                fw = json.load(f)
            w_c = fw.get(disease, {}).get("classical", 0.6)
            w_q = fw.get(disease, {}).get("quantum", 0.4)
        else:
            w_c, w_q = 0.6, 0.4

        hybrid_preds = w_c * classical_preds + w_q * quantum_preds
        m = compute_metrics(y_test, hybrid_preds)
        all_metrics["hybrid"] = m
        print(f"\n  Hybrid ({w_c:.1f}×Classical + {w_q:.1f}×Quantum):")
        for k, v in m.items():
            print(f"    {k:12s}: {v:.4f}")

        # ROC curves
        plot_roc_curve(
            y_test,
            {"Classical": classical_preds, "Quantum": quantum_preds, "Hybrid": hybrid_preds},
            f"ROC Curves — {disease.upper()}",
            os.path.join(RESULTS_DIR, "roc_curves", f"roc_{disease}.png")
        )

        # Calibration
        plot_calibration(
            y_test,
            {"Classical": classical_preds, "Quantum": quantum_preds, "Hybrid": hybrid_preds},
            f"Calibration — {disease.upper()}",
            os.path.join(RESULTS_DIR, "calibration_curves", f"calibration_{disease}.png")
        )

    # Latency
    latency = benchmark_latency(disease, X_test)
    all_metrics["latency"] = latency

    return all_metrics


if __name__ == "__main__":
    print("=" * 70)
    print("QuantumHealthAI — Model Evaluation")
    print("=" * 70)

    all_results = {}
    for disease in ["diabetes", "cvd", "ckd"]:
        try:
            all_results[disease] = evaluate_disease(disease)
        except Exception as e:
            print(f"[FAIL] {disease}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    print("\n" + "=" * 90)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Disease':<12} {'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10}")
    print("-" * 90)

    rows = []
    for disease, results in all_results.items():
        for model_type in ["classical", "quantum", "hybrid"]:
            if model_type in results:
                m = results[model_type]
                print(f"{disease:<12} {model_type:<12} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['auc']:<10.4f}")
                rows.append({
                    "disease": disease,
                    "model": model_type,
                    **m
                })

    # Save performance table
    if rows:
        df_results = pd.DataFrame(rows)
        out_path = os.path.join(RESULTS_DIR, "performance_table.csv")
        df_results.to_csv(out_path, index=False)
        print(f"\n[SAVED] Performance table: {out_path}")

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print("\nNext step: python training/scripts/06_demo_inference.py")
