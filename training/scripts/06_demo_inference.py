"""
Script 06: Real-time doctor prediction demo.

Simulates a complete end-to-end inference:
    1. Takes sample patient blood report values
    2. Runs through all 6 models (RF, XGBoost, VQC × 3 diseases)
    3. Outputs risk scores + SHAP explanation + LLM recommendation
    4. Benchmarks total latency (target: < 3 seconds)

Usage:
    python training/scripts/06_demo_inference.py
"""
import os
import sys
import time
import json
import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
N_QUBITS = 6

FEATURE_NAMES = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level"
]

FEATURE_LABELS = {
    "glucose": "Blood Glucose (mmol/L)",
    "hba1c": "HbA1c (%)",
    "creatinine": "Creatinine (µmol/L)",
    "cholesterol": "Total Cholesterol (mmol/L)",
    "triglycerides": "Triglycerides (mmol/L)",
    "hemoglobin": "Hemoglobin (g/dL)",
    "bmi": "BMI (kg/m²)",
    "age": "Age (years)",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "smoking_encoded": "Smoking Status (0=never, 0.5=former, 1=current)",
    "exercise_frequency": "Exercise (days/week)",
    "sleep_hours": "Sleep (hours/night)",
    "stress_level": "Stress Level (1-10)",
}

# Sample patient — high-risk profile
SAMPLE_PATIENT = {
    "name": "Patient A (High-Risk Demo)",
    "features": {
        "glucose": 7.8,        # Elevated (normal: 3.9-6.1 mmol/L)
        "hba1c": 7.2,          # Diabetic range (normal: 4.0-5.6%)
        "creatinine": 130.0,   # Elevated (normal: 53-106 µmol/L)
        "cholesterol": 6.2,    # High (normal: <5.2 mmol/L)
        "triglycerides": 2.1,  # High (normal: <1.7 mmol/L)
        "hemoglobin": 11.5,    # Low (normal: 12-17 g/dL)
        "bmi": 31.5,           # Obese
        "age": 58.0,
        "systolic_bp": 148.0,  # Hypertensive
        "diastolic_bp": 92.0,
        "smoking_encoded": 0.5,  # Former smoker
        "exercise_frequency": 1.0,  # Low
        "sleep_hours": 5.5,    # Poor sleep
        "stress_level": 8.0,   # High stress
    }
}

SAMPLE_PATIENT_NORMAL = {
    "name": "Patient B (Low-Risk Demo)",
    "features": {
        "glucose": 5.1,
        "hba1c": 5.2,
        "creatinine": 75.0,
        "cholesterol": 4.5,
        "triglycerides": 1.1,
        "hemoglobin": 14.5,
        "bmi": 23.0,
        "age": 35.0,
        "systolic_bp": 118.0,
        "diastolic_bp": 76.0,
        "smoking_encoded": 0.0,
        "exercise_frequency": 5.0,
        "sleep_hours": 7.5,
        "stress_level": 3.0,
    }
}


def get_feature_vector(patient: dict) -> np.ndarray:
    return np.array([patient["features"][f] for f in FEATURE_NAMES]).reshape(1, -1)


def classical_predict(disease: str, X: np.ndarray) -> float:
    rf = joblib.load(os.path.join(MODELS_DIR, f"rf_{disease}.joblib"))
    xgb = joblib.load(os.path.join(MODELS_DIR, f"xgb_{disease}.joblib"))
    rf_p = rf.predict_proba(X)[0, 1]
    xgb_p = xgb.predict_proba(X)[0, 1]
    return (rf_p + xgb_p) / 2


def quantum_predict(disease: str, X: np.ndarray) -> float:
    try:
        import pennylane as qml
        from pennylane import numpy as pnp

        scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.joblib"))
        pca = joblib.load(os.path.join(MODELS_DIR, "pca_scaler.joblib"))
        weights = pnp.array(np.load(os.path.join(MODELS_DIR, f"vqc_{disease}_weights.npy")), requires_grad=False)

        X_scaled = scaler.transform(X)
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

        x = pnp.array(X_pca[0], requires_grad=False)
        expval = float(circuit(x, weights))
        return 1 / (1 + np.exp(-expval))  # sigmoid
    except Exception as e:
        return None


def get_shap_top3(disease: str, X: np.ndarray) -> list:
    try:
        import shap
        rf = joblib.load(os.path.join(MODELS_DIR, f"rf_{disease}.joblib"))
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        sv = shap_values[0]
        top3_idx = np.argsort(np.abs(sv))[::-1][:3]
        return [
            {
                "feature": FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]),
                "shap_value": float(sv[i]),
                "direction": "↑ increases risk" if sv[i] > 0 else "↓ decreases risk"
            }
            for i in top3_idx
        ]
    except Exception:
        return []


def generate_recommendation(disease: str, risk_score: float, top_features: list) -> str:
    """Template-based recommendation (no LLM API needed for demo)."""
    level = "HIGH" if risk_score >= 70 else "MODERATE" if risk_score >= 40 else "LOW"
    recs = {
        "diabetes": {
            "HIGH": "Immediate HbA1c monitoring and endocrinologist referral recommended. Consider metformin therapy.",
            "MODERATE": "Lifestyle modification: reduce refined carbohydrates, increase fiber intake, 30 min daily walk.",
            "LOW": "Maintain healthy diet and regular exercise. Annual glucose screening recommended.",
        },
        "cvd": {
            "HIGH": "Cardiology referral urgently recommended. Statin therapy and BP management indicated.",
            "MODERATE": "Mediterranean diet, omega-3 supplementation, and BP monitoring advised.",
            "LOW": "Continue heart-healthy lifestyle. Lipid panel check every 2 years.",
        },
        "ckd": {
            "HIGH": "Nephrology referral recommended. Low-protein diet, sodium restriction, BP control critical.",
            "MODERATE": "Monitor creatinine and eGFR quarterly. Adequate hydration and avoid NSAIDs.",
            "LOW": "Annual kidney function tests. Maintain healthy BP and blood sugar.",
        },
    }
    base = recs.get(disease, {}).get(level, "Consult your physician.")
    if top_features:
        top_name = top_features[0]["feature"].split(" (")[0]
        base += f" Key driver: {top_name}."
    return base


def run_demo(patient: dict):
    """Run full inference pipeline for a patient."""
    print(f"\n{'='*70}")
    print(f"PATIENT: {patient['name']}")
    print(f"{'='*70}")

    print("\nBlood Panel Values:")
    for feat, val in patient["features"].items():
        label = FEATURE_LABELS.get(feat, feat)
        print(f"  {label}: {val}")

    X = get_feature_vector(patient)

    # Load fusion weights
    fusion_path = os.path.join(MODELS_DIR, "fusion_weights.json")
    fusion_weights = {}
    if os.path.exists(fusion_path):
        with open(fusion_path) as f:
            fusion_weights = json.load(f)

    print(f"\n{'─'*70}")
    print("PREDICTION RESULTS")
    print(f"{'─'*70}")

    total_start = time.perf_counter()

    for disease in ["diabetes", "cvd", "ckd"]:
        print(f"\n  {disease.upper()}:")

        # Classical
        t0 = time.perf_counter()
        try:
            c_prob = classical_predict(disease, X)
            c_score = c_prob * 100
            c_time = (time.perf_counter() - t0) * 1000
            print(f"    Classical score:  {c_score:5.1f}%  ({c_time:.1f}ms)")
        except Exception as e:
            c_score = 50.0
            print(f"    Classical: [FAIL] {e}")

        # Quantum
        t0 = time.perf_counter()
        q_prob = quantum_predict(disease, X)
        q_time = (time.perf_counter() - t0) * 1000
        if q_prob is not None:
            q_score = q_prob * 100
            print(f"    Quantum score:    {q_score:5.1f}%  ({q_time:.1f}ms)")
        else:
            q_score = None
            print(f"    Quantum: [UNAVAILABLE]")

        # Hybrid
        w_c = fusion_weights.get(disease, {}).get("classical", 0.6)
        w_q = fusion_weights.get(disease, {}).get("quantum", 0.4)
        if q_score is not None:
            hybrid_score = w_c * c_score + w_q * q_score
        else:
            hybrid_score = c_score
        print(f"    Hybrid score:     {hybrid_score:5.1f}%  (w_c={w_c}, w_q={w_q})")

        # Risk level
        level = "🔴 HIGH" if hybrid_score >= 70 else "🟡 MODERATE" if hybrid_score >= 40 else "🟢 LOW"
        print(f"    Risk Level:       {level}")

        # SHAP top 3
        top3 = get_shap_top3(disease, X)
        if top3:
            print(f"    Top contributing factors:")
            for i, f in enumerate(top3, 1):
                print(f"      {i}. {f['feature']}: {f['shap_value']:+.3f} ({f['direction']})")

        # Recommendation
        rec = generate_recommendation(disease, hybrid_score, top3)
        print(f"    Recommendation: {rec}")

    total_time = (time.perf_counter() - total_start) * 1000
    print(f"\n{'─'*70}")
    print(f"Total inference time: {total_time:.1f}ms")
    if total_time < 3000:
        print(f"✅ Under 3-second target")
    else:
        print(f"⚠️  Exceeded 3-second target (quantum simulation is slow on CPU)")
    print(f"{'─'*70}")


if __name__ == "__main__":
    print("=" * 70)
    print("QuantumHealthAI — Real-Time Inference Demo")
    print("=" * 70)

    # Check models exist
    required = ["rf_diabetes.joblib", "xgb_diabetes.joblib"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"\n[WARN] Missing models: {missing}")
        print("Run training scripts 01-04 first to train models.")
        print("Running with demo mode (random predictions)...")

    run_demo(SAMPLE_PATIENT)
    run_demo(SAMPLE_PATIENT_NORMAL)

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)
