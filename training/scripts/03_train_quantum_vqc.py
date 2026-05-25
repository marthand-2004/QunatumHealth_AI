"""
Script 03: Train PennyLane VQC for all three diseases.

Architecture:
    14D input → PCA(6) → 6-qubit circuit
    AngleEmbedding + StronglyEntanglingLayers × 3
    ADAM optimizer, 80 epochs, batch size 32

Saves:
    - training/models/vqc_{disease}_weights.npy
    - training/models/pca_scaler.joblib
    - training/models/feature_scaler.joblib
    - training/results/training_curves/vqc_{disease}_loss.png
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("[WARN] PennyLane not installed. Install with: pip install pennylane==0.38.0")

SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "training_curves")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level"
]

# VQC hyperparameters
N_QUBITS = 6
N_LAYERS = 3
N_EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.05


def build_vqc_circuit():
    """Build the VQC circuit using PennyLane."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(features, weights):
        qml.AngleEmbedding(features, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return circuit


def binary_cross_entropy(predictions, targets):
    """Binary cross-entropy loss."""
    eps = 1e-7
    predictions = pnp.clip(predictions, eps, 1 - eps)
    return -pnp.mean(targets * pnp.log(predictions) + (1 - targets) * pnp.log(1 - predictions))


def sigmoid(x):
    return 1 / (1 + pnp.exp(-x))


def train_vqc(X_train, y_train, X_test, y_test, disease: str):
    """Train VQC for a single disease."""
    if not PENNYLANE_AVAILABLE:
        print(f"  [SKIP] PennyLane not available")
        return None, []

    print(f"  Building VQC circuit ({N_QUBITS} qubits, {N_LAYERS} layers)...")

    # Initialize weights
    weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    weights = pnp.random.uniform(-np.pi, np.pi, size=weights_shape, requires_grad=True)

    circuit = build_vqc_circuit()
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    n_samples = len(X_train)
    loss_history = []
    auc_history = []

    print(f"  Training VQC for {disease} ({N_EPOCHS} epochs, batch size {BATCH_SIZE})...")

    for epoch in range(N_EPOCHS):
        # Shuffle training data
        perm = np.random.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, BATCH_SIZE):
            X_batch = X_shuffled[i:i + BATCH_SIZE]
            y_batch = y_shuffled[i:i + BATCH_SIZE]

            def cost(w):
                preds = pnp.array([
                    sigmoid(circuit(pnp.array(x, requires_grad=False), w))
                    for x in X_batch
                ])
                return binary_cross_entropy(preds, pnp.array(y_batch, dtype=float))

            weights, batch_loss = opt.step_and_cost(cost, weights)
            epoch_loss += float(batch_loss)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        # Evaluate AUC every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_preds = np.array([
                float(sigmoid(circuit(pnp.array(x, requires_grad=False), weights)))
                for x in X_test
            ])
            auc = roc_auc_score(y_test, test_preds)
            auc_history.append(auc)
            print(f"    Epoch {epoch+1:3d}/{N_EPOCHS} | Loss: {avg_loss:.4f} | Test AUC: {auc:.4f}")

    # Final evaluation
    final_preds = np.array([
        float(sigmoid(circuit(pnp.array(x, requires_grad=False), weights)))
        for x in X_test
    ])
    final_auc = roc_auc_score(y_test, final_preds)
    print(f"  Final Test AUC: {final_auc:.4f}")

    return weights, loss_history


def plot_training_curves(loss_history: list, disease: str):
    """Plot and save VQC training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title(f"VQC Training Loss — {disease.upper()}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"vqc_{disease}_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] Training curve: {path}")


def print_circuit_info():
    """Print quantum circuit architecture details."""
    if not PENNYLANE_AVAILABLE:
        return

    print("\n  Quantum Circuit Architecture:")
    print(f"    Device: default.qubit")
    print(f"    Qubits: {N_QUBITS}")
    print(f"    Layers: {N_LAYERS} × StronglyEntanglingLayers")
    print(f"    Embedding: AngleEmbedding (Y-rotation)")
    print(f"    Measurement: qml.expval(PauliZ(0))")

    weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    n_params = np.prod(weights_shape)
    print(f"    Trainable parameters: {n_params}")

    # Estimate gate count
    # AngleEmbedding: N_QUBITS RY gates
    # StronglyEntanglingLayers: ~3*N_QUBITS rotations + N_QUBITS CNOT per layer
    ry_gates = N_QUBITS
    rot_gates = N_LAYERS * N_QUBITS * 3
    cnot_gates = N_LAYERS * N_QUBITS
    total_gates = ry_gates + rot_gates + cnot_gates
    circuit_depth = 1 + N_LAYERS * 2  # embedding + layers

    print(f"    Gate count: ~{total_gates} (RY: {ry_gates}, Rot: {rot_gates}, CNOT: {cnot_gates})")
    print(f"    Circuit depth: ~{circuit_depth}")


def train_disease_vqc(disease: str, scaler, pca):
    """Full VQC training pipeline for one disease."""
    print(f"\n{'='*70}")
    print(f"Training VQC for: {disease.upper()}")
    print(f"{'='*70}")

    path = os.path.join(PROCESSED_DIR, f"{disease}_aligned.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found.")
        return

    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA reduction: 14D → 6D
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Normalize PCA output to [-π, π] for AngleEmbedding
    pca_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_pca = pca_scaler.fit_transform(X_train_pca)
    X_test_pca = pca_scaler.transform(X_test_pca)

    # Save PCA angle scaler
    pca_angle_path = os.path.join(MODELS_DIR, f"pca_angle_scaler_{disease}.joblib")
    joblib.dump(pca_scaler, pca_angle_path)

    # Train VQC
    weights, loss_history = train_vqc(X_train_pca, y_train, X_test_pca, y_test, disease)

    if weights is not None:
        # Save weights
        weights_path = os.path.join(MODELS_DIR, f"vqc_{disease}_weights.npy")
        np.save(weights_path, np.array(weights))
        print(f"  [SAVED] VQC weights: {weights_path}")

        # Plot training curves
        if loss_history:
            plot_training_curves(loss_history, disease)


if __name__ == "__main__":
    print("=" * 70)
    print("QuantumHealthAI — Quantum VQC Training")
    print("=" * 70)

    print_circuit_info()

    # Load or create feature scaler (fit on all data combined)
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")
    pca_path = os.path.join(MODELS_DIR, "pca_scaler.joblib")

    # Fit scaler on diabetes data (largest dataset) as reference
    diabetes_path = os.path.join(PROCESSED_DIR, "diabetes_aligned.csv")
    if os.path.exists(diabetes_path):
        df_ref = pd.read_csv(diabetes_path)
        X_ref = df_ref[FEATURE_NAMES].values
        scaler = MinMaxScaler()
        scaler.fit(X_ref)
        pca = PCA(n_components=N_QUBITS, random_state=SEED)
        pca.fit(scaler.transform(X_ref))
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca, pca_path)
        print(f"\n[SAVED] Feature scaler: {scaler_path}")
        print(f"[SAVED] PCA scaler: {pca_path}")
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        print("[WARN] No reference dataset found. Using identity scaler.")
        scaler = MinMaxScaler()
        scaler.fit(np.zeros((1, 14)))
        pca = PCA(n_components=N_QUBITS, random_state=SEED)
        pca.fit(np.random.randn(100, 14))

    # Train VQC for each disease
    for disease in ["diabetes", "cvd", "ckd"]:
        try:
            train_disease_vqc(disease, scaler, pca)
        except Exception as e:
            print(f"[FAIL] VQC training for {disease} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("VQC Training Complete")
    print("=" * 70)
    print("\nNext step: python training/scripts/04_hybrid_fusion.py")
