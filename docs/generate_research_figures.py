"""
Research-grade visualization script for QuantumHealthAI v2 Enhanced.
Generates all figures for the project overview document.

Run from project root:
    python docs/generate_research_figures.py

Outputs saved to: docs/figures/
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "training", "results")
MODELS_DIR = os.path.join(ROOT, "training", "models")
DATA_MODELS_DIR = os.path.join(ROOT, "data", "models")
PROCESSED_DIR = os.path.join(ROOT, "training", "processed")
OUT_DIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
DISEASES = ["diabetes", "cvd", "ckd"]
DISEASE_LABELS = {"diabetes": "Diabetes", "cvd": "CVD", "ckd": "CKD"}
DISEASE_COLORS = {"diabetes": "#E74C3C", "cvd": "#3498DB", "ckd": "#2ECC71"}
MODEL_COLORS = {"rf": "#4C72B0", "xgb": "#DD8452", "vqc": "#55A868"}
MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "vqc": "VQC (Quantum)"}

DISEASE_FEATURE_SUBSETS = {
    "diabetes": ["glucose", "hba1c", "bmi", "age", "systolic_bp", "diastolic_bp"],
    "cvd":      ["age", "cholesterol", "systolic_bp", "diastolic_bp", "smoking_encoded", "bmi"],
    "ckd":      ["creatinine", "hemoglobin", "systolic_bp", "diastolic_bp", "age"],
}

# Research-grade style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

def load_test_data(disease):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{disease}_aligned.csv"))
    features = DISEASE_FEATURE_SUBSETS[disease]
    X = df[features].values
    y = df["target"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = joblib.load(os.path.join(MODELS_DIR, f"scaler_{disease}.joblib"))
    X_test_s = scaler.transform(X_test)
    return X_test_s, y_test

print("Generating research figures for QuantumHealthAI v2 Enhanced...")
print(f"Output directory: {OUT_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Model Performance Comparison (AUC Bar Chart)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_model_performance():
    print("\n[Fig 1] Model Performance Comparison...")
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json")) as f:
        report = json.load(f)

    diseases = DISEASES
    models = ["rf", "xgb", "vqc"]
    x = np.arange(len(diseases))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Test AUC
    ax = axes[0]
    for i, model in enumerate(models):
        aucs = [report[d][model]["test_auc"] for d in diseases]
        bars = ax.bar(x + i*width - width, aucs, width, label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Disease")
    ax.set_ylabel("AUC (Area Under ROC Curve)")
    ax.set_title("Test Set AUC by Model and Disease", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DISEASE_LABELS[d] for d in diseases])
    ax.set_ylim(0.5, 1.08)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.legend(loc="lower right", fontsize=9)

    # F1 Score
    ax = axes[1]
    for i, model in enumerate(models):
        f1s = [report[d][model]["f1"] for d in diseases]
        bars = ax.bar(x + i*width - width, f1s, width, label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Disease")
    ax.set_ylabel("F1 Score")
    ax.set_title("Test Set F1 Score by Model and Disease", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DISEASE_LABELS[d] for d in diseases])
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("QuantumHealthAI v2 — Model Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_model_performance.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")

fig1_model_performance()

# =============================================================================
# FIGURE 1: Multi-panel ROC Curves (all models, all diseases)
# =============================================================================
def fig1_roc_curves():
    print("\n[Fig 1] Multi-panel ROC curves...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ROC Curves — QuantumHealthAI v2 Enhanced", fontsize=15, fontweight="bold", y=1.02)

    for ax, disease in zip(axes, DISEASES):
        X_test_s, y_test = load_test_data(disease)
        ax.plot([0,1],[0,1],"k--",alpha=0.4,linewidth=1,label="Random (AUC=0.50)")

        for model_name, color in MODEL_COLORS.items():
            path = os.path.join(DATA_MODELS_DIR if model_name != "vqc" else MODELS_DIR,
                                f"{model_name}_{disease}.joblib" if model_name != "vqc" else "")
            if model_name == "vqc":
                # Load VQC predictions via PennyLane
                try:
                    import pennylane as qml
                    from pennylane import numpy as pnp
                    pca = joblib.load(os.path.join(MODELS_DIR, f"pca_{disease}.joblib"))
                    weights_np = np.load(os.path.join(MODELS_DIR, f"vqc_{disease}_weights.npy"))
                    weights = pnp.array(weights_np, requires_grad=False)
                    n_q = pca.n_components_
                    X_pca = pca.transform(X_test_s)
                    dev = qml.device("lightning.qubit", wires=n_q)
                    @qml.qnode(dev)
                    def circuit(f, w):
                        qml.AngleEmbedding(f, wires=range(n_q), rotation="Y")
                        qml.StronglyEntanglingLayers(w, wires=range(n_q))
                        return qml.expval(qml.PauliZ(0))
                    y_score = np.array([float((circuit(pnp.array(x,requires_grad=False),weights)+1)/2) for x in X_pca])
                    auc = roc_auc_score(y_test, y_score)
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    ax.plot(fpr, tpr, color=color, linewidth=2.5, linestyle="-.",
                            label=f"VQC (AUC={auc:.3f})")
                except Exception as e:
                    print(f"  VQC ROC skipped for {disease}: {e}")
                continue
            if not os.path.exists(path):
                continue
            model = joblib.load(path)
            y_score = model.predict_proba(X_test_s)[:,1]
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            ls = "-" if model_name == "rf" else "--"
            ax.plot(fpr, tpr, color=color, linewidth=2.5, linestyle=ls,
                    label=f"{MODEL_LABELS[model_name]} (AUC={auc:.3f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{DISEASE_LABELS[disease]}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_roc_curves.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 2: Model Performance Comparison (grouped bar chart)
# =============================================================================
def fig2_model_comparison():
    print("\n[Fig 2] Model performance comparison...")
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json")) as f:
        report = json.load(f)

    metrics = ["test_auc", "accuracy", "f1"]
    metric_labels = ["Test AUC", "Accuracy", "F1 Score"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Performance Comparison Across Diseases", fontsize=15, fontweight="bold", y=1.02)

    x = np.arange(len(DISEASES))
    bar_w = 0.25
    models = ["rf", "xgb", "vqc"]

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        for i, (m, color) in enumerate(zip(models, MODEL_COLORS.values())):
            vals = [report[d].get(m, {}).get(metric, 0) for d in DISEASES]
            bars = ax.bar(x + (i-1)*bar_w, vals, bar_w, label=MODEL_LABELS[m],
                          color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([DISEASE_LABELS[d] for d in DISEASES], fontweight="bold")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=9)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_model_comparison.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 3: SHAP Global Feature Importance (3-panel)
# =============================================================================
def fig3_shap_importance():
    print("\n[Fig 3] SHAP global feature importance...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SHAP Global Feature Importance (Random Forest)", fontsize=15, fontweight="bold", y=1.02)

    feature_display = {
        "glucose": "Glucose", "hba1c": "HbA1c", "bmi": "BMI", "age": "Age",
        "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP",
        "cholesterol": "Cholesterol", "smoking_encoded": "Smoking",
        "creatinine": "Creatinine", "hemoglobin": "Haemoglobin",
    }

    for ax, disease in zip(axes, DISEASES):
        shap_path = os.path.join(RESULTS_DIR, f"shap_global_{disease}.json")
        if not os.path.exists(shap_path):
            continue
        with open(shap_path) as f:
            data = json.load(f)

        features = [feature_display.get(f, f) for f in data["features"]]
        values = data["mean_abs_shap"]
        sorted_idx = np.argsort(values)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_values = [values[i] for i in sorted_idx]

        colors = [DISEASE_COLORS[disease] if i >= len(sorted_idx)-3 else "#BDC3C7"
                  for i in range(len(sorted_idx))]
        bars = ax.barh(sorted_features, sorted_values, color=colors, edgecolor="white")

        for bar, v in zip(bars, sorted_values):
            if v > 0.005:
                ax.text(v+0.002, bar.get_y()+bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"{DISEASE_LABELS[disease]}", fontweight="bold")
        ax.set_xlim(0, max(sorted_values)*1.25)

        top3_patch = mpatches.Patch(color=DISEASE_COLORS[disease], label="Top 3 features")
        other_patch = mpatches.Patch(color="#BDC3C7", label="Other features")
        ax.legend(handles=[top3_patch, other_patch], fontsize=9, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_shap_importance.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 4: VQC Layer Comparison
# =============================================================================
def fig4_vqc_layers():
    print("\n[Fig 4] VQC layer comparison...")
    with open(os.path.join(RESULTS_DIR, "quantum_justification.json")) as f:
        qj = json.load(f)

    # Use evaluation_report for real classical AUCs
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json")) as f:
        report = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Quantum VQC Architecture Comparison", fontsize=15, fontweight="bold")

    # Left: VQC layer AUC comparison per disease
    ax = axes[0]
    layer_labels = ["2 Layers", "3 Layers", "4 Layers"]
    layer_keys = ["vqc_2layers_auc", "vqc_3layers_auc", "vqc_4layers_auc"]
    x = np.arange(len(layer_labels))
    bar_w = 0.25

    for i, disease in enumerate(["cvd", "ckd"]):  # diabetes has placeholder 0.5
        vals = [qj[disease].get(k, 0) for k in layer_keys]
        bars = ax.bar(x + (i-0.5)*bar_w, vals, bar_w,
                      label=DISEASE_LABELS[disease], color=DISEASE_COLORS[disease],
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(layer_labels, fontweight="bold")
    ax.set_ylabel("Validation AUC"); ax.set_ylim(0, 1.1)
    ax.set_title("VQC Validation AUC by Layer Count", fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1, label="Random baseline")

    # Right: Classical vs VQC test AUC comparison
    ax2 = axes[1]
    model_names = ["RF", "XGBoost", "VQC (4L)"]
    model_keys_map = {"RF": "rf", "XGBoost": "xgb", "VQC (4L)": "vqc"}
    x2 = np.arange(len(model_names))
    bar_w2 = 0.25

    for i, disease in enumerate(DISEASES):
        vals = []
        for mn in model_names:
            mk = model_keys_map[mn]
            vals.append(report[disease].get(mk, {}).get("test_auc", 0))
        bars = ax2.bar(x2 + (i-1)*bar_w2, vals, bar_w2,
                       label=DISEASE_LABELS[disease], color=DISEASE_COLORS[disease],
                       alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x2); ax2.set_xticklabels(model_names, fontweight="bold")
    ax2.set_ylabel("Test AUC"); ax2.set_ylim(0, 1.12)
    ax2.set_title("Classical vs Quantum Test AUC", fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.axhline(0.9, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax2.text(2.4, 0.905, "AUC=0.90", color="red", fontsize=8, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_vqc_comparison.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 5: Robustness Analysis
# =============================================================================
def fig5_robustness():
    print("\n[Fig 5] Robustness analysis...")
    with open(os.path.join(RESULTS_DIR, "robustness_report.json")) as f:
        rob = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Robustness Under Input Perturbations", fontsize=15, fontweight="bold", y=1.02)

    for ax, disease in zip(axes, DISEASES):
        d = rob[disease]
        noise_levels = ["Unperturbed", "±5% Noise", "±10% Noise"]
        mean_changes = [0, d["noise_5pct"]["mean_risk_change"], d["noise_10pct"]["mean_risk_change"]]
        std_changes = [0, d["noise_5pct"]["std_risk_change"], d["noise_10pct"]["std_risk_change"]]
        pct_changed = [0, d["noise_5pct"]["pct_changed_risk_level"], d["noise_10pct"]["pct_changed_risk_level"]]

        x = np.arange(len(noise_levels))
        color = DISEASE_COLORS[disease]

        bars = ax.bar(x, mean_changes, color=color, alpha=0.75, edgecolor="white", width=0.5)
        ax.errorbar(x, mean_changes, yerr=std_changes, fmt="none", color="black",
                    capsize=5, linewidth=1.5, capthick=1.5)

        for bar, v, pct in zip(bars, mean_changes, pct_changed):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        f"{v:.2f} pts\n({pct:.1f}% changed)", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        rob_score = d["robustness_score"]
        ax.set_xticks(x); ax.set_xticklabels(noise_levels, fontsize=10)
        ax.set_ylabel("Mean Absolute Risk Score Change")
        ax.set_title(f"{DISEASE_LABELS[disease]}\nRobustness Score: {rob_score:.4f}", fontweight="bold")
        ax.set_ylim(0, max(mean_changes)*1.6 + 1)

        score_color = "#27AE60" if rob_score > 0.97 else "#F39C12"
        ax.text(0.98, 0.95, f"Score: {rob_score:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, fontweight="bold", color=score_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=score_color, alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_robustness.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 6: Fusion Weights Visualization
# =============================================================================
def fig6_fusion_weights():
    print("\n[Fig 6] Fusion weights...")
    fusion_path = os.path.join(MODELS_DIR, "fusion_weights_v2.json")
    with open(fusion_path) as f:
        fw = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("AUC-Weighted Hybrid Fusion Weights", fontsize=15, fontweight="bold", y=1.02)

    for ax, disease in zip(axes, DISEASES):
        entry = fw.get(disease, {})
        weights = entry.get("weights", entry)
        labels = [MODEL_LABELS.get(k, k) for k in weights.keys()]
        values = list(weights.values())
        colors_pie = [MODEL_COLORS.get(k, "#95A5A6") for k in weights.keys()]

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors_pie,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10)
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")

        ax.set_title(f"{DISEASE_LABELS[disease]}", fontweight="bold", fontsize=13)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_fusion_weights.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 7: System Architecture Diagram
# =============================================================================
def fig7_architecture():
    print("\n[Fig 7] System architecture diagram...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    def box(x, y, w, h, text, color, fontsize=10, text_color="white", style="round,pad=0.1"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle=style,
                               facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                wrap=True, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5), zorder=5)

    # Title
    ax.text(7, 9.6, "QuantumHealthAI v2 Enhanced — System Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")

    # Input
    box(5.5, 8.8, 3, 0.6, "Patient PDF Upload", "#2C3E50", fontsize=10)
    arrow(7, 8.8, 7, 8.2)

    # OCR
    box(5.5, 7.5, 3, 0.6, "OCR Extraction\n(Tesseract / OCRmyPDF)", "#7F8C8D", fontsize=9)
    arrow(7, 7.5, 7, 6.9)

    # Data Governance
    box(4.5, 6.2, 5, 0.6, "Data Governance Module\n(Validate · Convert Units · IQR Clip · Impute)", "#8E44AD", fontsize=9)
    arrow(7, 6.2, 7, 5.6)

    # Feature Mapper
    box(4.5, 4.9, 5, 0.6, "Feature Mapper\n(Disease-Specific Subsets: 6 / 6 / 5 features)", "#1ABC9C", fontsize=9)

    # Three branches
    arrow(5.5, 4.9, 2.5, 4.3)
    arrow(7, 4.9, 7, 4.3)
    arrow(8.5, 4.9, 11.5, 4.3)

    # RF branch
    box(1, 3.6, 3, 0.6, "StandardScaler\n→ Random Forest", "#4C72B0", fontsize=9)
    # XGB branch
    box(5.5, 3.6, 3, 0.6, "StandardScaler\n→ XGBoost (GPU)", "#DD8452", fontsize=9)
    # VQC branch
    box(10, 3.6, 3, 0.6, "Scaler → PCA\n→ VQC Circuit", "#55A868", fontsize=9)

    arrow(2.5, 3.6, 5.5, 3.0)
    arrow(7, 3.6, 7, 3.0)
    arrow(11.5, 3.6, 8.5, 3.0)

    # Hybrid Fusion
    box(4.5, 2.3, 5, 0.6, "AUC-Weighted Hybrid Fusion\nrisk_score = Σ(weight_i × prob_i × 100)", "#E74C3C", fontsize=9)
    arrow(7, 2.3, 7, 1.7)

    # Clinical Rules
    box(4.5, 1.0, 5, 0.6, "Clinical Rule Engine  →  SHAP Explainability", "#F39C12", fontsize=9)
    arrow(7, 1.0, 7, 0.4)

    # Output
    box(4.5, -0.2, 5, 0.55, "Structured JSON Response\n(risk_score · risk_level · explanation · shap_features)", "#2C3E50", fontsize=9)

    # Labels on branches
    ax.text(2.5, 4.55, "RF prob", ha="center", fontsize=8, color="#4C72B0", style="italic")
    ax.text(7, 4.55, "XGB prob", ha="center", fontsize=8, color="#DD8452", style="italic")
    ax.text(11.5, 4.55, "VQC prob", ha="center", fontsize=8, color="#55A868", style="italic")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_architecture.png")
    plt.savefig(path, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 8: VQC Circuit Diagram
# =============================================================================
def fig8_vqc_circuit():
    print("\n[Fig 8] VQC circuit diagram...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7, 6.2, "Variational Quantum Circuit (VQC) — 6-Qubit Architecture",
            ha="center", fontsize=14, fontweight="bold", color="#2C3E50")
    ax.text(7, 5.85, "AngleEmbedding (RY) → StronglyEntanglingLayers × 4 → PauliZ Measurement",
            ha="center", fontsize=11, color="#7F8C8D")

    n_qubits = 6
    qubit_labels = ["q₀ (PC1)", "q₁ (PC2)", "q₂ (PC3)", "q₃ (PC4)", "q₄ (PC5)", "q₅ (PC6)"]
    y_positions = [5.2 - i*0.85 for i in range(n_qubits)]

    # Wire lines
    for i, y in enumerate(y_positions):
        ax.plot([0.8, 13.2], [y, y], color="#BDC3C7", linewidth=1.5, zorder=1)
        ax.text(0.5, y, qubit_labels[i], ha="right", va="center", fontsize=10,
                fontweight="bold", color="#2C3E50")

    # Input state labels
    for i, y in enumerate(y_positions):
        ax.text(0.85, y+0.15, "|0⟩", ha="left", fontsize=9, color="#7F8C8D")

    # AngleEmbedding gates (RY)
    for i, y in enumerate(y_positions):
        rect = FancyBboxPatch((1.5, y-0.22), 1.2, 0.44, boxstyle="round,pad=0.05",
                               facecolor="#3498DB", edgecolor="white", linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(2.1, y, f"RY(x{i})", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=4)

    ax.text(2.1, y_positions[-1]-0.5, "AngleEmbedding", ha="center", fontsize=9,
            color="#3498DB", fontweight="bold")

    # StronglyEntanglingLayers (4 layers shown as blocks)
    layer_colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71"]
    layer_x = [4.0, 6.2, 8.4, 10.6]

    for li, (lx, lc) in enumerate(zip(layer_x, layer_colors)):
        # Layer block
        rect = FancyBboxPatch((lx, y_positions[-1]-0.35), 1.8, y_positions[0]-y_positions[-1]+0.7,
                               boxstyle="round,pad=0.05", facecolor=lc, edgecolor="white",
                               linewidth=1.5, alpha=0.15, zorder=2)
        ax.add_patch(rect)

        # Rotation gates
        for i, y in enumerate(y_positions):
            rect2 = FancyBboxPatch((lx+0.1, y-0.18), 0.7, 0.36, boxstyle="round,pad=0.03",
                                    facecolor=lc, edgecolor="white", linewidth=1, zorder=3)
            ax.add_patch(rect2)
            ax.text(lx+0.45, y, "Rot", ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=4)

        # CNOT entanglement lines
        for i in range(n_qubits-1):
            y1, y2 = y_positions[i], y_positions[i+1]
            ax.plot([lx+1.1, lx+1.1], [y1, y2], color=lc, linewidth=1.5, alpha=0.7, zorder=2)
            ax.plot(lx+1.1, y2, "o", color=lc, markersize=8, zorder=3)
            ax.plot(lx+1.1, y1, "+", color="white", markersize=10, markeredgewidth=2, zorder=4)

        ax.text(lx+0.9, y_positions[-1]-0.5, f"Layer {li+1}", ha="center", fontsize=9,
                color=lc, fontweight="bold")

    # Measurement
    for i, y in enumerate(y_positions):
        rect = FancyBboxPatch((12.5, y-0.22), 0.6, 0.44, boxstyle="round,pad=0.05",
                               facecolor="#2C3E50", edgecolor="white", linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(12.8, y, "M", ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=4)

    ax.text(12.8, y_positions[-1]-0.5, "⟨Z₀⟩", ha="center", fontsize=10,
            color="#2C3E50", fontweight="bold")

    # Output arrow
    ax.annotate("", xy=(13.5, y_positions[0]), xytext=(13.1, y_positions[0]),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2))
    ax.text(13.6, y_positions[0], "prob", ha="left", va="center", fontsize=10,
            fontweight="bold", color="#2C3E50")

    # Legend
    legend_items = [
        mpatches.Patch(color="#3498DB", label="AngleEmbedding (RY rotations)"),
        mpatches.Patch(color="#E74C3C", alpha=0.6, label="StronglyEntanglingLayers (×4)"),
        mpatches.Patch(color="#2C3E50", label="PauliZ Measurement"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=3, fontsize=10,
              bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig8_vqc_circuit.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 9: Cross-Validation Stability (AUC distribution)
# =============================================================================
def fig9_cv_stability():
    print("\n[Fig 9] CV stability...")
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json")) as f:
        report = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("5-Fold Cross-Validation AUC Stability", fontsize=15, fontweight="bold")

    x = np.arange(len(DISEASES))
    bar_w = 0.3
    models_cv = ["rf", "xgb"]

    for i, (m, color) in enumerate(zip(models_cv, [MODEL_COLORS["rf"], MODEL_COLORS["xgb"]])):
        means = [report[d][m]["mean_auc"] for d in DISEASES]
        stds = [report[d][m]["std_auc"] for d in DISEASES]
        bars = ax.bar(x + (i-0.5)*bar_w, means, bar_w, yerr=stds,
                      label=MODEL_LABELS[m], color=color, alpha=0.85,
                      capsize=6, error_kw=dict(elinewidth=2, capthick=2),
                      edgecolor="white")
        for bar, m_val, s_val in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s_val+0.005,
                    f"{m_val:.4f}\n±{s_val:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DISEASE_LABELS[d] for d in DISEASES], fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean AUC (5-fold CV)", fontsize=12)
    ax.set_ylim(0.7, 1.08)
    ax.legend(fontsize=11)
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.4, linewidth=1.5)
    ax.text(2.45, 0.905, "AUC = 0.90 threshold", color="red", fontsize=9, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig9_cv_stability.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 10: Summary Dashboard (research poster style)
# =============================================================================
def fig10_summary_dashboard():
    print("\n[Fig 10] Summary dashboard...")
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json")) as f:
        report = json.load(f)
    with open(os.path.join(RESULTS_DIR, "robustness_report.json")) as f:
        rob = json.load(f)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#F0F4F8")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Title
    fig.text(0.5, 0.97, "QuantumHealthAI v2 Enhanced — Research Summary Dashboard",
             ha="center", fontsize=16, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.945, "Hybrid Quantum-Classical Disease Risk Prediction System",
             ha="center", fontsize=12, color="#7F8C8D")

    # ── Row 1: Key metrics per disease ───────────────────────────────────────
    for col, disease in enumerate(DISEASES):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("white")
        models_show = ["rf", "xgb", "vqc"]
        aucs = [report[disease][m]["test_auc"] for m in models_show]
        colors_bar = [MODEL_COLORS[m] for m in models_show]
        bars = ax.bar(["RF", "XGB", "VQC"], aucs, color=colors_bar, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0.5, 1.1)
        ax.set_ylabel("Test AUC")
        ax.set_title(f"{DISEASE_LABELS[disease]}\nTest AUC", fontweight="bold", fontsize=11)
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.3, linewidth=1)

    # ── Row 1 col 4: Robustness scores ───────────────────────────────────────
    ax_rob = fig.add_subplot(gs[0, 3])
    ax_rob.set_facecolor("white")
    rob_scores = [rob[d]["robustness_score"] for d in DISEASES]
    colors_rob = [DISEASE_COLORS[d] for d in DISEASES]
    bars = ax_rob.bar([DISEASE_LABELS[d] for d in DISEASES], rob_scores,
                       color=colors_rob, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, rob_scores):
        ax_rob.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_rob.set_ylim(0.95, 1.005)
    ax_rob.set_ylabel("Robustness Score")
    ax_rob.set_title("Model Robustness\n(±5% noise)", fontweight="bold", fontsize=11)

    # ── Row 2: SHAP importance for all diseases ───────────────────────────────
    feature_display = {
        "glucose": "Glucose", "hba1c": "HbA1c", "bmi": "BMI", "age": "Age",
        "systolic_bp": "Sys. BP", "diastolic_bp": "Dia. BP",
        "cholesterol": "Cholesterol", "smoking_encoded": "Smoking",
        "creatinine": "Creatinine", "hemoglobin": "Haemoglobin",
    }
    for col, disease in enumerate(DISEASES):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("white")
        shap_path = os.path.join(RESULTS_DIR, f"shap_global_{disease}.json")
        if os.path.exists(shap_path):
            with open(shap_path) as f:
                data = json.load(f)
            features = [feature_display.get(ft, ft) for ft in data["features"]]
            values = data["mean_abs_shap"]
            sorted_idx = np.argsort(values)
            sf = [features[i] for i in sorted_idx]
            sv = [values[i] for i in sorted_idx]
            colors_shap = [DISEASE_COLORS[disease] if i >= len(sorted_idx)-3 else "#BDC3C7"
                           for i in range(len(sorted_idx))]
            ax.barh(sf, sv, color=colors_shap, edgecolor="white")
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title(f"SHAP — {DISEASE_LABELS[disease]}", fontweight="bold", fontsize=11)

    # ── Row 2 col 4: Fusion weights radar-style ───────────────────────────────
    ax_fw = fig.add_subplot(gs[1, 3])
    ax_fw.set_facecolor("white")
    fusion_path = os.path.join(MODELS_DIR, "fusion_weights_v2.json")
    with open(fusion_path) as f:
        fw = json.load(f)
    x_fw = np.arange(3)
    bar_w_fw = 0.25
    for i, disease in enumerate(DISEASES):
        entry = fw.get(disease, {})
        weights = entry.get("weights", entry)
        vals = list(weights.values())[:3]
        ax_fw.bar(x_fw + (i-1)*bar_w_fw, vals, bar_w_fw,
                  label=DISEASE_LABELS[disease], color=DISEASE_COLORS[disease], alpha=0.8)
    ax_fw.set_xticks(x_fw)
    ax_fw.set_xticklabels(["RF", "XGB", "VQC"], fontweight="bold")
    ax_fw.set_ylabel("Fusion Weight")
    ax_fw.set_title("Fusion Weights\nper Disease", fontweight="bold", fontsize=11)
    ax_fw.legend(fontsize=8)

    # ── Row 3: Summary stats table ────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    ax_table.set_facecolor("white")

    table_data = []
    col_labels = ["Disease", "RF AUC", "XGB AUC", "VQC AUC",
                  "Best Model", "CV AUC (mean±std)", "F1 (XGB)", "Robustness"]
    for disease in DISEASES:
        r = report[disease]
        best = max(["rf","xgb","vqc"], key=lambda m: r[m]["test_auc"])
        table_data.append([
            DISEASE_LABELS[disease],
            f"{r['rf']['test_auc']:.4f}",
            f"{r['xgb']['test_auc']:.4f}",
            f"{r['vqc']['test_auc']:.4f}",
            MODEL_LABELS[best],
            f"{r['xgb']['mean_auc']:.4f} ± {r['xgb']['std_auc']:.4f}",
            f"{r['xgb']['f1']:.4f}",
            f"{rob[disease]['robustness_score']:.4f}",
        ])

    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                            loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, disease in enumerate(DISEASES):
        table[i+1, 0].set_facecolor(DISEASE_COLORS[disease])
        table[i+1, 0].set_text_props(color="white", fontweight="bold")

    ax_table.set_title("Complete Performance Summary", fontweight="bold", fontsize=12, pad=15)

    path = os.path.join(OUT_DIR, "fig10_summary_dashboard.png")
    plt.savefig(path, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    fig1_roc_curves()
    fig2_model_comparison()
    fig3_shap_importance()
    fig4_vqc_layers()
    fig5_robustness()
    fig6_fusion_weights()
    fig7_architecture()
    fig8_vqc_circuit()
    fig9_cv_stability()
    fig10_summary_dashboard()

    print(f"\n{'='*60}")
    print("All figures generated successfully!")
    print(f"{'='*60}")
    print(f"\nFigures saved to: {OUT_DIR}")
    print("\nFigure list:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
            print(f"  {f}  ({size} KB)")
