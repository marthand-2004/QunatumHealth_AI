# QuantumHealthAI — Model Training Pipeline

This directory contains all training scripts, datasets, and trained model artifacts for the QuantumHealthAI disease risk prediction system.

## Overview

The system uses a **hybrid classical-quantum ML architecture**:
- **Classical**: Random Forest + XGBoost ensemble
- **Quantum**: PennyLane Variational Quantum Classifier (VQC)
- **Hybrid**: Weighted fusion (60% classical + 40% quantum)

## Target Diseases

1. **Diabetes** (Type 2)
2. **Cardiovascular Disease (CVD)**
3. **Chronic Kidney Disease (CKD)**

## Feature Vector (14 dimensions)

| # | Feature | Source | Unit |
|---|---|---|---|
| 1 | glucose | Lab | mmol/L |
| 2 | hba1c | Lab | % |
| 3 | creatinine | Lab | µmol/L |
| 4 | cholesterol | Lab | mmol/L |
| 5 | triglycerides | Lab | mmol/L |
| 6 | hemoglobin | Lab | g/dL |
| 7 | bmi | Lifestyle | kg/m² |
| 8 | age | Lifestyle | years |
| 9 | systolic_bp | Lab | mmHg |
| 10 | diastolic_bp | Lab | mmHg |
| 11 | smoking_encoded | Lifestyle | 0/0.5/1 |
| 12 | exercise_frequency | Lifestyle | days/week |
| 13 | sleep_hours | Lifestyle | hours |
| 14 | stress_level | Lifestyle | 1-10 |

## Datasets

### 1. Pima Indians Diabetes Dataset
- **Source**: UCI Machine Learning Repository / Kaggle
- **URL**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Samples**: 768
- **Features**: 8 (glucose, BMI, age, blood pressure, insulin, etc.)
- **Target**: Binary (0=no diabetes, 1=diabetes)

### 2. Framingham Heart Study
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
- **Samples**: ~4,000
- **Features**: 15 (cholesterol, BP, smoking, age, etc.)
- **Target**: Binary (0=no CVD, 1=CVD risk)

### 3. Chronic Kidney Disease Dataset
- **Source**: UCI
- **URL**: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- **Samples**: 400
- **Features**: 24 (creatinine, hemoglobin, albumin, etc.)
- **Target**: Binary (ckd/notckd)

## Directory Structure

```
training/
├── README.md                    # This file
├── datasets/                    # Raw downloaded datasets
│   ├── diabetes_pima.csv
│   ├── framingham_cvd.csv
│   └── ckd_uci.csv
├── notebooks/                   # Jupyter notebooks for EDA
│   ├── 01_diabetes_eda.ipynb
│   ├── 02_cvd_eda.ipynb
│   └── 03_ckd_eda.ipynb
├── scripts/                     # Training scripts
│   ├── 01_preprocess_datasets.py
│   ├── 02_train_classical_models.py
│   ├── 03_train_quantum_vqc.py
│   ├── 04_hybrid_fusion.py
│   ├── 05_evaluate_all_models.py
│   └── 06_demo_inference.py
├── models/                      # Trained model artifacts
│   ├── rf_diabetes.joblib
│   ├── rf_cvd.joblib
│   ├── rf_ckd.joblib
│   ├── xgb_diabetes.joblib
│   ├── xgb_cvd.joblib
│   ├── xgb_ckd.joblib
│   ├── vqc_diabetes_weights.npy
│   ├── vqc_cvd_weights.npy
│   ├── vqc_ckd_weights.npy
│   ├── pca_scaler.joblib
│   └── feature_scaler.joblib
├── results/                     # Evaluation outputs
│   ├── performance_table.csv
│   ├── confusion_matrices/
│   ├── shap_plots/
│   ├── training_curves/
│   └── calibration_curves/
└── requirements_training.txt    # Additional training dependencies
```

## Setup Instructions

### 1. Install System Dependencies

**Tesseract OCR** (required for OCRmyPDF):
```bash
# Windows (via Chocolatey)
choco install tesseract

# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Ghostscript** (required for PDF processing):
```bash
# Windows
choco install ghostscript

# Or download from: https://www.ghostscript.com/download/gsdnld.html
```

**Poppler** (optional, for pdfplumber):
```bash
# Windows
choco install poppler

# Or download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -r training/requirements_training.txt
```

### 3. Download Datasets

```bash
cd training
python scripts/00_download_datasets.py
```

Or manually download from the URLs above and place in `training/datasets/`.

### 4. Run Training Pipeline

```bash
# Step 1: Preprocess and align datasets to 14-feature schema
python training/scripts/01_preprocess_datasets.py

# Step 2: Train classical models (RF + XGBoost)
python training/scripts/02_train_classical_models.py

# Step 3: Train quantum VQC circuits
python training/scripts/03_train_quantum_vqc.py

# Step 4: Optimize hybrid fusion weights
python training/scripts/04_hybrid_fusion.py

# Step 5: Generate evaluation report
python training/scripts/05_evaluate_all_models.py

# Step 6: Run demo inference
python training/scripts/06_demo_inference.py
```

## Performance Targets

| Disease | Classical AUC | Quantum AUC | Hybrid AUC |
|---|---|---|---|
| Diabetes | ≥ 0.85 | ≥ 0.80 | ≥ 0.87 |
| CVD | ≥ 0.80 | ≥ 0.75 | ≥ 0.82 |
| CKD | ≥ 0.90 | ≥ 0.85 | ≥ 0.92 |

## Quantum Circuit Architecture

```
Input: 14D feature vector
    ↓
PCA(n_components=6) — dimensionality reduction
    ↓
6-qubit circuit (default.qubit simulator)
    ↓
AngleEmbedding(features, wires=range(6), rotation='Y')
    ↓
StronglyEntanglingLayers(weights, wires=range(6)) × 3 layers
    ↓
qml.expval(qml.PauliZ(0))
    ↓
Sigmoid activation → [0, 1] probability
    ↓
× 100 → Risk Score [0, 100]
```

**Trainable parameters**: 3 layers × 6 qubits × 3 params = 54 parameters per disease

## Training Configuration

- **Optimizer**: ADAM (lr=0.01 for classical, lr=0.05 for quantum)
- **Epochs**: 100 (classical), 80 (quantum)
- **Batch size**: 32
- **Train/test split**: 80/20, stratified
- **Cross-validation**: 5-fold for hyperparameter tuning
- **Random seed**: 42 (all splits, all models)

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- AUC-ROC, AUC-PR
- Confusion matrix
- Calibration curve (reliability diagram)
- SHAP feature importance (top 10)
- Inference latency (mean ± std, n=100)

## Notes

- Classical models are trained first and used as baselines
- Quantum models are trained on the same train/test splits
- Hybrid fusion weights are optimized on a validation set (10% of training data)
- All models are evaluated on the same held-out test set
- SHAP values are computed using TreeExplainer for classical, KernelExplainer for quantum
