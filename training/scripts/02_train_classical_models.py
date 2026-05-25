"""
Script 02: Train classical RF + XGBoost models for all three diseases.

Trains 6 models total:
    - rf_diabetes.joblib, xgb_diabetes.joblib
    - rf_cvd.joblib, xgb_cvd.joblib
    - rf_ckd.joblib, xgb_ckd.joblib

Saves models to: ../data/models/
Saves scalers to: training/models/feature_scaler.joblib

Performance targets:
    - Diabetes AUC ≥ 0.85
    - CVD AUC ≥ 0.80
    - CKD AUC ≥ 0.90
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level"
]


def load_dataset(disease: str) -> tuple:
    """Load aligned dataset and split into train/test."""
    path = os.path.join(PROCESSED_DIR, f"{disease}_aligned.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 01_preprocess_datasets.py first.")

    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Class balance (train): {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test


def train_rf(X_train, y_train, disease: str):
    """Train Random Forest with grid search."""
    print(f"  Training Random Forest for {disease}...")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", None],
    }
    rf = RandomForestClassifier(random_state=SEED)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"    Best params: {grid.best_params_}")
    print(f"    Best CV AUC: {grid.best_score_:.4f}")

    return grid.best_estimator_


def train_xgb(X_train, y_train, disease: str):
    """Train XGBoost with grid search."""
    print(f"  Training XGBoost for {disease}...")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "scale_pos_weight": [1, 2, 5],  # handle class imbalance
    }
    xgb = XGBClassifier(random_state=SEED, eval_metric="logloss", use_label_encoder=False)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"    Best params: {grid.best_params_}")
    print(f"    Best CV AUC: {grid.best_score_:.4f}")

    return grid.best_estimator_


def evaluate(model, X_test, y_test, model_name: str):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  {model_name} Test Performance:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def train_disease(disease: str):
    """Train RF + XGBoost for a single disease."""
    print(f"\n{'='*70}")
    print(f"Training models for: {disease.upper()}")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test = load_dataset(disease)

    # Train RF
    rf = train_rf(X_train, y_train, disease)
    rf_metrics = evaluate(rf, X_test, y_test, "Random Forest")

    # Train XGBoost
    xgb = train_xgb(X_train, y_train, disease)
    xgb_metrics = evaluate(xgb, X_test, y_test, "XGBoost")

    # Save models
    rf_path = os.path.join(MODELS_DIR, f"rf_{disease}.joblib")
    xgb_path = os.path.join(MODELS_DIR, f"xgb_{disease}.joblib")
    joblib.dump(rf, rf_path)
    joblib.dump(xgb, xgb_path)
    print(f"\n  [SAVED] {rf_path}")
    print(f"  [SAVED] {xgb_path}")

    return rf_metrics, xgb_metrics


if __name__ == "__main__":
    print("=" * 70)
    print("QuantumHealthAI — Classical Model Training")
    print("=" * 70)

    results = {}

    try:
        rf_d, xgb_d = train_disease("diabetes")
        results["diabetes"] = {"rf": rf_d, "xgb": xgb_d}
    except Exception as e:
        print(f"[FAIL] Diabetes training failed: {e}")

    try:
        rf_c, xgb_c = train_disease("cvd")
        results["cvd"] = {"rf": rf_c, "xgb": xgb_c}
    except Exception as e:
        print(f"[FAIL] CVD training failed: {e}")

    try:
        rf_k, xgb_k = train_disease("ckd")
        results["ckd"] = {"rf": rf_k, "xgb": xgb_k}
    except Exception as e:
        print(f"[FAIL] CKD training failed: {e}")

    print("\n" + "=" * 70)
    print("Classical Training Complete")
    print("=" * 70)
    print("\nSummary:")
    for disease, metrics in results.items():
        print(f"\n{disease.upper()}:")
        print(f"  RF AUC:  {metrics['rf']['auc']:.4f}")
        print(f"  XGB AUC: {metrics['xgb']['auc']:.4f}")

    print("\nNext step: python training/scripts/03_train_quantum_vqc.py")
