"""
Reproducibility Verification Script (Task 24.2)

Verifies that training with seed=42 produces reproducible results by:
1. Checking random_state attributes on existing trained models
2. Re-training a fresh RF + XGBoost model for diabetes with seed=42
3. Comparing feature_importances_ arrays between original and re-trained models
4. Verifying scaler mean_ and scale_ arrays are identical between runs

Requirements: 23.5
"""
import os
import sys
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
TRAINING_MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_MODELS_DIR = os.path.join(WORKSPACE_ROOT, "data", "models")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "processed")

SEED = 42
DISEASE = "diabetes"
FEATURES = ["glucose", "hba1c", "bmi", "age", "systolic_bp", "diastolic_bp"]

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = {}


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: random_state attributes on existing models
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Reproducibility Verification — QuantumHealthAI v2")
print("=" * 70)
print()
print("CHECK 1: Verify random_state=42 on existing trained models")
print("-" * 70)

for disease in ["diabetes", "cvd", "ckd"]:
    rf_path = os.path.join(DATA_MODELS_DIR, f"rf_{disease}.joblib")
    xgb_path = os.path.join(DATA_MODELS_DIR, f"xgb_{disease}.joblib")

    if not os.path.exists(rf_path):
        print(f"  [{disease.upper()}] RF model not found at {rf_path}")
        results[f"rf_{disease}_random_state"] = False
        continue
    if not os.path.exists(xgb_path):
        print(f"  [{disease.upper()}] XGB model not found at {xgb_path}")
        results[f"xgb_{disease}_random_state"] = False
        continue

    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)

    rf_rs = getattr(rf, "random_state", None)
    xgb_rs = xgb.get_params().get("random_state", None)

    rf_ok = rf_rs == SEED
    xgb_ok = xgb_rs == SEED

    results[f"rf_{disease}_random_state"] = rf_ok
    results[f"xgb_{disease}_random_state"] = xgb_ok

    print(f"  [{disease.upper()}] RF  random_state={rf_rs}  → {PASS if rf_ok else FAIL}")
    print(f"  [{disease.upper()}] XGB random_state={xgb_rs} → {PASS if xgb_ok else FAIL}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Scaler attributes (mean_ and scale_ exist, confirming fitted)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("CHECK 2: Verify scalers are fitted (have mean_ and scale_ attributes)")
print("-" * 70)

for disease in ["diabetes", "cvd", "ckd"]:
    scaler_path = os.path.join(TRAINING_MODELS_DIR, f"scaler_{disease}.joblib")
    if not os.path.exists(scaler_path):
        print(f"  [{disease.upper()}] Scaler not found at {scaler_path}")
        results[f"scaler_{disease}_fitted"] = False
        continue

    scaler = joblib.load(scaler_path)
    has_mean = hasattr(scaler, "mean_") and scaler.mean_ is not None
    has_scale = hasattr(scaler, "scale_") and scaler.scale_ is not None
    ok = has_mean and has_scale
    results[f"scaler_{disease}_fitted"] = ok
    print(f"  [{disease.upper()}] mean_={scaler.mean_.round(4) if has_mean else 'MISSING'}, "
          f"scale_={scaler.scale_.round(4) if has_scale else 'MISSING'} → {PASS if ok else FAIL}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Re-train diabetes models with seed=42 and compare feature_importances_
# ─────────────────────────────────────────────────────────────────────────────

print()
print("CHECK 3: Re-train diabetes models with seed=42 and compare feature_importances_")
print("-" * 70)

# Load original models
orig_rf_path = os.path.join(DATA_MODELS_DIR, f"rf_{DISEASE}.joblib")
orig_xgb_path = os.path.join(DATA_MODELS_DIR, f"xgb_{DISEASE}.joblib")
orig_scaler_path = os.path.join(TRAINING_MODELS_DIR, f"scaler_{DISEASE}.joblib")

if not all(os.path.exists(p) for p in [orig_rf_path, orig_xgb_path, orig_scaler_path]):
    print("  [SKIP] Original diabetes model artifacts not found.")
    results["diabetes_feature_importances_rf"] = False
    results["diabetes_feature_importances_xgb"] = False
    results["diabetes_scaler_mean_identical"] = False
    results["diabetes_scaler_scale_identical"] = False
else:
    orig_rf = joblib.load(orig_rf_path)
    orig_xgb = joblib.load(orig_xgb_path)
    orig_scaler = joblib.load(orig_scaler_path)

    # Load dataset
    data_path = os.path.join(PROCESSED_DIR, f"{DISEASE}_aligned.csv")
    df = pd.read_csv(data_path)
    X = df[FEATURES].values.astype(np.float64)
    y = df["target"].values.astype(int)

    # Reproduce the exact same split as script 02
    np.random.seed(SEED)
    import random
    random.seed(SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Re-fit scaler on same training split
    new_scaler = StandardScaler()
    X_train_s = new_scaler.fit_transform(X_train)

    # Re-train RF with same params
    orig_rf_params = orig_rf.get_params()
    new_rf = RandomForestClassifier(
        n_estimators=orig_rf_params.get("n_estimators", 500),
        max_depth=orig_rf_params.get("max_depth", None),
        min_samples_split=orig_rf_params.get("min_samples_split", 2),
        min_samples_leaf=orig_rf_params.get("min_samples_leaf", 1),
        max_features=orig_rf_params.get("max_features", "sqrt"),
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    new_rf.fit(X_train_s, y_train)

    # Re-train XGBoost with same params
    orig_xgb_params = orig_xgb.get_params()
    new_xgb = XGBClassifier(
        n_estimators=orig_xgb_params.get("n_estimators", 500),
        max_depth=orig_xgb_params.get("max_depth", 6),
        learning_rate=orig_xgb_params.get("learning_rate", 0.05),
        subsample=orig_xgb_params.get("subsample", 0.8),
        colsample_bytree=orig_xgb_params.get("colsample_bytree", 0.8),
        gamma=orig_xgb_params.get("gamma", 0),
        reg_alpha=orig_xgb_params.get("reg_alpha", 0),
        reg_lambda=orig_xgb_params.get("reg_lambda", 1),
        min_child_weight=orig_xgb_params.get("min_child_weight", 1),
        scale_pos_weight=orig_xgb_params.get("scale_pos_weight", 1),
        random_state=SEED,
        eval_metric="auc",
        verbosity=0,
        tree_method="hist",
    )
    new_xgb.fit(X_train_s, y_train)

    # Compare feature_importances_
    orig_rf_fi = orig_rf.feature_importances_
    new_rf_fi = new_rf.feature_importances_

    orig_xgb_fi = orig_xgb.feature_importances_
    new_xgb_fi = new_xgb.feature_importances_

    rf_fi_close = np.allclose(orig_rf_fi, new_rf_fi, atol=1e-6)
    xgb_fi_close = np.allclose(orig_xgb_fi, new_xgb_fi, atol=1e-6)

    results["diabetes_feature_importances_rf"] = rf_fi_close
    results["diabetes_feature_importances_xgb"] = xgb_fi_close

    print(f"  RF  feature_importances_ identical (atol=1e-6): {PASS if rf_fi_close else FAIL}")
    if not rf_fi_close:
        max_diff = np.max(np.abs(orig_rf_fi - new_rf_fi))
        print(f"       Max absolute difference: {max_diff:.2e}")
        print(f"       Original: {orig_rf_fi.round(6)}")
        print(f"       Re-trained: {new_rf_fi.round(6)}")

    print(f"  XGB feature_importances_ identical (atol=1e-6): {PASS if xgb_fi_close else FAIL}")
    if not xgb_fi_close:
        max_diff = np.max(np.abs(orig_xgb_fi - new_xgb_fi))
        print(f"       Max absolute difference: {max_diff:.2e}")
        print(f"       Original: {orig_xgb_fi.round(6)}")
        print(f"       Re-trained: {new_xgb_fi.round(6)}")

    # Compare scaler mean_ and scale_
    scaler_mean_close = np.allclose(orig_scaler.mean_, new_scaler.mean_, atol=1e-10)
    scaler_scale_close = np.allclose(orig_scaler.scale_, new_scaler.scale_, atol=1e-10)

    results["diabetes_scaler_mean_identical"] = scaler_mean_close
    results["diabetes_scaler_scale_identical"] = scaler_scale_close

    print(f"  Scaler mean_  identical (atol=1e-10): {PASS if scaler_mean_close else FAIL}")
    if not scaler_mean_close:
        print(f"       Original:   {orig_scaler.mean_.round(6)}")
        print(f"       Re-trained: {new_scaler.mean_.round(6)}")

    print(f"  Scaler scale_ identical (atol=1e-10): {PASS if scaler_scale_close else FAIL}")
    if not scaler_scale_close:
        print(f"       Original:   {orig_scaler.scale_.round(6)}")
        print(f"       Re-trained: {new_scaler.scale_.round(6)}")

    # Also compare predictions on test set
    X_test_s_orig = orig_scaler.transform(X_test)
    X_test_s_new = new_scaler.transform(X_test)

    orig_rf_preds = orig_rf.predict_proba(X_test_s_orig)[:, 1]
    new_rf_preds = new_rf.predict_proba(X_test_s_new)[:, 1]
    rf_preds_close = np.allclose(orig_rf_preds, new_rf_preds, atol=1e-6)

    orig_xgb_preds = orig_xgb.predict_proba(X_test_s_orig)[:, 1]
    new_xgb_preds = new_xgb.predict_proba(X_test_s_new)[:, 1]
    xgb_preds_close = np.allclose(orig_xgb_preds, new_xgb_preds, atol=1e-6)

    results["diabetes_rf_predictions_identical"] = rf_preds_close
    results["diabetes_xgb_predictions_identical"] = xgb_preds_close

    print()
    print("  Test-set prediction comparison (n={}):".format(len(X_test)))
    print(f"  RF  predictions identical (atol=1e-6): {PASS if rf_preds_close else FAIL}")
    if not rf_preds_close:
        max_diff = np.max(np.abs(orig_rf_preds - new_rf_preds))
        print(f"       Max absolute difference: {max_diff:.2e}")

    print(f"  XGB predictions identical (atol=1e-6): {PASS if xgb_preds_close else FAIL}")
    if not xgb_preds_close:
        max_diff = np.max(np.abs(orig_xgb_preds - new_xgb_preds))
        print(f"       Max absolute difference: {max_diff:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: VQC weights files exist (quantum reproducibility via seed)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("CHECK 4: Verify VQC weight files exist (quantum seed=42 reproducibility)")
print("-" * 70)

for disease in ["diabetes", "cvd", "ckd"]:
    weights_path = os.path.join(TRAINING_MODELS_DIR, f"vqc_{disease}_weights.npy")
    if os.path.exists(weights_path):
        weights = np.load(weights_path)
        ok = weights.ndim >= 2 and weights.size > 0
        results[f"vqc_{disease}_weights_exist"] = ok
        print(f"  [{disease.upper()}] VQC weights shape={weights.shape}, "
              f"non-zero={np.any(weights != 0)} → {PASS if ok else FAIL}")
    else:
        results[f"vqc_{disease}_weights_exist"] = False
        print(f"  [{disease.upper()}] VQC weights not found → {FAIL}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("REPRODUCIBILITY VERIFICATION SUMMARY")
print("=" * 70)

all_pass = True
for check_name, passed in results.items():
    status = PASS if passed else FAIL
    print(f"  {status}  {check_name}")
    if not passed:
        all_pass = False

print()
if all_pass:
    print("✅ ALL CHECKS PASSED — Training is reproducible with seed=42")
else:
    failed = [k for k, v in results.items() if not v]
    print(f"❌ {len(failed)} CHECK(S) FAILED: {', '.join(failed)}")
    sys.exit(1)
