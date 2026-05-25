"""
Script 01: Preprocess and align datasets to 14-feature schema.

Reads raw datasets and produces:
    - training/processed/diabetes_aligned.csv
    - training/processed/cvd_aligned.csv
    - training/processed/ckd_aligned.csv

Each output has exactly 14 features + 1 target column.

v2 changes:
    - Stratified 80/20 train-test split performed FIRST (before any scaling/PCA)
    - Training-set medians computed (not means) for imputation
    - IQR fence values (Q1 − 1.5×IQR, Q3 + 1.5×IQR) computed on training split only
    - Medians and IQR fences saved to training/models/data_governance.json
    - Class distribution logged before and after any balancing
    - Fixed random seeds: numpy.random.seed(42), random.seed(42)
"""
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(SCRIPT_DIR, "..", "datasets")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "processed")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Target 14-feature schema ──────────────────────────────────────────────────
FEATURE_NAMES = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level",
]

# Disease-specific feature subsets (v2)
DISEASE_FEATURE_SUBSETS = {
    "diabetes": ["glucose", "hba1c", "bmi", "age", "systolic_bp", "diastolic_bp"],
    "cvd":      ["age", "cholesterol", "systolic_bp", "diastolic_bp", "smoking_encoded", "bmi"],
    "ckd":      ["creatinine", "hemoglobin", "systolic_bp", "diastolic_bp", "age"],
}

# Acceptable input ranges (pre-unit-conversion, in original units)
ACCEPTABLE_RANGES = {
    "glucose":      (50, 500),    # mg/dL
    "creatinine":   (0.5, 15),    # mg/dL
    "systolic_bp":  (80, 250),    # mmHg
    "diastolic_bp": (50, 150),    # mmHg
    "hemoglobin":   (3, 20),      # g/dL
    "cholesterol":  (50, 400),    # mg/dL
    "hba1c":        (3, 15),      # %
    "bmi":          (10, 70),     # kg/m²
}

# Population defaults for features not present in a given dataset
POPULATION_DEFAULTS = {
    "glucose":          5.5,    # mmol/L (post-conversion)
    "hba1c":            5.4,    # %
    "creatinine":       80.0,   # µmol/L (post-conversion)
    "cholesterol":      5.0,    # mmol/L (post-conversion)
    "triglycerides":    1.3,    # mmol/L
    "hemoglobin":       14.0,   # g/dL
    "bmi":              25.0,   # kg/m²
    "age":              45.0,   # years
    "systolic_bp":      120.0,  # mmHg
    "diastolic_bp":     80.0,   # mmHg
    "smoking_encoded":  0.0,
    "exercise_frequency": 3.0,
    "sleep_hours":      7.0,
    "stress_level":     5.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_class_distribution(df: pd.DataFrame, label: str) -> None:
    """Log class distribution counts and percentages."""
    counts = df["target"].value_counts().sort_index()
    total = len(df)
    parts = [f"class {cls}: {cnt} ({cnt / total * 100:.1f}%)" for cls, cnt in counts.items()]
    print(f"  Class distribution ({label}): {', '.join(parts)}")


def compute_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_governance_stats(train_df: pd.DataFrame, features: list[str]) -> dict:
    """
    Compute per-feature training-set medians and IQR fence values.
    Only called on the training split to prevent data leakage.
    """
    stats: dict = {"medians": {}, "iqr_fences": {}}
    for feat in features:
        if feat not in train_df.columns:
            continue
        series = train_df[feat].dropna()
        median = float(series.median())
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        stats["medians"][feat] = median
        stats["iqr_fences"][feat] = {"lower": lower_fence, "upper": upper_fence}
    return stats


# ── Per-disease preprocessing ─────────────────────────────────────────────────

def preprocess_diabetes() -> pd.DataFrame | None:
    """Align Pima Indians Diabetes dataset to 14-feature schema."""
    print("\n[1/3] Processing Pima Indians Diabetes dataset...")
    path = os.path.join(DATASETS_DIR, "diabetes_pima.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found. Run 00_download_datasets.py first.")
        return None

    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

    aligned = pd.DataFrame()
    # Glucose: Pima stores mg/dL → convert to mmol/L
    aligned["glucose"] = df["Glucose"] * 0.05551
    aligned["hba1c"] = POPULATION_DEFAULTS["hba1c"]
    aligned["creatinine"] = POPULATION_DEFAULTS["creatinine"]
    aligned["cholesterol"] = POPULATION_DEFAULTS["cholesterol"]
    aligned["triglycerides"] = POPULATION_DEFAULTS["triglycerides"]
    aligned["hemoglobin"] = POPULATION_DEFAULTS["hemoglobin"]
    aligned["bmi"] = df["BMI"]
    aligned["age"] = df["Age"]
    aligned["systolic_bp"] = df["BloodPressure"]
    aligned["diastolic_bp"] = df["BloodPressure"] * 0.67
    aligned["smoking_encoded"] = POPULATION_DEFAULTS["smoking_encoded"]
    aligned["exercise_frequency"] = POPULATION_DEFAULTS["exercise_frequency"]
    aligned["sleep_hours"] = POPULATION_DEFAULTS["sleep_hours"]
    aligned["stress_level"] = POPULATION_DEFAULTS["stress_level"]
    aligned["target"] = df["Outcome"]

    # Remove rows with zero glucose (data quality issue in Pima)
    aligned = aligned[aligned["glucose"] > 0].reset_index(drop=True)

    log_class_distribution(aligned, "before split")

    out_path = os.path.join(PROCESSED_DIR, "diabetes_aligned.csv")
    aligned.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(aligned)} samples to {out_path}")
    return aligned


def preprocess_cvd() -> pd.DataFrame | None:
    """Align Framingham CVD dataset to 14-feature schema."""
    print("\n[2/3] Processing Framingham CVD dataset...")
    path = os.path.join(DATASETS_DIR, "framingham_cvd.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found. Download manually from Kaggle.")
        return None

    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

    aligned = pd.DataFrame()
    if "glucose" in df.columns:
        aligned["glucose"] = df["glucose"] * 0.05551  # mg/dL → mmol/L
    else:
        aligned["glucose"] = POPULATION_DEFAULTS["glucose"]
    aligned["hba1c"] = POPULATION_DEFAULTS["hba1c"]
    aligned["creatinine"] = POPULATION_DEFAULTS["creatinine"]
    if "totChol" in df.columns:
        aligned["cholesterol"] = df["totChol"] * 0.02586  # mg/dL → mmol/L
    else:
        aligned["cholesterol"] = POPULATION_DEFAULTS["cholesterol"]
    aligned["triglycerides"] = POPULATION_DEFAULTS["triglycerides"]
    aligned["hemoglobin"] = POPULATION_DEFAULTS["hemoglobin"]
    aligned["bmi"] = df.get("BMI", POPULATION_DEFAULTS["bmi"])
    aligned["age"] = df.get("age", POPULATION_DEFAULTS["age"])
    aligned["systolic_bp"] = df.get("sysBP", POPULATION_DEFAULTS["systolic_bp"])
    aligned["diastolic_bp"] = df.get("diaBP", POPULATION_DEFAULTS["diastolic_bp"])
    if "cigsPerDay" in df.columns:
        aligned["smoking_encoded"] = df["cigsPerDay"].apply(
            lambda x: 0.0 if x == 0 else (0.5 if x <= 10 else 1.0)
        )
    else:
        aligned["smoking_encoded"] = POPULATION_DEFAULTS["smoking_encoded"]
    aligned["exercise_frequency"] = POPULATION_DEFAULTS["exercise_frequency"]
    aligned["sleep_hours"] = POPULATION_DEFAULTS["sleep_hours"]
    aligned["stress_level"] = POPULATION_DEFAULTS["stress_level"]
    aligned["target"] = df.get("TenYearCHD", df.get("target", 0))

    aligned = aligned.dropna(subset=["age", "bmi", "systolic_bp"]).reset_index(drop=True)

    log_class_distribution(aligned, "before split")

    out_path = os.path.join(PROCESSED_DIR, "cvd_aligned.csv")
    aligned.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(aligned)} samples to {out_path}")
    return aligned


def preprocess_ckd() -> pd.DataFrame | None:
    """Align UCI CKD dataset to 14-feature schema."""
    print("\n[3/3] Processing UCI CKD dataset...")
    path = os.path.join(DATASETS_DIR, "ckd_uci.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found. Download manually from UCI or Kaggle.")
        return None

    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

    aligned = pd.DataFrame()
    if "bgr" in df.columns:
        aligned["glucose"] = df["bgr"] * 0.05551  # mg/dL → mmol/L
    else:
        aligned["glucose"] = POPULATION_DEFAULTS["glucose"]
    aligned["hba1c"] = POPULATION_DEFAULTS["hba1c"]
    if "sc" in df.columns:
        aligned["creatinine"] = df["sc"] * 88.42  # mg/dL → µmol/L
    else:
        aligned["creatinine"] = POPULATION_DEFAULTS["creatinine"]
    aligned["cholesterol"] = POPULATION_DEFAULTS["cholesterol"]
    aligned["triglycerides"] = POPULATION_DEFAULTS["triglycerides"]
    aligned["hemoglobin"] = df.get("hemo", POPULATION_DEFAULTS["hemoglobin"])
    aligned["bmi"] = POPULATION_DEFAULTS["bmi"]
    aligned["age"] = df.get("age", POPULATION_DEFAULTS["age"])
    aligned["systolic_bp"] = df.get("bp", POPULATION_DEFAULTS["systolic_bp"])
    aligned["diastolic_bp"] = df.get("bp", POPULATION_DEFAULTS["diastolic_bp"]) * 0.67
    aligned["smoking_encoded"] = POPULATION_DEFAULTS["smoking_encoded"]
    aligned["exercise_frequency"] = POPULATION_DEFAULTS["exercise_frequency"]
    aligned["sleep_hours"] = POPULATION_DEFAULTS["sleep_hours"]
    aligned["stress_level"] = POPULATION_DEFAULTS["stress_level"]

    if "classification" in df.columns:
        aligned["target"] = df["classification"].apply(
            lambda x: 1 if str(x).strip().lower() == "ckd" else 0
        )
    else:
        aligned["target"] = df.get("class", 0)

    aligned = aligned.dropna(subset=["age", "creatinine", "hemoglobin"]).reset_index(drop=True)

    log_class_distribution(aligned, "before split")

    out_path = os.path.join(PROCESSED_DIR, "ckd_aligned.csv")
    aligned.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(aligned)} samples to {out_path}")
    return aligned


# ── Train-test split and governance stats ─────────────────────────────────────

def compute_and_save_governance(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """
    For each disease dataset:
      1. Perform stratified 80/20 train-test split (FIRST operation, before any scaling/PCA)
      2. Compute training-set medians and IQR fences on the training split only
      3. Save split indices and governance stats to training/models/data_governance.json

    This is the canonical source of truth for imputation and outlier clipping at inference time.
    """
    print("\n[Governance] Computing training-set medians and IQR fences...")

    governance: dict = {}

    for disease, df in datasets.items():
        print(f"\n  Disease: {disease} — {len(df)} samples")

        # ── Step 1: Stratified 80/20 split — FIRST operation ─────────────────
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=SEED,
            stratify=df["target"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"  Train: {len(train_df)} samples | Test: {len(test_df)} samples")
        log_class_distribution(train_df, f"{disease} train split")
        log_class_distribution(test_df, f"{disease} test split")

        # ── Step 2: Compute governance stats on training split only ───────────
        subset_features = DISEASE_FEATURE_SUBSETS[disease]
        stats = compute_governance_stats(train_df, subset_features)

        # Also compute stats for all 14 features (used for full-vector imputation)
        all_stats = compute_governance_stats(train_df, FEATURE_NAMES)

        governance[disease] = {
            "disease_feature_subset": subset_features,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_class_distribution": train_df["target"].value_counts().sort_index().to_dict(),
            "test_class_distribution": test_df["target"].value_counts().sort_index().to_dict(),
            "subset_medians": stats["medians"],
            "subset_iqr_fences": stats["iqr_fences"],
            "all_feature_medians": all_stats["medians"],
            "all_feature_iqr_fences": all_stats["iqr_fences"],
        }

        print(f"  Medians computed for {len(stats['medians'])} subset features")
        print(f"  IQR fences computed for {len(stats['iqr_fences'])} subset features")

    # ── Step 3: Save to training/models/data_governance.json ─────────────────
    out_path = os.path.join(MODELS_DIR, "data_governance.json")
    with open(out_path, "w") as f:
        json.dump(governance, f, indent=2)
    print(f"\n  [OK] Saved data_governance.json to {out_path}")


# ── Dataset and environment manifests ─────────────────────────────────────────

def save_dataset_manifest(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Compute SHA-256 hash of each raw CSV and save a dataset manifest.
    Saved to training/results/dataset_manifest.json.
    """
    print("\n[Manifest] Computing dataset manifest...")
    manifest: dict = {}

    disease_to_csv = {
        "diabetes": "diabetes_pima.csv",
        "cvd":      "framingham_cvd.csv",
        "ckd":      "ckd_uci.csv",
    }

    for disease, csv_name in disease_to_csv.items():
        csv_path = os.path.join(DATASETS_DIR, csv_name)
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} not found — skipping manifest entry.")
            continue

        sha256 = compute_sha256(csv_path)
        df = datasets.get(disease)
        if df is not None:
            row_count = len(df)
            columns = list(df.columns)
        else:
            # Read just the header to get column names
            tmp = pd.read_csv(csv_path, nrows=0)
            row_count = sum(1 for _ in open(csv_path)) - 1  # subtract header
            columns = list(tmp.columns)

        manifest[disease] = {
            "filename": csv_name,
            "sha256": sha256,
            "row_count": row_count,
            "columns": columns,
        }
        print(f"  {csv_name}: {row_count} rows, sha256={sha256[:16]}...")

    out_path = os.path.join(RESULTS_DIR, "dataset_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [OK] Saved dataset_manifest.json to {out_path}")


def save_environment_manifest() -> None:
    """
    Record Python and key library versions.
    Saved to training/results/environment_manifest.json.
    """
    print("\n[Manifest] Recording environment manifest...")
    import sys

    env: dict = {
        "python": sys.version,
    }

    # Collect versions gracefully — some packages may not be installed yet
    packages = {
        "numpy":      "numpy",
        "sklearn":    "sklearn",
        "xgboost":    "xgboost",
        "pennylane":  "pennylane",
        "pandas":     "pandas",
    }
    for label, module_name in packages.items():
        try:
            mod = __import__(module_name)
            env[label] = getattr(mod, "__version__", "unknown")
        except (ImportError, Exception) as exc:
            # Catch broad exceptions to handle broken installs (e.g. pennylane
            # dependency conflicts) without aborting the manifest step.
            env[label] = f"error: {type(exc).__name__}"

    out_path = os.path.join(RESULTS_DIR, "environment_manifest.json")
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"  [OK] Saved environment_manifest.json to {out_path}")
    for k, v in env.items():
        print(f"    {k}: {v}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("QuantumHealthAI v2 — Dataset Preprocessing")
    print("=" * 70)
    print(f"Random seeds: numpy={SEED}, random={SEED}")

    # ── 1. Align raw datasets to 14-feature schema ────────────────────────────
    diabetes_df = preprocess_diabetes()
    cvd_df = preprocess_cvd()
    ckd_df = preprocess_ckd()

    # Collect successfully processed datasets
    datasets: dict[str, pd.DataFrame] = {}
    if diabetes_df is not None:
        datasets["diabetes"] = diabetes_df
    if cvd_df is not None:
        datasets["cvd"] = cvd_df
    if ckd_df is not None:
        datasets["ckd"] = ckd_df

    if not datasets:
        print("\n[ERROR] No datasets were processed. Exiting.")
        raise SystemExit(1)

    # ── 2. Stratified split + governance stats (FIRST, before any scaling) ────
    compute_and_save_governance(datasets)

    # ── 3. Dataset and environment manifests ──────────────────────────────────
    save_dataset_manifest(datasets)
    save_environment_manifest()

    print("\n" + "=" * 70)
    print("Preprocessing complete.")
    print(f"  Aligned datasets  → {PROCESSED_DIR}")
    print(f"  Governance stats  → {MODELS_DIR}/data_governance.json")
    print(f"  Dataset manifest  → {RESULTS_DIR}/dataset_manifest.json")
    print(f"  Env manifest      → {RESULTS_DIR}/environment_manifest.json")
    print("=" * 70)
    print("\nNext step: python training/scripts/02_train_classical_models_v2.py")
