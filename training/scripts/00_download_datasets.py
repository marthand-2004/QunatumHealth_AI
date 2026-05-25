"""
Script 00: Download datasets from Kaggle/UCI.

Usage:
    python training/scripts/00_download_datasets.py

Requirements:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
    - OR manually download and place in training/datasets/

Manual download URLs:
    Diabetes: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    CVD:      https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
    CKD:      https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
"""
import os
import sys
import urllib.request

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# Direct download URLs (public mirrors)
DATASETS = {
    "diabetes_pima.csv": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
         "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    ),
    "framingham_cvd.csv": None,  # Requires Kaggle auth — see manual instructions
    "ckd_uci.csv": None,         # Requires UCI download — see manual instructions
}

def download_diabetes():
    """Download Pima Indians Diabetes dataset (public)."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    dest = os.path.join(DATASETS_DIR, "diabetes_pima.csv")
    if os.path.exists(dest):
        print(f"  [SKIP] {dest} already exists")
        return

    print(f"  Downloading Pima Indians Diabetes dataset...")
    headers = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    try:
        urllib.request.urlretrieve(url, dest)
        # Add header row
        with open(dest, "r") as f:
            content = f.read()
        with open(dest, "w") as f:
            f.write(",".join(headers) + "\n" + content)
        print(f"  [OK] Saved to {dest}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        print(f"  Manual: Download from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")


def download_cvd():
    """Download Framingham CVD dataset via Kaggle API."""
    dest = os.path.join(DATASETS_DIR, "framingham_cvd.csv")
    if os.path.exists(dest):
        print(f"  [SKIP] {dest} already exists")
        return

    print("  Attempting Kaggle download for Framingham CVD dataset...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "dileep070/heart-disease-prediction-using-logistic-regression",
            path=DATASETS_DIR,
            unzip=True
        )
        # Rename to standard name
        for f in os.listdir(DATASETS_DIR):
            if "framingham" in f.lower() and f.endswith(".csv"):
                os.rename(os.path.join(DATASETS_DIR, f), dest)
                break
        print(f"  [OK] Saved to {dest}")
    except Exception as e:
        print(f"  [FAIL] Kaggle download failed: {e}")
        print(f"""
  Manual download instructions for Framingham CVD:
  1. Go to: https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
  2. Download framingham.csv
  3. Save as: training/datasets/framingham_cvd.csv
""")


def download_ckd():
    """Download UCI CKD dataset."""
    dest = os.path.join(DATASETS_DIR, "ckd_uci.csv")
    if os.path.exists(dest):
        print(f"  [SKIP] {dest} already exists")
        return

    # Try direct UCI download
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.rar"
    print("  UCI CKD dataset requires manual download (RAR format).")
    print(f"""
  Manual download instructions for CKD:
  1. Go to: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
  2. Download the dataset
  3. Extract chronic_kidney_disease.arff
  4. Convert to CSV and save as: training/datasets/ckd_uci.csv

  Alternative (Kaggle mirror):
  https://www.kaggle.com/datasets/mansoordaku/ckdisease
  Download kidney_disease.csv and save as training/datasets/ckd_uci.csv
""")

    # Try Kaggle mirror
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mansoordaku/ckdisease",
            path=DATASETS_DIR,
            unzip=True
        )
        for f in os.listdir(DATASETS_DIR):
            if "kidney" in f.lower() and f.endswith(".csv"):
                os.rename(os.path.join(DATASETS_DIR, f), dest)
                break
        print(f"  [OK] Saved to {dest}")
    except Exception as e:
        print(f"  [INFO] Kaggle mirror also unavailable: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("QuantumHealthAI — Dataset Download")
    print("=" * 60)

    print("\n[1/3] Pima Indians Diabetes Dataset")
    download_diabetes()

    print("\n[2/3] Framingham CVD Dataset")
    download_cvd()

    print("\n[3/3] UCI Chronic Kidney Disease Dataset")
    download_ckd()

    print("\n" + "=" * 60)
    print("Dataset download complete.")
    print(f"Check: {DATASETS_DIR}")
    print("=" * 60)
