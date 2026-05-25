"""
Script 00b: Generate synthetic datasets for CVD and CKD.

Uses clinically validated distributions to generate realistic synthetic data
when real datasets are unavailable. These are for development/demo purposes.
Replace with real datasets for production use.

Distributions based on:
- Framingham Heart Study published statistics
- UCI CKD dataset published statistics
"""
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


def generate_cvd_dataset(n_samples: int = 4000):
    """Generate synthetic Framingham-like CVD dataset."""
    print("  Generating synthetic CVD dataset (Framingham-like)...")

    # Negative class (no CVD) — ~85% of population
    n_neg = int(n_samples * 0.85)
    n_pos = n_samples - n_neg

    def gen_class(n, is_positive):
        """Generate samples for one class."""
        age_mean = 55 if is_positive else 45
        chol_mean = 240 if is_positive else 200
        sbp_mean = 145 if is_positive else 125
        dbp_mean = 90 if is_positive else 80
        bmi_mean = 28 if is_positive else 25
        cigs_mean = 10 if is_positive else 2

        data = {
            "age": np.clip(np.random.normal(age_mean, 10, n), 30, 80),
            "totChol": np.clip(np.random.normal(chol_mean, 40, n), 100, 400),
            "sysBP": np.clip(np.random.normal(sbp_mean, 20, n), 90, 200),
            "diaBP": np.clip(np.random.normal(dbp_mean, 12, n), 60, 130),
            "BMI": np.clip(np.random.normal(bmi_mean, 5, n), 15, 50),
            "cigsPerDay": np.clip(np.random.exponential(cigs_mean, n), 0, 60).astype(int),
            "glucose": np.clip(np.random.normal(100 if is_positive else 85, 20, n), 60, 300),
            "TenYearCHD": np.ones(n, dtype=int) if is_positive else np.zeros(n, dtype=int),
        }
        return pd.DataFrame(data)

    df_neg = gen_class(n_neg, False)
    df_pos = gen_class(n_pos, True)
    df = pd.concat([df_neg, df_pos], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    out_path = os.path.join(DATASETS_DIR, "framingham_cvd.csv")
    df.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(df)} samples to {out_path}")
    print(f"  Class distribution: {df['TenYearCHD'].value_counts().to_dict()}")


def generate_ckd_dataset(n_samples: int = 400):
    """Generate synthetic UCI CKD-like dataset."""
    print("  Generating synthetic CKD dataset (UCI-like)...")

    n_ckd = int(n_samples * 0.625)  # UCI has ~250 CKD, 150 not-CKD
    n_notckd = n_samples - n_ckd

    def gen_class(n, is_ckd):
        sc_mean = 4.5 if is_ckd else 0.9   # serum creatinine mg/dL
        hemo_mean = 10.5 if is_ckd else 14.5
        bp_mean = 80 if is_ckd else 70
        bgr_mean = 160 if is_ckd else 100
        age_mean = 55 if is_ckd else 45

        data = {
            "age": np.clip(np.random.normal(age_mean, 15, n), 2, 90),
            "bp": np.clip(np.random.normal(bp_mean, 15, n), 50, 180),
            "bgr": np.clip(np.random.normal(bgr_mean, 50, n), 60, 490),
            "bu": np.clip(np.random.normal(80 if is_ckd else 35, 30, n), 10, 200),
            "sc": np.clip(np.random.normal(sc_mean, 2.5 if is_ckd else 0.3, n), 0.4, 15),
            "hemo": np.clip(np.random.normal(hemo_mean, 2, n), 3.1, 17.8),
            "pcv": np.clip(np.random.normal(32 if is_ckd else 44, 6, n), 9, 54),
            "classification": ["ckd"] * n if is_ckd else ["notckd"] * n,
        }
        return pd.DataFrame(data)

    df_ckd = gen_class(n_ckd, True)
    df_notckd = gen_class(n_notckd, False)
    df = pd.concat([df_ckd, df_notckd], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    out_path = os.path.join(DATASETS_DIR, "ckd_uci.csv")
    df.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(df)} samples to {out_path}")
    print(f"  Class distribution: {df['classification'].value_counts().to_dict()}")


if __name__ == "__main__":
    print("=" * 60)
    print("QuantumHealthAI — Synthetic Dataset Generation")
    print("(Use real datasets for production)")
    print("=" * 60)

    # Only generate if not already present
    cvd_path = os.path.join(DATASETS_DIR, "framingham_cvd.csv")
    if not os.path.exists(cvd_path):
        generate_cvd_dataset()
    else:
        print(f"  [SKIP] CVD dataset already exists: {cvd_path}")

    ckd_path = os.path.join(DATASETS_DIR, "ckd_uci.csv")
    if not os.path.exists(ckd_path):
        generate_ckd_dataset()
    else:
        print(f"  [SKIP] CKD dataset already exists: {ckd_path}")

    print("\nDone. Replace with real datasets for production use.")
