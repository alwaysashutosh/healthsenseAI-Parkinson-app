"""CTGAN-based data augmentation to address class imbalance in the UCI dataset.

The UCI Parkinson's dataset is imbalanced (~147 PD vs ~48 healthy). This script
trains a Conditional Tabular GAN (CTGAN) on the TRAINING split only, generates
synthetic minority-class (healthy) rows to balance the training set, retrains the
KNN classifier on the augmented data, and compares it against the unaugmented
baseline on a held-out REAL test set.

Leakage safety: CTGAN never sees the test split. The test set is 100% real and is
identical to the one used by train_knn.py (same stratified split, random_state=42),
so the before/after comparison is fair.

Synthetic medical data is for augmentation only and must never be presented as
real patient data.

Outputs (ml/artifacts/):
    ctgan_model.pkl            trained CTGAN
    synthetic_healthy.csv      generated minority-class rows
    augmented_train.csv        real train + synthetic healthy
    augmentation_metrics.json  baseline vs augmented comparison + synthetic-data QA

Usage:
    python ml/augment_ctgan.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

import torch
from ctgan import CTGAN
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from feature_spec import FEATURE_NAMES, LABEL_COLUMN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "uci_parkinsons.csv")
ARTIFACTS = os.path.join(HERE, "artifacts")
RANDOM_STATE = 42
# Same deployed KNN configuration as train_knn.py
KNN_PARAMS = {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}
CTGAN_EPOCHS = 400
CTGAN_BATCH = 50   # multiple of pac(=10), < training rows


def _seed():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def _evaluate(knn, scaler, X_test, y_test):
    Xs = scaler.transform(X_test)
    pred = knn.predict(Xs)
    proba = knn.predict_proba(Xs)[:, list(knn.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _train_knn(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    knn = KNeighborsClassifier(**KNN_PARAMS).fit(scaler.transform(X_train), y_train)
    return knn, scaler


def main():
    _seed()
    os.makedirs(ARTIFACTS, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURE_NAMES + [LABEL_COLUMN]].copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    X = df[FEATURE_NAMES].values
    y = df[LABEL_COLUMN].values

    # 1. Leakage-safe split (identical to train_knn.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    n_pd = int((y_train == 1).sum())
    n_hc = int((y_train == 0).sum())
    print(f"Train: PD={n_pd}, Healthy={n_hc} (imbalance {n_pd/max(n_hc,1):.2f}:1) | Test rows={len(y_test)}")

    train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    train_df[LABEL_COLUMN] = y_train

    # 2. Train CTGAN on the TRAINING data only (status as discrete conditional col)
    print(f"Training CTGAN ({CTGAN_EPOCHS} epochs)...")
    ctgan = CTGAN(epochs=CTGAN_EPOCHS, batch_size=CTGAN_BATCH, verbose=False)
    ctgan.fit(train_df, discrete_columns=[LABEL_COLUMN])

    # 3. Conditionally generate synthetic HEALTHY (status=0) rows to balance
    n_needed = n_pd - n_hc
    print(f"Generating {n_needed} synthetic healthy rows...")
    synth = ctgan.sample(n_needed, condition_column=LABEL_COLUMN, condition_value=0)
    synth[LABEL_COLUMN] = 0  # enforce the conditioned label
    synth = synth[FEATURE_NAMES + [LABEL_COLUMN]]

    # 4. Augmented training set (test set stays 100% real)
    aug_df = pd.concat([train_df, synth], ignore_index=True)

    # 5. Synthetic-data quality checks (synthetic vs real healthy, per feature)
    real_hc = train_df[train_df[LABEL_COLUMN] == 0][FEATURE_NAMES]
    qa = {}
    for col in FEATURE_NAMES:
        ks_stat, ks_p = ks_2samp(real_hc[col].values, synth[col].values)
        qa[col] = {
            "real_mean": round(float(real_hc[col].mean()), 4),
            "synth_mean": round(float(synth[col].mean()), 4),
            "ks_stat": round(float(ks_stat), 3),
            "ks_p": round(float(ks_p), 3),
        }
    nan_count = int(synth.isna().sum().sum())
    # mean KS statistic: lower = synthetic distribution closer to real
    mean_ks = round(float(np.mean([v["ks_stat"] for v in qa.values()])), 3)
    print(f"Synthetic QA: NaNs={nan_count}, mean KS stat={mean_ks} (lower is better)")

    # 6 & 7. Baseline vs augmented KNN, both evaluated on the REAL test set
    base_knn, base_scaler = _train_knn(X_train, y_train)
    baseline = _evaluate(base_knn, base_scaler, X_test, y_test)

    Xa = aug_df[FEATURE_NAMES].values
    ya = aug_df[LABEL_COLUMN].values
    aug_knn, aug_scaler = _train_knn(Xa, ya)
    augmented = _evaluate(aug_knn, aug_scaler, X_test, y_test)

    print("\n=== Held-out test set (real data only) ===")
    print(f"{'metric':<12}{'baseline':>10}{'augmented':>12}")
    for m in ["accuracy", "precision", "recall", "specificity", "macro_f1", "roc_auc"]:
        print(f"{m:<12}{baseline[m]:>10}{augmented[m]:>12}")

    # 8. Persist artifacts
    ctgan.save(os.path.join(ARTIFACTS, "ctgan_model.pkl"))
    synth.to_csv(os.path.join(ARTIFACTS, "synthetic_healthy.csv"), index=False)
    aug_df.to_csv(os.path.join(ARTIFACTS, "augmented_train.csv"), index=False)

    report = {
        "method": "CTGAN conditional tabular augmentation (minority class = healthy)",
        "ctgan": {"epochs": CTGAN_EPOCHS, "batch_size": CTGAN_BATCH},
        "train_before": {"PD": n_pd, "healthy": n_hc},
        "synthetic_healthy_added": int(n_needed),
        "train_after": {"PD": n_pd, "healthy": n_hc + int(n_needed)},
        "test_size": int(len(y_test)),
        "knn_params": KNN_PARAMS,
        "metrics_baseline": baseline,
        "metrics_augmented": augmented,
        "synthetic_quality": {"nan_count": nan_count, "mean_ks_stat": mean_ks, "per_feature": qa},
        "note": ("Test set is 100% real and identical to train_knn.py's split; CTGAN "
                 "trained on the training split only. Synthetic data is augmentation, "
                 "not real patient data."),
    }
    with open(os.path.join(ARTIFACTS, "augmentation_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    main()
