"""Deploy the CTGAN-augmented model as the production classifier.

The held-out experiment in `augment_ctgan.py` showed that balancing the training
data with CTGAN improves minority-class performance. This script promotes that
approach to production:

1. Train a CTGAN on the FULL UCI dataset (more data => better final model, standard
   practice for a deployed model fit on all available data).
2. Generate synthetic healthy rows to balance the full dataset.
3. Refit the StandardScaler + KNN (same config) on the balanced full data.
4. Overwrite the served artifacts (knn_model.pkl, scaler.pkl).
5. Update metrics.json so the dashboard reports the model's honest *held-out*
   augmented metrics (taken from augmentation_metrics.json), flagged as augmented.

Run `augment_ctgan.py` first (this script reuses its held-out metrics for reporting).

Usage:
    python ml/deploy_augmented.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

import torch
from ctgan import CTGAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from feature_spec import FEATURE_NAMES, LABEL_COLUMN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "uci_parkinsons.csv")
ARTIFACTS = os.path.join(HERE, "artifacts")
RANDOM_STATE = 42
KNN_PARAMS = {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}
CTGAN_EPOCHS = 400
CTGAN_BATCH = 50


def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    df = pd.read_csv(DATA_PATH)[FEATURE_NAMES + [LABEL_COLUMN]].copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    y = df[LABEL_COLUMN].values
    n_pd, n_hc = int((y == 1).sum()), int((y == 0).sum())
    print(f"Full dataset: PD={n_pd}, Healthy={n_hc} (imbalance {n_pd/max(n_hc,1):.2f}:1)")

    # 1-2. CTGAN on full data -> synthetic healthy to balance
    print(f"Training deployment CTGAN ({CTGAN_EPOCHS} epochs) on full data...")
    ctgan = CTGAN(epochs=CTGAN_EPOCHS, batch_size=CTGAN_BATCH, verbose=False)
    ctgan.fit(df, discrete_columns=[LABEL_COLUMN])
    n_needed = n_pd - n_hc
    synth = ctgan.sample(n_needed, condition_column=LABEL_COLUMN, condition_value=0)
    synth[LABEL_COLUMN] = 0
    synth = synth[FEATURE_NAMES + [LABEL_COLUMN]]
    aug = pd.concat([df, synth], ignore_index=True)
    print(f"Balanced full set: PD={n_pd}, Healthy={n_hc + n_needed} "
          f"(+{n_needed} synthetic)")

    # 3-4. Refit and overwrite the served artifacts
    Xa = aug[FEATURE_NAMES].values
    ya = aug[LABEL_COLUMN].values
    scaler = StandardScaler().fit(Xa)
    knn = KNeighborsClassifier(**KNN_PARAMS).fit(scaler.transform(Xa), ya)
    joblib.dump(knn, os.path.join(ARTIFACTS, "knn_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS, "scaler.pkl"))
    aug.to_csv(os.path.join(ARTIFACTS, "augmented_full.csv"), index=False)
    print("Deployed augmented model -> knn_model.pkl, scaler.pkl")

    # 5. Update metrics.json with the honest held-out augmented metrics
    metrics_path = os.path.join(ARTIFACTS, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    aug_path = os.path.join(ARTIFACTS, "augmentation_metrics.json")
    if os.path.exists(aug_path):
        with open(aug_path) as f:
            held = json.load(f)["metrics_augmented"]
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"):
            if k in held:
                metrics[k] = held[k]
        metrics["specificity"] = held.get("specificity")
        metrics["macro_f1"] = held.get("macro_f1")
    else:
        print("WARNING: augmentation_metrics.json not found; reported metrics not updated. "
              "Run ml/augment_ctgan.py first for honest held-out numbers.")

    metrics["model"] = "KNeighborsClassifier (CTGAN-augmented)"
    metrics["augmented"] = True
    metrics["training_rows"] = {"real": int(len(y)), "synthetic_healthy": int(n_needed),
                                "total": int(len(ya))}
    metrics["dataset"] = "UCI Parkinson's (195 recordings) + CTGAN-balanced training"
    metrics["note"] = ("Deployed model is fit on CTGAN-balanced full data; reported "
                       "metrics are the held-out (30% real test) augmented results "
                       "from augment_ctgan.py.")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {metrics_path}")
    print(f"  accuracy={metrics.get('accuracy')} precision={metrics.get('precision')} "
          f"recall={metrics.get('recall')} specificity={metrics.get('specificity')} "
          f"macro_f1={metrics.get('macro_f1')}")


if __name__ == "__main__":
    main()
