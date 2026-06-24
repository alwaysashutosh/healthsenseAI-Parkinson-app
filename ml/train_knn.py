"""Train the K-Nearest Neighbors classifier on the UCI Parkinson's dataset.

Matches the project report: KNN (Euclidean distance, distance weighting), with
systematic hyperparameter tuning over k, trained on the 195 voice recordings /
31 subjects of the UCI Parkinson's dataset (22 acoustic biomarkers).

Outputs (ml/artifacts/):
    knn_model.pkl        fitted KNeighborsClassifier
    scaler.pkl           fitted StandardScaler (22 features)
    feature_names.json   ordered feature list (for the serving layer)
    metrics.json         real evaluation metrics + k-vs-accuracy curve

Usage:
    python ml/train_knn.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from feature_spec import FEATURE_NAMES, LABEL_COLUMN, ID_COLUMN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "uci_parkinsons.csv")
ARTIFACTS = os.path.join(HERE, "artifacts")
RANDOM_STATE = 42

# The configuration the project report specifies for the deployed classifier:
# KNN with k=5, Euclidean distance, distance weighting.
REPORT_PARAMS = {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}


def subject_of(name: str) -> str:
    """phon_R01_S01_1 -> S01 (groups the multiple recordings of one subject)."""
    for part in str(name).split("_"):
        if part.startswith("S"):
            return part
    return str(name)


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded UCI dataset: {df.shape[0]} recordings, {df.shape[1]} columns")

    X = df[FEATURE_NAMES].values
    y = df[LABEL_COLUMN].values.astype(int)
    subjects = df[ID_COLUMN].map(subject_of)
    print(f"Subjects: {subjects.nunique()} | class balance: "
          f"PD={int((y == 1).sum())}, Healthy={int((y == 0).sum())}")

    # Held-out test split (stratified). This mirrors the report's evaluation setup.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # Scale on train only (no leakage).
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Hyperparameter tuning over k (report: "KNN Hyperparameter Tuning") ---
    param_grid = {
        "n_neighbors": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_s, y_train)
    print(f"Grid-search optimum: {grid.best_params_} (CV acc={grid.best_score_:.4f})")
    print(f"Deploying report-specified config: {REPORT_PARAMS}")

    # Deployed classifier uses the report's specified hyperparameters (k=5).
    best = KNeighborsClassifier(**REPORT_PARAMS).fit(X_train_s, y_train)

    # k-vs-accuracy curve (Euclidean, distance weighting) for the report figure.
    k_curve = []
    for k in range(1, 21):
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
        cv_acc = cross_val_score(
            clf, X_train_s, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
        ).mean()
        k_curve.append({"k": k, "accuracy": round(float(cv_acc), 4)})

    # --- Evaluate the tuned model on the held-out test set ---
    y_pred = best.predict(X_test_s)
    y_proba = best.predict_proba(X_test_s)[:, 1]
    metrics = {
        "model": "KNeighborsClassifier",
        "params": REPORT_PARAMS,
        "grid_search_optimum": grid.best_params_,
        "cv_accuracy": round(float(grid.best_score_), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_size": int(len(y_test)),
        "n_features": len(FEATURE_NAMES),
        "dataset": "UCI Parkinson's (195 recordings, 31 subjects)",
        "k_accuracy_curve": k_curve,
    }
    print("\nClassification report (held-out test):")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    # --- Refit best model on ALL data for deployment ---
    final_scaler = StandardScaler().fit(X)
    final_model = KNeighborsClassifier(**REPORT_PARAMS).fit(final_scaler.transform(X), y)

    joblib.dump(final_model, os.path.join(ARTIFACTS, "knn_model.pkl"))
    joblib.dump(final_scaler, os.path.join(ARTIFACTS, "scaler.pkl"))
    with open(os.path.join(ARTIFACTS, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to {ARTIFACTS}")
    print(f"Test accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
          f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
