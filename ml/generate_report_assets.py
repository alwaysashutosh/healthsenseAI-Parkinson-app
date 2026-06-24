"""Generate real result figures and metrics for the project report.

Trains KNN (default vs tuned), SVM, Random Forest and Logistic Regression on the
SAME stratified held-out split of the UCI Parkinson's dataset and produces:

  BE_Report-2/model_comparison.png   grouped bar chart across models
  BE_Report-2/roc_curve.png          ROC curves (all models) with AUC
  BE_Report-2/confusion_matrix.png   heatmap for the deployed KNN
  BE_Report-2/k_accuracy.png         k-value vs CV accuracy
  ml/artifacts/report_assets.json    all numeric results

These feed the Results chapter so the tables/figures use real measured values.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

from feature_spec import FEATURE_NAMES, LABEL_COLUMN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "uci_parkinsons.csv")
FIG = os.path.join(HERE, "..", "BE_Report-2")
ART = os.path.join(HERE, "artifacts")
RS = 42
BLUE = "#3b82f6"


def metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "specificity": round(tn / (tn + fp) * 100, 2) if (tn + fp) else 0.0,
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "auc": round(roc_auc_score(y_true, y_proba) * 100, 2),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df[FEATURE_NAMES].values
    y = df[LABEL_COLUMN].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=RS, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RS),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RS),
    }
    results = {}
    proba_store = {}
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xte_s)
        proba = clf.predict_proba(Xte_s)[:, 1]
        proba_store[name] = proba
        results[name] = metrics(yte, pred, proba)

    # Before vs after optimization (default KNN vs tuned KNN), both scaled
    knn_default = KNeighborsClassifier().fit(Xtr_s, ytr)
    before = metrics(yte, knn_default.predict(Xte_s), knn_default.predict_proba(Xte_s)[:, 1])
    after = results["KNN"]

    # k vs CV accuracy
    ks = list(range(1, 21))
    k_acc = []
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
        k_acc.append(round(cross_val_score(
            clf, Xtr_s, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
            scoring="accuracy").mean() * 100, 2))

    # ---- Figures ----
    # 1. model comparison grouped bar chart
    mets = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    xpos = np.arange(len(mets))
    width = 0.2
    plt.figure(figsize=(9, 5))
    for i, name in enumerate(models):
        vals = [results[name][m] for m in mets]
        plt.bar(xpos + i * width, vals, width, label=name)
    plt.xticks(xpos + 1.5 * width, labels)
    plt.ylabel("Score (%)")
    plt.ylim(60, 100)
    plt.title("Model Performance Comparison (UCI held-out test set)")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "model_comparison.png"), dpi=150)
    plt.close()

    # 2. ROC curves
    plt.figure(figsize=(6.5, 5.5))
    for name in models:
        fpr, tpr, _ = roc_curve(yte, proba_store[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['auc']/100:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "roc_curve.png"), dpi=150)
    plt.close()

    # 3. confusion matrix heatmap for KNN
    cm = np.array(results["KNN"]["cm"])
    plt.figure(figsize=(5, 4.2))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black", fontsize=14, fontweight="bold")
    plt.xticks([0, 1], ["Healthy", "Parkinson's"])
    plt.yticks([0, 1], ["Healthy", "Parkinson's"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (KNN)")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 4. k vs accuracy
    plt.figure(figsize=(7, 4.2))
    plt.plot(ks, k_acc, marker="o", color=BLUE)
    best_k = ks[int(np.argmax(k_acc))]
    plt.axvline(5, color="red", ls="--", alpha=0.5, label="k = 5 (deployed)")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("5-fold CV accuracy (%)")
    plt.title("K-Value vs Cross-Validation Accuracy")
    plt.xticks(ks)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "k_accuracy.png"), dpi=150)
    plt.close()

    out = {
        "test_size": int(len(yte)),
        "models": results,
        "optimization": {"before": before, "after": after},
        "k_curve": dict(zip(ks, k_acc)),
        "best_cv_k": int(best_k),
    }
    with open(os.path.join(ART, "report_assets.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("Saved figures to BE_Report-2/ and metrics to ml/artifacts/report_assets.json\n")
    print(f"{'Model':<22}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}")
    for name in models:
        r = results[name]
        print(f"{name:<22}{r['accuracy']:>7}{r['precision']:>7}{r['recall']:>7}{r['f1']:>7}{r['auc']:>7}")
    print(f"\nBefore opt (default KNN): acc={before['accuracy']} f1={before['f1']} auc={before['auc']}")
    print(f"After opt  (tuned KNN):   acc={after['accuracy']} f1={after['f1']} auc={after['auc']}")


if __name__ == "__main__":
    main()
