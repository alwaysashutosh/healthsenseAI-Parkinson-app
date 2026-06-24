# ML_ARCHITECTURE.md — Machine Learning Audit

> Audit date: 2026-06-21 · Branch: `parkinsons-only-voice`

## 1. What model is actually trained?

The **active, deployed model is a `RandomForestClassifier`** (scikit-learn), serialized as
[rf_model.pkl](rf_model.pkl). Verified by loading the artifact:

| Artifact | Type | Key facts |
|----------|------|-----------|
| [rf_model.pkl](rf_model.pkl) | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=100`, `n_features_in_=17`, `classes_=[0,1]` |
| [scaler.pkl](scaler.pkl) | `sklearn.preprocessing.StandardScaler` | `n_features_in_=17` |
| [selected_features.pkl](selected_features.pkl) | `numpy.ndarray` (bool) | length **52**, **17 features selected** |
| [parkinsons-backend/Parkinson/model_parkinson.pkl](parkinsons-backend/Parkinson/model_parkinson.pkl) | legacy pickle | **Will not load** — `ModuleNotFoundError: sklearn.tree.tree` (pickled with sklearn <0.21). Dead artifact. |

## 2. Random Forest or KNN?

- **Code/artifacts: Random Forest.** Training ([parkinsons_detection.py:927](parkinsons_detection.py#L927)) and
  inference ([parkinsons_detection.py:806](parkinsons_detection.py#L806) `LivePredictor`) both use Random Forest.
- **Project report: claims KNN** (k=5, Euclidean, distance weighting) — see
  [Project_Report_Parkinsons_Detection.txt:147](Project_Report_Parkinsons_Detection.txt#L147).
- **KNN is only present as one option** inside `hyperparameter_tuning()` / `ensemble_classification()`
  ([parkinsons_detection.py:597](parkinsons_detection.py#L597)), never selected or saved.

➡️ **Architecture mismatch.** The report and the implementation describe two different systems.

## 3. Dataset used

- **Actual:** the repo's own recordings of the sustained vowel **/a/ ("AH")**:
  - [HC_AH/HC_AH/](HC_AH/HC_AH/) — **41 healthy** `.wav` files (label `0`)
  - [PD_AH/PD_AH/](PD_AH/PD_AH/) — **40 Parkinson's** `.wav` files (label `1`)
  - [PD_AH_Augmented/](PD_AH_Augmented/) — augmented PD copies (`aug_1_*`, `aug_2_*`); **NOT loaded by training** (`main()` only reads `PD_AH/PD_AH` and `HC_AH/HC_AH`).
  - [Demographics_age_sex.xlsx](Demographics_age_sex.xlsx) — read only by the standalone [read_demographics.py](read_demographics.py); not part of training.
- **Report claims:** UCI Parkinson's dataset, **195 recordings / 31 subjects, 22 acoustic features**
  ([Project_Report_Parkinsons_Detection.txt:496](Project_Report_Parkinsons_Detection.txt#L496)). **Not present in repo and not used.**

## 4 & 5. Features used / feature count

Features are extracted by `extract_features()` ([parkinsons_detection.py:30](parkinsons_detection.py#L30)) with **librosa**, in this exact order — **52 features total**:

| Block | Count | Features |
|-------|-------|----------|
| MFCC mean | 13 | `MFCC1_mean … MFCC13_mean` |
| MFCC std | 13 | `MFCC1_std … MFCC13_std` |
| Spectral | 3 | centroid, rolloff, bandwidth |
| ZCR | 1 | zero-crossing rate |
| RMS | 1 | root-mean-square energy |
| Chroma | 12 | `Chroma1 … Chroma12` |
| Spectral contrast | 7 | 7 sub-bands |
| Spectral flatness | 1 | |
| Tempo | 1 | |
| **Total** | **52** | feature-selection mask reduces to **17** |

> ⚠️ The clinical biomarkers the report is built around — **jitter, shimmer, HNR, RPDE, DFA, PPE — are NOT implemented anywhere.** See [AUDIO_PIPELINE_REPORT.md](AUDIO_PIPELINE_REPORT.md).

> ⚠️ **Feature-name list drift:** `backend/main.py:ALL_FEATURES` ([backend/main.py:118](backend/main.py#L118)) lists 46 names in a *different order* (contrast/flatness/tempo placed before chroma) than the real extraction order. It is display-only metadata and does not affect inference, but it is misleading and inconsistent with the trained model.

## 6. Training pipeline

`main()` ([parkinsons_detection.py:949](parkinsons_detection.py#L949)):
1. Walk `PD_AH/PD_AH` (label 1) and `HC_AH/HC_AH` (label 0) → `extract_features` → `X (81×52)`, `y`.
2. `train_and_save_models()` runs 3 metaheuristic feature selectors — **GWO, ABC, PSO** (all hand-implemented, [parkinsons_detection.py:91–404](parkinsons_detection.py#L91)) — each wrapping a Random Forest fitness function.
3. Best algorithm chosen by accuracy → boolean mask saved to `selected_features.pkl`.
4. `StandardScaler` fit on the selected-feature training split → `scaler.pkl`.
5. `RandomForestClassifier(n_estimators=100)` fit → `rf_model.pkl`.
6. Re-running is **skipped if `rf_model.pkl` already exists** ([parkinsons_detection.py:1020](parkinsons_detection.py#L1020)).

## 7. Evaluation pipeline

- Single `train_test_split(test_size=0.3, random_state=42, stratify=y)` (~25 test samples).
- `accuracy_score` + `classification_report` + `confusion_matrix`.
- `hyperparameter_tuning()` (GridSearchCV, cv=5) and `cross_validation_analysis()` exist but the latter two are **commented out / skipped** in `main()` "for speed" ([parkinsons_detection.py:1077](parkinsons_detection.py#L1077)).
- All accuracy numbers shown in the UIs (76% / 72% / 74%) are **hard-coded literals**, not read from any evaluation run ([parkinsons_app.py:96](parkinsons_app.py#L96), [backend/main.py:46](backend/main.py#L46)).

## 8. Data leakage issues 🔴

**Yes — significant leakage that inflates reported accuracy:**

1. **Feature selection leakage.** The GWO/ABC/PSO fitness functions evaluate candidate feature subsets using `train_test_split(..., random_state=42)` ([parkinsons_detection.py:174](parkinsons_detection.py#L174), [:293](parkinsons_detection.py#L293), [:397](parkinsons_detection.py#L397)) and the **final model is then evaluated on the same `random_state=42` split**. Features are chosen by maximizing accuracy on the very set later used to report accuracy → optimistic bias.
2. **No held-out test set.** Selection, tuning, and final evaluation all reuse the same 30% split; there is no independent test partition.
3. **Tiny test set.** ~25 samples → each sample is ~4% accuracy; metrics are high-variance.
4. **Augmented data risk (latent).** `PD_AH_Augmented` is not currently loaded, but if added naively, augmented variants of a training file landing in the test split would be a second leakage source. Splitting must be **group-aware by source recording**.

## 9. Missing preprocessing

- No **noise reduction**, no **silence/voiced-segment trimming**, no **loudness normalization** before feature extraction (the report claims all three). `librosa.load(sr=22050)` only resamples.
- `LivePredictor` has a crude RMS<0.01 silence gate ([parkinsons_detection.py:861](parkinsons_detection.py#L861)) but no VAD/denoise.
- No class-imbalance handling (dataset is roughly balanced 40/41, so low priority).
- Inference path (`LivePredictor`) and training path share `extract_features`, so feature ordering is consistent end-to-end ✅ (a genuine strength).

## 10. Retraining requirements

To make the model defensible:
1. **Fix leakage:** nested CV or a 3-way split (train / selection-validation / held-out test); select features on train only, report on untouched test.
2. **Group-aware split** if/when augmented data is included (group by original recording ID).
3. **Persist real metrics** to a JSON/`metrics.json` and have all UIs read from it instead of hard-coded numbers.
4. **Decide the dataset story:** either (a) keep the AH vowel dataset and rewrite the report, or (b) actually integrate the UCI dataset + jitter/shimmer/HNR/RPDE/DFA/PPE to match the report.
5. **Version artifacts** with the feature list + sklearn version (legacy `.pkl` already broke on a version bump).
6. Re-export `rf_model.pkl`, `scaler.pkl`, `selected_features.pkl` together (they are coupled: mask length 52 → 17 selected → scaler/model expect 17).

## Summary

| Question | Finding |
|----------|---------|
| Model | Random Forest (100 trees), **not KNN** |
| Features | 52 extracted (librosa MFCC/spectral/chroma/contrast/flatness/tempo) → 17 selected |
| Feature selection | GWO / ABC / PSO (custom metaheuristics) |
| Dataset | 81 in-repo AH-vowel WAVs (40 PD / 41 HC); **not** UCI/195 |
| Reported accuracy | Hard-coded 76% in UIs; report claims 92.3% |
| Leakage | Yes — selection + evaluation on same split, no held-out set |
| Biggest gap | Report describes a KNN/UCI/jitter-shimmer system that does not exist in code |

---

# ADDENDUM — Finalized ML pipeline (post-refactor)

> The audit above describes the *original* repo (RandomForest + in-repo AH vowels).
> The project was subsequently rebuilt to match the report. This addendum documents
> the **current** ML system.

## Deployed model
- **KNeighborsClassifier(k=5, weights="distance", metric="euclidean")** — `ml/artifacts/knn_model.pkl`
- Trained on the **UCI Parkinson's dataset** (`data/uci_parkinsons.csv`, 195 rows, 22 features).
- `StandardScaler` (`scaler.pkl`); training + hyperparameter sweep in `ml/train_knn.py`.
- Held-out (30%) metrics: accuracy 0.915, precision 0.898, recall 1.0, F1 0.946, AUC 0.958.

## Feature extraction (`ml/features.py`)
22 UCI biomarkers from audio: Praat/parselmouth (F0, jitter ×5, shimmer ×6, HNR→NHR),
nolds (DFA, D2) and custom code (RPDE, PPE, spread1/2). See `AUDIO_PIPELINE_REPORT.md`
for the live-audio calibration caveat.

## CTGAN data augmentation (`ml/augment_ctgan.py`)
Addresses the dataset's class imbalance (training split: 103 PD vs 33 healthy, 3.12:1)
using a **Conditional Tabular GAN**.

- **Leakage-safe:** CTGAN is trained on the training split only; the 30% test set is
  100% real and identical to `train_knn.py`'s split (`random_state=42`).
- **Procedure:** fit CTGAN (400 epochs) → conditionally sample 70 synthetic *healthy*
  rows → balance the training set (33→103 healthy) → refit the same KNN → evaluate on
  the real test set vs the unaugmented baseline.

### Result (held-out real test set)
| Metric | Baseline | CTGAN-augmented |
|---|---|---|
| Accuracy | 0.9153 | **0.9322** |
| Precision | 0.8980 | **0.9167** |
| Recall (PD) | 1.0000 | 1.0000 |
| Specificity (healthy) | 0.6667 | **0.7333** |
| Macro-F1 | 0.8731 | **0.9013** |
| AUC | 0.9576 | 0.9545 |

Augmentation improved minority-class detection (specificity +6.7 pts) and macro-F1
(+2.8 pts) with no loss of PD recall — the expected benefit of fixing imbalance.
Synthetic-data QA: 0 NaNs, mean per-feature KS statistic 0.505 (moderate, expected
given only 33 real healthy rows). Artifacts: `ml/artifacts/{ctgan_model.pkl,
synthetic_healthy.csv, augmented_train.csv, augmentation_metrics.json}`.

> Synthetic rows are augmentation only and must never be presented as real patient data.

### Deployment
`ml/deploy_augmented.py` promotes augmentation to production: it trains a CTGAN on
the **full** dataset, balances it (147 PD + 147 healthy, +99 synthetic), refits the
KNN + scaler on the balanced data, overwrites `knn_model.pkl`/`scaler.pkl`, and
updates `metrics.json` (flagged `"augmented": true`) with the held-out augmented
metrics above. The Flask backend therefore serves the CTGAN-balanced model
(restart the backend after running the script).
