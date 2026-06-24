"""Canonical feature specification for the Parkinson's KNN model.

Single source of truth shared by the training script (ml/train_knn.py) and the
inference-time audio feature extractor (ml/features.py). The order MUST match the
order used to fit the scaler/model, so both training and serving import this list.

These are the 22 acoustic biomarkers of the UCI Parkinson's dataset (Little et al.):
fundamental-frequency, jitter, shimmer, noise (NHR/HNR) and nonlinear dynamical
measures (RPDE, DFA, spread1, spread2, D2, PPE).
"""

# Order is the UCI column order with `status` (the label) removed.
FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

LABEL_COLUMN = "status"   # 1 = Parkinson's, 0 = healthy
ID_COLUMN = "name"        # e.g. phon_R01_S01_1  -> subject = S01

N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 22
