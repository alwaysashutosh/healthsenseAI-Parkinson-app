"""ML inference service: loads the KNN artifacts and predicts from an audio file.

Bridges the Flask backend to the ml/ package (feature_spec, features, train_knn).
"""
import os
import sys
import json
import joblib
import numpy as np

from config import Config

# Make the ml/ package importable (feature_spec, features).
if Config.ML_DIR not in sys.path:
    sys.path.insert(0, Config.ML_DIR)

from feature_spec import FEATURE_NAMES  # noqa: E402


class MLService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.feature_names = FEATURE_NAMES
        self._extractor = None
        self._load()

    def _load(self):
        art = Config.ARTIFACTS_DIR
        model_path = os.path.join(art, "knn_model.pkl")
        scaler_path = os.path.join(art, "scaler.pkl")
        metrics_path = os.path.join(art, "metrics.json")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

    @property
    def ready(self):
        return self.model is not None and self.scaler is not None

    def _get_extractor(self):
        # Lazy import: parselmouth/librosa are heavy and only needed at predict time.
        if self._extractor is None:
            from features import extract_feature_vector
            self._extractor = extract_feature_vector
        return self._extractor

    def predict(self, audio_path):
        """Extract 22 biomarkers from audio_path and run the KNN classifier."""
        if not self.ready:
            raise RuntimeError("Model artifacts not loaded. Run ml/train_knn.py.")

        extractor = self._get_extractor()
        vector = extractor(audio_path)                 # 1 x 22
        scaled = self.scaler.transform(vector)
        pred = int(self.model.predict(scaled)[0])
        proba = self.model.predict_proba(scaled)[0]
        # classes_ is [0, 1]; index 1 == Parkinson's
        pd_prob = float(proba[list(self.model.classes_).index(1)])
        healthy_prob = float(proba[list(self.model.classes_).index(0)])

        return {
            "label": "Parkinson's" if pred == 1 else "Healthy",
            "prediction": pred,
            "pd_probability": pd_prob,
            "healthy_probability": healthy_prob,
            "probability": pd_prob if pred == 1 else healthy_prob,
            "features": {name: float(v) for name, v in zip(FEATURE_NAMES, vector[0])},
        }


# Singleton used across requests.
ml_service = MLService()
