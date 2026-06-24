"""Public (unauthenticated) API: health, model metrics, feature catalogue."""
from flask import Blueprint, jsonify

from ml_service import ml_service

api_bp = Blueprint("api", __name__, url_prefix="/api")

FEATURE_DESCRIPTIONS = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
    "MDVP:Jitter(%)": "Frequency perturbation (relative)",
    "MDVP:Jitter(Abs)": "Frequency perturbation (absolute, s)",
    "MDVP:RAP": "Relative average perturbation",
    "MDVP:PPQ": "Five-point period perturbation quotient",
    "Jitter:DDP": "Avg abs difference of consecutive jitter",
    "MDVP:Shimmer": "Amplitude perturbation (relative)",
    "MDVP:Shimmer(dB)": "Amplitude perturbation (dB)",
    "Shimmer:APQ3": "Three-point amplitude perturbation quotient",
    "Shimmer:APQ5": "Five-point amplitude perturbation quotient",
    "MDVP:APQ": "Eleven-point amplitude perturbation quotient",
    "Shimmer:DDA": "Avg abs difference of consecutive shimmer",
    "NHR": "Noise-to-harmonics ratio",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Recurrence period density entropy (nonlinear)",
    "DFA": "Detrended fluctuation analysis (nonlinear)",
    "spread1": "Nonlinear F0 variation measure 1",
    "spread2": "Nonlinear F0 variation measure 2",
    "D2": "Correlation dimension (nonlinear)",
    "PPE": "Pitch period entropy (nonlinear)",
}


@api_bp.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": ml_service.ready})


@api_bp.get("/results")
def results():
    return jsonify(ml_service.metrics)


@api_bp.get("/features")
def features():
    return jsonify({
        "features": [
            {"name": n, "description": FEATURE_DESCRIPTIONS.get(n, "")}
            for n in ml_service.feature_names
        ]
    })
