"""Extract the 22 UCI Parkinson's acoustic biomarkers from a voice recording.

This is the inference-time counterpart to training on the UCI dataset. Given an
uploaded sustained-vowel WAV, it produces a feature vector in the exact order of
`feature_spec.FEATURE_NAMES`, so it can be scaled and fed to the KNN model.

Linear voice-quality measures (F0, jitter, shimmer, NHR, HNR) are computed with
Praat via parselmouth. The nonlinear dynamical measures (RPDE, DFA, spread1,
spread2, D2, PPE) follow Little et al. (2007, 2009) using documented
approximations (nolds for DFA / correlation dimension, custom code for the rest).

IMPORTANT CALIBRATION CAVEAT
----------------------------
The UCI dataset features were produced by Kay Pentax MDVP and Little's MATLAB
pipeline. Praat/Python re-implementations are *correlated* but not numerically
identical, especially for the nonlinear measures. Predictions on freshly recorded
audio are therefore approximate and must not be used for clinical decisions. This
limitation is inherent to the report's design (train on UCI, infer on new audio),
not a defect of this implementation.
"""
import numpy as np
import librosa

try:
    import parselmouth
    from parselmouth.praat import call
    _PARSELMOUTH = True
except Exception:  # pragma: no cover
    _PARSELMOUTH = False

try:
    import nolds
    _NOLDS = True
except Exception:  # pragma: no cover
    _NOLDS = False

from feature_spec import FEATURE_NAMES

F0_MIN = 75.0   # Praat default pitch floor (Hz)
F0_MAX = 600.0  # pitch ceiling for sustained phonation


# --------------------------------------------------------------------------- #
# Audio loading (format-robust)
# --------------------------------------------------------------------------- #
def _load_audio(path):
    """Load mono audio as (samples, sample_rate).

    Tries librosa/soundfile first (handles WAV/FLAC/OGG, and MP3/WebM where ffmpeg
    is available), then falls back to Praat's own reader. Returns float64 samples.
    """
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        if y is not None and y.size > 0:
            return y.astype(np.float64), int(sr)
    except Exception:
        pass
    snd = parselmouth.Sound(path)  # last resort: Praat-readable formats only
    return snd.values[0].astype(np.float64), int(snd.sampling_frequency)


# --------------------------------------------------------------------------- #
# Linear voice-quality measures (Praat / parselmouth)
# --------------------------------------------------------------------------- #
def _praat_measures(snd):
    pitch = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]  # voiced frames only

    point_process = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)

    def jit(kind):
        return call(point_process, f"Get jitter ({kind})", 0, 0, 0.0001, 0.02, 1.3)

    def shim(kind):
        return call([snd, point_process], f"Get shimmer ({kind})",
                    0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    # NHR (noise-to-harmonics ratio) approximated from HNR in dB.
    nhr = 10 ** (-hnr / 10.0) if np.isfinite(hnr) else 0.0

    return {
        "MDVP:Fo(Hz)": float(np.mean(f0)) if f0.size else 0.0,
        "MDVP:Fhi(Hz)": float(np.max(f0)) if f0.size else 0.0,
        "MDVP:Flo(Hz)": float(np.min(f0)) if f0.size else 0.0,
        "MDVP:Jitter(%)": float(jit("local")),
        "MDVP:Jitter(Abs)": float(jit("local, absolute")),
        "MDVP:RAP": float(jit("rap")),
        "MDVP:PPQ": float(jit("ppq5")),
        "Jitter:DDP": float(jit("ddp")),
        "MDVP:Shimmer": float(shim("local")),
        "MDVP:Shimmer(dB)": float(shim("local_dB")),
        "Shimmer:APQ3": float(shim("apq3")),
        "Shimmer:APQ5": float(shim("apq5")),
        "MDVP:APQ": float(shim("apq11")),
        "Shimmer:DDA": float(shim("dda")),
        "NHR": float(nhr),
        "HNR": float(hnr) if np.isfinite(hnr) else 0.0,
        "_f0": f0,
    }


# --------------------------------------------------------------------------- #
# Nonlinear dynamical measures
# --------------------------------------------------------------------------- #
def _rpde(signal, dim=4, tau=None, n_bins=100):
    """Recurrence Period Density Entropy (normalized), Little et al. 2007.

    Embed the signal, find the distribution of recurrence periods around a close
    return, then take the normalized Shannon entropy of that distribution.
    """
    x = signal[:: max(1, len(signal) // 20000)]  # subsample for tractability
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    if tau is None:
        tau = 1
    n = len(x) - (dim - 1) * tau
    if n < 100:
        return 0.0
    emb = np.array([x[i:i + n] for i in range(0, dim * tau, tau)]).T
    # radius = fraction of attractor size
    radius = 0.12 * np.sqrt(dim)
    periods = []
    step = max(1, n // 2000)
    for i in range(0, n, step):
        d = np.sqrt(np.sum((emb[i + 1:] - emb[i]) ** 2, axis=1))
        close = np.where(d < radius)[0]
        if close.size:
            periods.append(close[0] + 1)  # first recurrence period
    if not periods:
        return 0.0
    hist, _ = np.histogram(periods, bins=n_bins, range=(1, max(periods) + 1))
    p = hist[hist > 0] / hist.sum()
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(n_bins))  # normalize to [0, 1]


def _ppe(f0):
    """Pitch Period Entropy, Little et al. 2009.

    Map F0 to a perceptual (semitone) scale relative to its median, whiten with a
    short linear-prediction filter to remove normal healthy variation, then take
    the normalized entropy of the residual distribution.
    """
    if f0.size < 20:
        return 0.0
    semitone = 12.0 * np.log2(np.clip(f0, 1e-6, None) / (np.median(f0) + 1e-12))
    # whiten with a 2nd-order linear predictor
    resid = semitone[2:] - 2 * semitone[1:-1] + semitone[:-2]
    hist, _ = np.histogram(resid, bins=50, density=False)
    p = hist[hist > 0] / hist.sum()
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(50))


def _spread(f0):
    """spread1 / spread2: nonlinear measures of fundamental-frequency variation.

    spread1 follows the dysphonia literature as a log-scaled dispersion of F0;
    spread2 captures the standard deviation of the semitone-mapped contour.
    """
    if f0.size < 5:
        return 0.0, 0.0
    semitone = 12.0 * np.log2(np.clip(f0, 1e-6, None) / (np.median(f0) + 1e-12))
    spread1 = float(np.log((np.std(f0) / (np.mean(f0) + 1e-12)) + 1e-12))
    spread2 = float(np.std(semitone))
    return spread1, spread2


def _nonlinear_measures(signal, f0):
    rpde = _rpde(signal)
    if _NOLDS:
        sub = signal[:: max(1, len(signal) // 10000)]
        sub = (sub - np.mean(sub)) / (np.std(sub) + 1e-12)
        try:
            dfa = float(nolds.dfa(sub))
        except Exception:
            dfa = 0.0
        try:
            d2 = float(nolds.corr_dim(sub, emb_dim=6))
        except Exception:
            d2 = 0.0
    else:
        dfa, d2 = 0.0, 0.0
    spread1, spread2 = _spread(f0)
    ppe = _ppe(f0)
    return {
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_features(path):
    """Return a dict {feature_name: value} for the 22 UCI biomarkers."""
    if not _PARSELMOUTH:
        raise RuntimeError("praat-parselmouth is required for feature extraction")

    y, sr = _load_audio(path)
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    measures = _praat_measures(snd)
    f0 = measures.pop("_f0")
    # Nonlinear measures operate on a 22.05 kHz signal for consistency.
    signal = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    measures.update(_nonlinear_measures(signal, f0))

    # sanitize: replace NaN/inf with 0.0 (KNN scaler cannot accept NaN)
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in measures.items()}


def extract_feature_vector(path):
    """Return a 1x22 numpy array in FEATURE_NAMES order."""
    feats = extract_features(path)
    return np.array([[feats[name] for name in FEATURE_NAMES]], dtype=float)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import json
        print(json.dumps(extract_features(sys.argv[1]), indent=2))
    else:
        print("usage: python ml/features.py <audio.wav>")
