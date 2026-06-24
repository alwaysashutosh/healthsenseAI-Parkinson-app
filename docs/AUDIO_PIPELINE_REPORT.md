# AUDIO_PIPELINE_REPORT.md

> Audit date: 2026-06-21

## Scope
Covers the augmentation scripts ([audio_augmentation.py](audio_augmentation.py),
[demo_augmentation.py](demo_augmentation.py), [test_augmentation.py](test_augmentation.py)) and the
feature-extraction / preprocessing path ([parkinsons_detection.py](parkinsons_detection.py) `extract_features`,
`LivePredictor`).

## A. Augmentation pipeline ([audio_augmentation.py](audio_augmentation.py))

Implemented, working, CLI-driven. Techniques (each a function, randomly combined 2–4 at a time):

| Technique | Function | Status |
|---|---|---|
| Additive noise | `add_noise` ([:11](audio_augmentation.py#L11)) | ✅ Implemented |
| Time stretch | `time_stretch` ([:19](audio_augmentation.py#L19)) | ✅ librosa |
| Pitch shift | `pitch_shift` ([:23](audio_augmentation.py#L23)) | ✅ librosa |
| Time shift | `time_shift` ([:27](audio_augmentation.py#L27)) | ✅ |
| Speed change | `change_speed` ([:38](audio_augmentation.py#L38)) | ✅ (resample-based) |
| Volume change | `change_volume` ([:45](audio_augmentation.py#L45)) | ✅ |
| Low-pass filter | `apply_low_pass_filter` ([:49](audio_augmentation.py#L49)) | ✅ (defensive butter unpacking) |
| High-pass filter | `apply_high_pass_filter` ([:63](audio_augmentation.py#L63)) | ✅ |

- `generate_multiple_samples` ([:132](audio_augmentation.py#L132)) batch-augments a directory; it produced [PD_AH_Augmented/](PD_AH_Augmented/) (`aug_1_*`, `aug_2_*`).
- `demo_augmentation.py` / `test_augmentation.py` are **demo/scratch scripts**; `test_augmentation.py` hard-codes an absolute Windows path ([test_augmentation.py:13](test_augmentation.py#L13)) and is not a real unit test.
- 🟡 **Augmented data is never used by training** — `main()` only loads `PD_AH/PD_AH` + `HC_AH/HC_AH`. The augmentation effort is currently orphaned.
- 🟡 Augmenting **only the PD class** would skew class balance if added without also augmenting HC.

## B. Preprocessing — claimed vs. actual

The report ([Project_Report…txt:140](Project_Report_Parkinsons_Detection.txt#L140)) claims "noise reduction, normalization, and framing."

| Step | Report | Actual code |
|---|---|---|
| Resampling | — | ✅ `librosa.load(sr=22050)` |
| Noise reduction | ✅ claimed | ❌ Not implemented |
| Normalization | ✅ claimed | ❌ No loudness/peak normalization (only `StandardScaler` on *features*, not audio) |
| Framing/windowing | ✅ claimed | ➖ Implicit inside librosa feature functions only |
| Silence/VAD | — | 🟡 Crude RMS<0.01 gate in `predict_live` ([parkinsons_detection.py:861](parkinsons_detection.py#L861)) |

## C. Feature extraction — claimed biomarkers vs. actual

The report is built around clinical dysphonia biomarkers. **Almost none are implemented.** Actual features come from `extract_features` ([parkinsons_detection.py:30](parkinsons_detection.py#L30)).

| Feature (report) | Implemented? | Notes |
|---|---|---|
| **MFCC** | ✅ Implemented | 13 mean + 13 std ([:51](parkinsons_detection.py#L51)) |
| **Jitter** (F0 perturbation) | ❌ **Missing** | Report describes local/RAP jitter ([:655](Project_Report_Parkinsons_Detection.txt#L655)); no code computes F0 cycles |
| **Shimmer** (amplitude perturbation) | ❌ **Missing** | ([report :659](Project_Report_Parkinsons_Detection.txt#L659)) |
| **HNR** (harmonics-to-noise) | ❌ **Missing** | ([report :663](Project_Report_Parkinsons_Detection.txt#L663)) |
| **RPDE** | ❌ **Missing** | nonlinear measure; not implemented |
| **DFA** | ❌ **Missing** | not implemented |
| **PPE** | ❌ **Missing** | not implemented |
| Spectral centroid/rolloff/bandwidth | ✅ Implemented (extra, not in report) | ([:56](parkinsons_detection.py#L56)) |
| ZCR, RMS | ✅ Implemented (extra) | ([:62](parkinsons_detection.py#L62)) |
| Chroma (12) | ✅ Implemented (extra) | ([:69](parkinsons_detection.py#L69)) |
| Spectral contrast (7) | ✅ Implemented (extra) | ([:74](parkinsons_detection.py#L74)) |
| Spectral flatness, Tempo | ✅ Implemented (extra) | ([:78](parkinsons_detection.py#L78)) |

### Per-biomarker verdict
- **Implemented:** MFCC (mean+std), spectral (centroid/rolloff/bandwidth/contrast/flatness), ZCR, RMS, chroma, tempo.
- **Partially implemented:** preprocessing (only resample; no denoise/normalize); silence handling (RMS gate only).
- **Missing:** jitter, shimmer, HNR, RPDE, DFA, PPE — i.e. **every clinical biomarker the report centers on**.
- **Incorrect / risky:**
  - Sample-rate mismatch: extractor + `LivePredictor` assume 22050 Hz, but the live browser stream is ~48 kHz (see [FRONTEND_AUDIT.md](FRONTEND_AUDIT.md)) → spectral/tempo features computed on wrong-rate audio at inference.
  - `tempo`/`beat_track` on sustained-vowel /a/ phonation is near-meaningless as a discriminative feature.

## D. Recommendations (no code changes yet)
1. If the report must stand: implement jitter/shimmer/HNR/RPDE/DFA/PPE (e.g. **Praat via `parselmouth`**, or `nolds` for RPDE/DFA/PPE) and retrain. This is the single biggest code-vs-report gap.
2. Add real preprocessing: trim silence, peak/loudness-normalize, optional spectral-gating denoise — before `extract_features`.
3. Fix the inference sample-rate (resample live audio to 22050 Hz).
4. Either wire `PD_AH_Augmented` into training **with group-aware splitting** and matching HC augmentation, or delete it.
5. Promote/replace `test_augmentation.py` with a real `pytest` test (no absolute paths).
