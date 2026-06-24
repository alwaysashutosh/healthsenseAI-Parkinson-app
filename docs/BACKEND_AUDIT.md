# BACKEND_AUDIT.md

> Audit date: 2026-06-21

There are **two** backends in the repo.

| | [backend/](backend/) | [parkinsons-backend/](parkinsons-backend/) |
|---|---|---|
| Framework | **FastAPI** + uvicorn | **Flask 1.0.2** + gunicorn |
| Port | 8000 | 5000 (gunicorn default) |
| Inference | Imports real `LivePredictor` from [parkinsons_detection.py](parkinsons_detection.py) → **Random Forest** | `pyAudioAnalysis` + broken legacy pickle, plus a **random** demo path |
| Transport | REST (`/api/*`) + **WebSocket** (`/ws/predict`) | REST (file upload + polling) |
| DB | None | SQLAlchemy → **PostgreSQL** (`os.environ['DATABASE_URL']`, `psycopg2`) |
| Model file | `../rf_model.pkl` (17-feature RF) | `Parkinson/model_parkinson.pkl` (1.6 KB, **won't load**) |
| Matches target stack (Flask + SQLite) | ❌ It's FastAPI, no DB | ⚠️ Flask but Postgres + broken deps |

## Which is latest / which should survive

- **Latest & functional: [backend/](backend/) (FastAPI).** It is the counterpart to the kept `frontend/` and to the most recent commit's real-time work. It loads the actual trained Random Forest.
- **[parkinsons-backend/](parkinsons-backend/) is the legacy "Apmycure" prototype** — old Flask, `pyAudioAnalysis` (Python-2-era, `audioFeatureExtraction` no longer exists in current versions), Postgres, and a model pickle that fails to deserialize.
- ➡️ **Survivor: `backend/` (FastAPI).** Archive/remove `parkinsons-backend/`.

> ⚠️ **Note vs. target stack:** the brief's target is **Flask**, but the working backend is **FastAPI**. Decision required (see [REFACTOR_PLAN.md](REFACTOR_PLAN.md) Phase 2): keep FastAPI (recommended — WebSocket support, already working) **or** port to Flask to match the report. This is an explicit architecture mismatch to resolve.

## backend/ (KEEP) — detailed checks

[main.py](backend/main.py):
- **Routes:**
  - `GET /api/health` ([:129](backend/main.py#L129)) — liveness.
  - `GET /api/results` ([:133](backend/main.py#L133)) — returns **hard-coded** `ALGORITHM_RESULTS` (76/72/74%), not live metrics.
  - `GET /api/features` ([:137](backend/main.py#L137)) — returns `ALL_FEATURES` (46-name list, order/length inconsistent with the 52-feature extractor — see [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md)).
  - `WS /ws/predict` ([:141](backend/main.py#L141)) — receives binary Float32 chunks → `LivePredictor.process_audio_chunk` → `predict_live` → JSON back.
- **Model loading:** lazy singleton `get_predictor()` ([:29](backend/main.py#L29)). It computes `model_path`/`scaler_path` but **never passes them** — `LivePredictor()` loads from **CWD** ([parkinsons_detection.py:810](parkinsons_detection.py#L810), `joblib.load('scaler.pkl')` is relative). 🔴 So the server only finds the artifacts if launched from the repo root; running from `backend/` will fail to load `scaler.pkl`/`rf_model.pkl`.
- **CORS:** `allow_origins=["*"]` ([:20](backend/main.py#L20)) — fine for dev, tighten for prod.
- **Audio processing:** delegates to `extract_features`/`LivePredictor`. Inherits the **22050 Hz assumption** while the browser sends ~48 kHz (see [FRONTEND_AUDIT.md](FRONTEND_AUDIT.md)).
- **Error handling:** WebSocket wrapped in try/except for disconnect + generic errors ([:171](backend/main.py#L171)); reasonable. REST endpoints have none (no validation/404 bodies).
- **Logging:** `print()` only; no structured logging.
- **Database:** none. The report's SQLite history layer does not exist here.
- **No requirements pin / Dockerfile** (deps unpinned in [requirements.txt](backend/requirements.txt)).

## parkinsons-backend/ (ARCHIVE) — detailed checks

- [app.py](parkinsons-backend/app.py): Flask + `CORS`; **requires `os.environ['DATABASE_URL']`** at import → crashes on startup if unset.
- [routes.py](parkinsons-backend/routes.py):
  - `/upload` ([:65](parkinsons-backend/routes.py#L65)) → real path via `Parkinson.ParkinsonCheck.predict`.
  - `/upload-demo` ([:80](parkinsons-backend/routes.py#L80)) → **random** `randint(30,80)` ([:30](parkinsons-backend/routes.py#L30)) — what the old frontend actually calls.
  - `/done`, `/getresult`, `/history` — polling + per-IP history.
  - Uses **module-global `done`/`result`** mutated by background threads → not concurrency-safe (race conditions across users).
  - `secure_filename` + `f.save(filename)` saves to CWD then `os.remove` — path/cleanup fragility.
- [Parkinson/ParkinsonCheck.py](parkinsons-backend/Parkinson/ParkinsonCheck.py): 🔴 **broken ML** — fits a fresh `StandardScaler` and `LabelEncoder` on a **single inference sample** ([:11–18](parkinsons-backend/Parkinson/ParkinsonCheck.py#L11)) and even label-encodes the feature values. This cannot produce meaningful predictions.
- [Parkinson/audio_analyzer.py](parkinsons-backend/Parkinson/audio_analyzer.py): `from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction` — `audioFeatureExtraction` was **removed** in modern pyAudioAnalysis → **import error** on any current install.
- [models.py](parkinsons-backend/models.py): SQLAlchemy `file_results` (composite PK time+ip). Postgres-oriented.
- [requirements.txt](parkinsons-backend/requirements.txt): heavily pinned to 2019-era versions (Flask 1.0.2, numpy 1.16.2, scikit-learn 0.20.3, psycopg2). Not installable cleanly on Python 3.11+.
- `requirements-new.txt` + `Procfile` (`web: gunicorn app:app`) indicate a prior Heroku-style deploy.

## Broken imports / dependency issues (cross-cutting)
- 🔴 `parkinsons-backend/Parkinson/audio_analyzer.py` — `pyAudioAnalysis.audioFeatureExtraction` no longer exists.
- 🔴 `parkinsons-backend/Parkinson/model_parkinson.pkl` — unpicklable (`sklearn.tree.tree` gone).
- 🔴 `parkinsons-backend/app.py` — hard dependency on `DATABASE_URL` env var.
- 🟡 `backend` artifact paths resolved relative to CWD, not to the file → run-location-sensitive.
- 🟡 All `requirements.txt` files are **unpinned** (root, backend) or **stale** (parkinsons-backend). No lockfile for Python.

## Recommendations (no code changes yet)
1. Keep `backend/` (FastAPI); archive `parkinsons-backend/`.
2. Decide FastAPI-vs-Flask for the final stack (architecture mismatch with the report/brief).
3. Make model paths absolute (relative to project root) so the server runs from any CWD.
4. Add the missing **SQLite** persistence + `/api/history` if history tracking is a requirement (it's in the report but not in `backend/`).
5. Serve `/api/results` from a real `metrics.json`, not hard-coded dicts.
6. Pin dependencies + add a Dockerfile.
