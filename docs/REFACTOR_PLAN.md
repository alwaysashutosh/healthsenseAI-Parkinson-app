# REFACTOR_PLAN.md

> Audit date: 2026-06-21 · **No code has been modified.** This is the staged plan only.
> Companion docs: [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md) · [FRONTEND_AUDIT.md](FRONTEND_AUDIT.md) · [BACKEND_AUDIT.md](BACKEND_AUDIT.md) · [AUDIO_PIPELINE_REPORT.md](AUDIO_PIPELINE_REPORT.md) · [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)

Guiding rules: preserve working code, prefer the latest implementation (`frontend/` + `backend/`), nothing deleted without being archived first.

## Phase 1 — Cleanup
- Move legacy prototype to `archive/`: [parkinsons-backend/](parkinsons-backend/), [parkinsons-frontend/](parkinsons-frontend/).
- Remove scratch/empty files: [Untitled4.ipynb](Untitled4.ipynb) (empty), `~$Demographics_age_sex.xlsx` (Excel lock file), [test_augmented.wav](test_augmented.wav) (scratch output).
- Populate the empty [README.md](README.md) with real run instructions.
- Add `.gitignore` for `node_modules/`, `__pycache__/`, `*.pkl` cache, SQLite db, build dirs.
- Decide whether `data/` (WAVs) belongs in git or in Git LFS / external storage.

## Phase 2 — Consolidation
- Adopt the folder structure in [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md): `backend/`, `frontend/`, `ml/`, `data/`, `docs/`.
- **Resolve the framework mismatch** (FastAPI vs report's Flask) — recommend keeping FastAPI.
- Make a **single source of truth** for `extract_features` and the feature-name list (shared by training + serving); reconcile the 43-vs-52 discrepancy across UIs.
- Move `parkinsons_detection.py`, `audio_augmentation.py` into `ml/`; keep `LivePredictor`/`features` import path stable for `backend/`.

## Phase 3 — Backend completion
- Fix artifact paths to resolve from project root, not CWD ([BACKEND_AUDIT.md](BACKEND_AUDIT.md)).
- Add `POST /api/predict` (one-shot upload) to support a wizard UX.
- Add **SQLite** persistence + `GET /api/history`.
- Serve `/api/results` from `metrics.json` instead of hard-coded dicts.
- Restrict CORS; add structured logging + request validation.
- Pin `requirements.txt`.

## Phase 4 — Frontend completion
- Keep `frontend/`; wire `Feature Analysis`, `Model Performance`, `About` into [App.jsx](frontend/src/App.jsx) (or drop them from the sidebar) — currently they silently render the Dashboard.
- Build out the two stub pages (port the Streamlit analytics views).
- Fix WebSocket: resample to 22050 Hz and route via the Vite `/ws` proxy; migrate `createScriptProcessor` → `AudioWorklet`.
- Replace hard-coded dashboard metrics with `/api/results`.
- (Optional) port the patient-form → upload → result wizard from the archived CRA app onto the real backend.

## Phase 5 — ML improvements
- **Fix data leakage** (3-way split / nested CV; select features on train only; report on held-out test) — see [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md).
- Write real metrics to `metrics.json`; stop hard-coding accuracy.
- Decide the dataset/biomarker story:
  - (a) Keep AH-vowel dataset + current 52 features → **rewrite the report** to match, **or**
  - (b) Implement jitter/shimmer/HNR/RPDE/DFA/PPE (parselmouth/nolds) and/or integrate the UCI dataset → **match the report**.
- Group-aware split if `PD_AH_Augmented` is included; augment HC too for balance.
- Re-export the coupled artifacts together and version them with the sklearn version + feature list.

## Phase 6 — Testing
- `pytest` for `extract_features` (shape=52, determinism), predictor (mask→17→scale→predict), API routes (TestClient), DB writes.
- Replace [test_augmentation.py](test_augmentation.py) (absolute path, not a test) with real unit tests.
- Frontend smoke test + `npm run build` in CI.
- GitHub Actions: lint + test + build on push.

## Phase 7 — Deployment
- `backend/Dockerfile` (python:3.11-slim + `libsndfile1`/`ffmpeg`), `frontend/Dockerfile` (node build → nginx).
- `docker-compose.yml` wiring frontend ↔ backend, SQLite volume.
- Production env config (CORS origins, artifact paths, DB path); health-check wiring.
- Update README with Docker quickstart.
