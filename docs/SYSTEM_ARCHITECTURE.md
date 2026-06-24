# SYSTEM_ARCHITECTURE.md — Proposed Final Architecture

> Audit date: 2026-06-21 · This is a **design proposal** (Phase 7). No code changed.

## Stack decision

| Layer | Target (brief) | Recommendation | Reasoning |
|---|---|---|---|
| Frontend | React + Vite + Tailwind | ✅ Keep [frontend/](frontend/) | Already React 18 + Vite + Tailwind; just needs page wiring + audio fixes |
| Backend | Flask | ⚠️ **Keep FastAPI** (or port to Flask) | Working backend is FastAPI w/ WebSocket; Flask has no native WS. Decision needed — see below |
| ML | scikit-learn | ✅ Random Forest already in sklearn | Keep RF; optionally honor report's KNN as a selectable model |
| Database | SQLite | ➕ **Add** (does not exist yet) | For prediction history; report claims it, code lacks it |
| Deployment | Docker | ➕ **Add** | No Dockerfile/compose currently exists |

> **Backend mismatch to resolve:** the brief says Flask but the only working server is FastAPI. **Recommended: keep FastAPI** (real-time WebSocket already implemented, async, Pydantic). If the academic report's "Flask" claim is binding, the file-upload REST flow can be served by Flask while losing live streaming. The design below is framework-light so it applies either way.

## Target folder structure

```
parkinsons-detection/
├── backend/                        # FastAPI (survivor)
│   ├── app/
│   │   ├── main.py                 # app + router registration
│   │   ├── api/
│   │   │   ├── health.py
│   │   │   ├── results.py          # serves metrics.json
│   │   │   ├── predict.py          # REST file-upload prediction
│   │   │   └── ws.py               # /ws/predict streaming
│   │   ├── ml/
│   │   │   ├── features.py         # extract_features (single source of truth)
│   │   │   ├── predictor.py        # LivePredictor + batch predict
│   │   │   └── artifacts/          # rf_model.pkl, scaler.pkl, selected_features.pkl, metrics.json
│   │   ├── db/
│   │   │   ├── database.py         # SQLite engine/session
│   │   │   └── models.py           # PredictionHistory
│   │   └── core/config.py          # paths, CORS, settings
│   ├── requirements.txt            # pinned
│   └── Dockerfile
├── frontend/                       # React + Vite + Tailwind (survivor)
│   └── src/{pages,components,lib}
├── ml/                             # training (not shipped in prod image)
│   ├── train.py                    # ex-parkinsons_detection.py, leakage-fixed
│   ├── feature_selection/{gwo,abc,pso}.py
│   ├── augmentation.py             # ex-audio_augmentation.py
│   └── evaluate.py                 # writes metrics.json
├── data/                           # HC_AH/, PD_AH/, PD_AH_Augmented/, Demographics.xlsx
├── docs/                           # these audit .md files + report
├── docker-compose.yml
└── README.md
```

## API architecture (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/results` | model metrics from `metrics.json` (replaces hard-coded dict) |
| GET | `/api/features` | feature list (single source: `features.py`) |
| POST | `/api/predict` | upload WAV → one-shot prediction (serves the wizard UX) |
| GET | `/api/history` | recent predictions (SQLite) |
| WS | `/ws/predict` | stream Float32 audio → rolling prediction |

CORS restricted to the frontend origin. Artifact paths resolved absolutely from a config root (fixes the CWD bug).

## Database schema (SQLite)

```sql
CREATE TABLE prediction_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,           -- ISO timestamp
    source       TEXT    NOT NULL,           -- 'upload' | 'live'
    label        TEXT    NOT NULL,           -- 'Healthy' | "Parkinson's"
    pd_probability      REAL NOT NULL,
    healthy_probability REAL NOT NULL,
    client_ip    TEXT,
    filename     TEXT
);
```
(Optional `users` table only if authentication is actually wanted — currently neither claimed feature exists.)

## ML pipeline (leakage-fixed)

```
WAV ─▶ preprocess (resample 22050, trim silence, normalize, optional denoise)
    ─▶ extract_features  (52 librosa features  [+ optional jitter/shimmer/HNR/RPDE/DFA/PPE])
    ─▶ apply selected_features mask (→17)
    ─▶ StandardScaler.transform
    ─▶ RandomForest.predict / predict_proba
    ─▶ {label, pd_probability, healthy_probability}
```
Training: 3-way split (train / selection-val / **held-out test**); feature selection on train only; metrics from untouched test written to `metrics.json`; group-aware split if augmented data is used.

## Deployment strategy (Docker)

- `backend/Dockerfile`: python:3.11-slim, install pinned reqs (librosa needs `libsndfile1`/`ffmpeg`), copy `app/` + artifacts, `uvicorn app.main:app`.
- `frontend/Dockerfile`: node build → static served by nginx (or Vite preview).
- `docker-compose.yml`: `frontend` (nginx :80, proxies `/api` + `/ws` to backend), `backend` (:8000) with a mounted volume for the SQLite file.
- Single source of truth for the feature list and metrics shared between training and serving via the `ml/` → `artifacts/` handoff.
