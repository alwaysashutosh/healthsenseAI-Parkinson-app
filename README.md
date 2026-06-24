# Parkinson's Disease Detection Using Speech Analysis

A web-based screening system that detects Parkinson's disease from a sustained-vowel
voice recording. It matches the architecture described in the project report:

- **Frontend:** React + Vite + Tailwind (`frontend/`)
- **Backend:** Flask REST API + JWT auth + **role-based access control** + SQLite (`backend/`)
- **ML:** K-Nearest Neighbors (k=5, Euclidean, distance-weighted) trained on the
  **UCI Parkinson's dataset** (195 recordings, 31 subjects, 22 acoustic biomarkers) (`ml/`)
- **Deployment:** Docker + docker-compose

## Architecture

```
React (Vite, :3000)  ──/api──►  Flask (:5000)  ──►  KNN model + StandardScaler
   role-routed UI:                  │                (ml/artifacts/*.pkl)
   • patient portal                 ├──► feature extraction (Praat/parselmouth + nolds)
   • doctor portal                  ├──► SQLite (users, patients, doctors,
   • admin console                  │           predictions, appointments)
                                    └──► OpenStreetMap (Nominatim + Overpass) hospital finder
```

## Roles (RBAC)

The platform is a patient↔doctor product with three roles, enforced on every API
route via a `role_required` decorator and JWT:

| Role | Can | Notes |
|---|---|---|
| **Patient** | Register/login, edit profile, run voice test, view own reports + doctor notes, **find nearby hospitals (OSM)**, **book appointments** with verified doctors | self-registers |
| **Doctor** | Login, view patients, review reports, add diagnosis notes, **manage appointment requests** (approve/complete/cancel) | self-registers but **requires admin approval** before access |
| **Admin** | Approve/revoke doctors, list users, view platform stats | seeded on first run (not self-registerable) |

A default admin is created on first startup — **username `admin`, password `admin12345`**
(override with `ADMIN_USERNAME` / `ADMIN_PASSWORD`; change it in production).

## Machine learning

| | |
|---|---|
| Classifier | `KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")` |
| Dataset | UCI Parkinson's (`data/uci_parkinsons.csv`, 147 PD / 48 healthy) |
| Features | 22 biomarkers: F0 (mean/hi/lo), jitter ×5, shimmer ×6, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE |
| Held-out test | accuracy 0.915 · precision 0.898 · recall 1.00 · F1 0.946 · AUC 0.958 |

Train / retrain:

```bash
pip install -r requirements.txt
python ml/train_knn.py        # baseline KNN -> ml/artifacts/{knn_model,scaler}.pkl + metrics.json
python ml/augment_ctgan.py    # CTGAN augmentation experiment (held-out baseline vs augmented)
python ml/deploy_augmented.py # promote the CTGAN-balanced model to production
```

Inference-time feature extraction from audio uses **Praat (parselmouth)** for the
linear voice-quality measures and **nolds**/custom code for the nonlinear measures.

### CTGAN data augmentation (class imbalance)
The dataset is imbalanced (~3:1 PD:healthy). `ml/augment_ctgan.py` trains a
**Conditional Tabular GAN** on the training split only (test set stays 100% real),
generates synthetic *healthy* rows to balance training, and re-evaluates the KNN.
On the held-out real test set this lifted **specificity 66.7%→73.3%** and
**macro-F1 0.873→0.901** with no loss of PD recall. Details + per-feature synthetic
QA in `ml/artifacts/augmentation_metrics.json` and `docs/ML_ARCHITECTURE.md`.

`ml/deploy_augmented.py` then promotes this to production: it retrains the final
model on the CTGAN-balanced **full** dataset (147 PD + 147 healthy, +99 synthetic)
and refreshes `metrics.json` with the held-out augmented numbers. **This is the
model the backend currently serves.** Restart the backend after running it.

> ⚠️ **Calibration caveat.** The UCI features were produced by Kay Pentax MDVP +
> Little's MATLAB pipeline. Open re-implementations are *correlated* but not
> numerically identical (especially the nonlinear measures), so predictions on
> freshly recorded audio are approximate. This is a screening aid, **not** a
> medical diagnosis. See `AUDIO_PIPELINE_REPORT.md`.

## Run locally (without Docker)

**Backend** (terminal 1):
```bash
pip install -r backend/requirements.txt
cd backend
python app.py            # http://localhost:5000
```

**Frontend** (terminal 2):
```bash
cd frontend
npm install
npm run dev              # http://localhost:3000  (proxies /api -> :5000)
```

Open http://localhost:3000 → register as a **patient** → **New Test** → record/upload "ahhh"
→ result saved to your reports. Log in as `admin`/`admin12345` to approve doctor accounts.

## Run with Docker

```bash
docker compose up --build
# frontend: http://localhost:3000   backend: http://localhost:5000
```

Set `SECRET_KEY` / `JWT_SECRET` env vars for production.

## API

| Method | Endpoint | Role | Purpose |
|---|---|---|---|
| POST | `/api/auth/register` | – | Create patient/doctor account → JWT |
| POST | `/api/auth/login` | – | Login → JWT |
| GET | `/api/auth/me` | any | Current user + profile |
| GET | `/api/health` `/results` `/features` | – | Public: status, model metrics, feature catalogue |
| GET/PUT | `/api/patient/profile` | patient | View / update profile |
| POST | `/api/patient/predict` | patient | Upload audio → prediction (saved to reports) |
| GET | `/api/patient/reports` | patient | Own screening history |
| GET | `/api/patient/doctors` | patient | Verified doctors available to book |
| GET/POST | `/api/patient/appointments` | patient | List / book appointments |
| POST | `/api/patient/appointments/<id>/cancel` | patient | Cancel an appointment |
| GET | `/api/hospitals?city=&radius=` | any | Nearby hospitals/clinics via OpenStreetMap |
| GET | `/api/doctor/patients` | doctor* | List patients + latest result |
| GET | `/api/doctor/patients/<id>/reports` | doctor* | A patient's reports |
| POST | `/api/doctor/reports/<id>/notes` | doctor* | Add diagnosis notes |
| GET | `/api/doctor/appointments` | doctor* | Appointment requests |
| POST | `/api/doctor/appointments/<id>/status` | doctor* | Approve / complete / cancel |
| GET | `/api/admin/stats` `/users` `/doctors` | admin | Platform analytics, user/doctor lists |
| POST | `/api/admin/doctors/<id>/approve` `/revoke` | admin | Verify / unverify a doctor |

\* doctor routes require an **approved** (admin-verified) doctor account.

## Repository layout

```
frontend/        React + Vite + Tailwind app
backend/         Flask API (app factory, auth, models, ml_service, routes)
ml/              feature_spec, features (extraction), train_knn, artifacts/
data/            uci_parkinsons.csv
archive/         retired prototypes (old FastAPI/Flask/CRA code, kept for reference)
docs/            repository audit reports (ML_ARCHITECTURE.md, etc.)
HC_AH/ PD_AH/     original sustained-vowel recordings (legacy dataset, not used by KNN)
```

## Audit reports
See `docs/`: `ML_ARCHITECTURE.md`, `BACKEND_AUDIT.md`, `FRONTEND_AUDIT.md`,
`AUDIO_PIPELINE_REPORT.md`, `SYSTEM_ARCHITECTURE.md`, `REFACTOR_PLAN.md`.
