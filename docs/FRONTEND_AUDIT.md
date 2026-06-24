# FRONTEND_AUDIT.md

> Audit date: 2026-06-21

There are **two** frontends in the repo.

| | [frontend/](frontend/) | [parkinsons-frontend/](parkinsons-frontend/) |
|---|---|---|
| Stack | **React 18 + Vite 5 + Tailwind 3** | React 19 + **Create React App (react-scripts 5)** |
| Routing | Local `useState` page switch ([src/App.jsx](frontend/src/App.jsx)) | `react-router-dom` v7 ([src/App.js](parkinsons-frontend/src/App.js)) |
| Purpose | Analytics dashboard + **live WebSocket** voice monitor | 3-step wizard: patient form → record/upload → result |
| Talks to | FastAPI [backend/main.py](backend/main.py) (`/api/*`, `/ws/predict`) | Legacy Flask [parkinsons-backend](parkinsons-backend/) `http://localhost:5000/upload-demo` |
| UI libs | recharts, framer-motion, lucide-react, axios | plain CSS ([App.css](parkinsons-frontend/src/App.css)) |
| State | Real-time streaming | Random demo result (`/upload-demo`) |
| Matches target stack (React+Vite+Tailwind) | ✅ Yes | ❌ No (CRA, no Tailwind) |

## Which is latest / working / to retain

- **Latest & aligned with the target stack: [frontend/](frontend/)** (Vite + Tailwind, the v2.0 "Healthcare AI" app referenced by the most recent commit `74cdaf5` about real-time monitoring).
- **[parkinsons-frontend/](parkinsons-frontend/) is the older prototype** ("Apmycure"-style wizard) wired to the legacy Flask demo endpoint that returns **random** numbers.
- ➡️ **Retain `frontend/`. Archive/remove `parkinsons-frontend/`** — unless the patient-form + file-upload wizard UX is wanted, in which case port `FormPage`/`VoiceInputPage`/`ResultPage` into `frontend/` and point them at the real backend.

## frontend/ (KEEP) — detailed checks

**Routes / pages** — declared in [Sidebar.jsx](frontend/src/components/Sidebar.jsx) (6 items) vs. wired in [App.jsx](frontend/src/App.jsx) (3 cases):

| Sidebar item | Component | Wired in App.jsx? |
|---|---|---|
| Overview | [Dashboard.jsx](frontend/src/pages/Dashboard.jsx) | ✅ |
| Real-Time Monitor | [RealTimeMonitor.jsx](frontend/src/pages/RealTimeMonitor.jsx) | ✅ |
| Comparison | [Comparison.jsx](frontend/src/pages/Comparison.jsx) | ✅ |
| Feature Analysis | [FeatureAnalysis.jsx](frontend/src/pages/FeatureAnalysis.jsx) | 🔴 **No** — falls to `default` → renders Dashboard |
| Model Performance | [ModelPerformance.jsx](frontend/src/pages/ModelPerformance.jsx) | 🔴 **No** — falls to `default` → renders Dashboard |
| About | (no component) | 🔴 **No** — renders Dashboard |

- **Broken routing:** clicking *Feature Analysis*, *Model Performance*, or *About* silently shows the Dashboard. `FeatureAnalysis.jsx` / `ModelPerformance.jsx` are also just "being migrated…" stubs ([FeatureAnalysis.jsx](frontend/src/pages/FeatureAnalysis.jsx)). The richer Streamlit versions of these views were never ported.

- **API calls:**
  - `Comparison.jsx` → `axios.get('/api/results')` ([Comparison.jsx:11](frontend/src/pages/Comparison.jsx#L11)) — proxied to FastAPI via [vite.config.js](frontend/vite.config.js).
  - `RealTimeMonitor.jsx` → opens raw `WebSocket` to `ws://<host>:8000/ws/predict` ([RealTimeMonitor.jsx:36](frontend/src/pages/RealTimeMonitor.jsx#L36)). **Note:** it hardcodes port `8000` and bypasses the Vite `/ws` proxy, so it only works when the API is literally on `:8000` of the same host.

- **Recording UI:** uses `getUserMedia` + `AudioContext` + **`createScriptProcessor(4096)`** ([RealTimeMonitor.jsx:31](frontend/src/pages/RealTimeMonitor.jsx#L31)). Streams **Float32 PCM at the browser's native rate (typically 48 kHz)** to the server, but the model/`LivePredictor` assumes **22050 Hz** → **sample-rate mismatch**: 5 s of buffer fills in ~2.3 s and features are computed on wrong-rate audio. Functional but acoustically incorrect.
  - `createScriptProcessor` is deprecated (should be `AudioWorklet`).

- **Result UI:** animated SVG gauge + recharts trend + log list — clean and complete for the live path.

- **Hard-coded data:** `Dashboard.jsx` stats (81 samples, 76%, 69.8%, "52 features") are literals ([Dashboard.jsx:5](frontend/src/pages/Dashboard.jsx#L5)). Note the inconsistency: Dashboard says "52 features", Comparison computes reduction off **52**, but the Streamlit app and report say **43**.

- **Build:** depends only on standard npm packages; no obvious syntax errors. Not built/verified in this audit (no `node_modules` present, would need `npm install`).

## parkinsons-frontend/ (ARCHIVE) — detailed checks

- 3-step flow: [FormPage.js](parkinsons-frontend/src/pages/FormPage.js) (name/age/gender) → [VoiceInputPage.js](parkinsons-frontend/src/pages/VoiceInputPage.js) (MediaRecorder record or file upload) → [ResultPage.js](parkinsons-frontend/src/pages/ResultPage.js) (risk score + hard-coded Pune hospital list).
- Posts to `http://localhost:5000/upload-demo` ([VoiceInputPage.js:114](parkinsons-frontend/src/pages/VoiceInputPage.js#L114)) — the **demo** endpoint that returns a **random** 30–80 value, so results are not real predictions.
- No authentication despite the report's "Login/Register Module" claim.
- React 19 + CRA: `react-scripts` is effectively unmaintained; not aligned with the Vite/Tailwind target.

## Authentication
**Neither frontend implements login/register.** The report's "USER INTERFACE LAYER … Login/Register Module" ([Project_Report…txt:827](Project_Report_Parkinsons_Detection.txt#L827)) does not exist.

## Recommendations (no code changes yet)
1. Keep `frontend/`; delete or `archive/` `parkinsons-frontend/`.
2. Wire `FeatureAnalysis`, `ModelPerformance`, and an `About` page into `App.jsx` (or remove them from the sidebar).
3. Fix the WebSocket sample-rate mismatch (resample to 22050 client- or server-side) and route through the Vite proxy.
4. Replace hard-coded dashboard metrics with a `/api/results`-driven source of truth; reconcile 43 vs 52 feature count.
5. Migrate `createScriptProcessor` → `AudioWorklet`.
