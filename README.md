# DataNova AI

DataNova AI is a full-stack analytics workspace that turns user-owned CSV and Excel datasets into business-friendly answers. Version 3 replaces the sales-only demo path with dynamic schema profiling, calculated quality scores, preserved original and processed files, controlled natural-language operations, generated charts, anomaly signals, and data-derived forecasting.

## Run locally

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`; health checks are `GET /` and `GET /health`.

## Routes

Public routes: `/`, `/login`, `/signup`, `/forgot-password`.

Protected workspace routes: `/dashboard`, `/datasets`, `/datasets/:id`, `/ask-data`, `/insights`, `/predictions`, `/anomalies`, `/recommendations`, `/what-if`, `/reports`, `/profile`, `/settings`.

## Demo flow

1. Open the landing page and choose **Get started**.
2. Create an account or sign in. Passwords are hashed before storage in `backend/datanova.db`.
3. Upload a `.csv`, `.xlsx`, or `.xls` file from **Datasets**. Files are validated, assigned a user-owned dataset ID, saved in `backend/data`, profiled, and copied into a processed CSV without changing the original upload. `.xls` support is provided by `xlrd`.
4. Use **Ask Data** for a highest-value question, anomaly question, or scenario such as `What if sales increase by 10%?`.
5. Explore the separate Insights, Predictions, Anomalies, Recommendations, What-If, and Reports pages.
6. Use **Voice input** in a browser that supports the Web Speech API.

## API

- `GET /api/health`
- `GET /health`
- `GET /`
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `GET /api/auth/me?token=...`
- `POST /api/auth/logout`
- `GET /api/datasets`
- `POST /api/datasets/upload` with multipart field `file`
- `GET /api/datasets/{id}`
- `GET /api/datasets/{id}/dashboard`
- `GET /api/datasets/{id}/forecast`
- `GET /api/datasets/{id}/insights` (findings computed from the dataset; empty list when nothing measurable)
- `GET /api/datasets/{id}/download?processed=true`
- `DELETE /api/datasets/{id}`
- `POST /api/query` with JSON `{ "question": "...", "dataset_name": "..." }`

Successful uploads return `dataset_id`, `dataset_name`, `file_name`, `row_count`, `column_count`, `columns`, `quality_score`, `upload_status`, and nested `metadata` in addition to the backward-compatible `id` and `name` fields.

## Architecture

- `frontend/`: React + TypeScript + Vite dashboard
- `frontend/src/App.tsx`: public pages, protected route shell, workspace pages, and API integration
- `backend/app/main.py`: FastAPI HTTP boundary
- `backend/app/services/analytics.py`: schema detection, profiling, preprocessing, charts, controlled queries, and forecasting
- `backend/app/services/preprocessing.py`: legacy quality helper retained for compatibility
- `backend/app/agents/coordinator.py`: multi-agent query routing boundary
- `sample-data/`: explicitly labelled test/demo datasets with different schemas
- `backend/data/`: uploaded files at runtime
- `reports/`: reserved for generated report exports

The analytics engine is deterministic and local: it never executes generated Python and only permits controlled filter/group/aggregate/trend/correlation/anomaly/simulation operations. Google OAuth is intentionally not claimed as complete until real credentials and a tested callback are configured. Sessions are persisted in SQLite for this prototype; production should use signed, secure cookies or a hardened server-side session store with expiry.

## Validation

```powershell
cd frontend
npm run build
npm run lint

cd ..
python -m compileall backend\app
```

The repository includes employee, customer, missing/duplicate, numeric-only, categorical-only, and sales-shaped fixtures for schema testing.
