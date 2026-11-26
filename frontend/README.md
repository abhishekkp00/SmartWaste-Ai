# SmartWaste AI Dashboard

React + Vite dashboard for the SmartWaste FastAPI backend.

## Run

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Set `VITE_API_BASE_URL` to the FastAPI URL. The interface displays a backend connection indicator and uses `/api/health`, `/api/analytics`, and `/api/predict`.

A trained checkpoint is required for valid model predictions. Without one, prediction requests may fail; this UI does not invent model results.
