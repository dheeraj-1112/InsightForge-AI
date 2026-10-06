# InsightForge AI — v1.0

**InsightForge AI** is a decision-intelligence platform for a final-year Data Science + Generative AI project. It turns uploaded business data and documents into explainable analytics, anomaly signals, behavioral segments, forecasts, and natural-language answers.

## What is implemented

- Professional responsive SaaS-style dashboard
- CSV / XLSX / XLS upload
- Automatic data profiling and quality score
- Dynamic KPIs and charts from the current dataset
- Automatic business insights
- Isolation Forest anomaly detection with feature-level evidence
- K-Means behavioral segmentation
- Transparent baseline time-series forecasting with uncertainty band
- Natural-language data analyst endpoint
- Optional real Generative AI Copilot through `OPENAI_API_KEY`
- Grounded local analyst fallback when no API key is configured
- PDF/TXT/Markdown knowledge upload
- TF-IDF retrieval with document/source snippets
- FastAPI backend + API docs
- Render and Docker deployment configuration
- Basic automated tests

## Local run (Windows)

Use Python 3.13.

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

If port 8000 is occupied, use:

```bat
uvicorn backend.main:app --reload --port 8001
```

## Real Generative AI

Copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
```

For deployment, set these as platform environment variables rather than committing `.env`.

Without an API key, the Copilot still works using InsightForge's deterministic, evidence-grounded analyst. This is useful for demos and avoids exposing a secret in the browser.

## Deployment — Render

1. Create a GitHub repository and upload this project.
2. In Render, create a new **Web Service** from the repository.
3. Render can use the included `render.yaml`, or set:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add `OPENAI_API_KEY` and `OPENAI_MODEL` under environment variables if you want real LLM generation.
5. Deploy and open the generated HTTPS URL.

The application is a single FastAPI service, so you do **not** need a separate frontend deployment.

## Viva/demo flow

1. Open the public URL.
2. Upload a business CSV.
3. Show live KPI/chart changes.
4. Open Analytics and explain profiling/data quality.
5. Open Anomalies and show evidence for an unusual row.
6. Open Segments and explain K-Means clustering.
7. Open Forecast and explain the transparent baseline.
8. Ask Copilot: “What are the biggest anomalies and why?”
9. Upload a project/report PDF under Knowledge and retrieve a relevant passage.
10. Explain that the system separates computation from generation: ML/statistics produce evidence; GenAI explains it.

## Architecture

```text
Browser UI
   |
   v
FastAPI API
   |
   +-- Data profiler
   +-- EDA/chart engine
   +-- Isolation Forest
   +-- K-Means segmentation
   +-- Forecast baseline
   +-- Knowledge retrieval
   +-- Evidence builder
   +-- Generative AI adapter
   |
   v
Grounded response + reliability + sources
```

## Important limitation

This is a strong academic/demo implementation, not a regulated production decision system. Forecasting is intentionally a transparent linear-trend baseline. Real business deployment would add authentication, persistent storage, monitoring, model validation, access control, and stronger time-series/model selection.
