# Customer Retention Prediction using ML

## Overview

This project builds an end-to-end machine learning system to predict customer churn. Given historical customer behaviour and account attributes, the model outputs a churn probability score that enables the business to proactively target at-risk customers with retention campaigns. Model decisions are made interpretable via SHAP explanations, and the system is exposed both as a REST API and an interactive Streamlit dashboard.

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, numpy |
| Modelling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Visualisation | Matplotlib, Seaborn |
| Serialisation | joblib |

## Project Structure

```
customer-retention-ml/
├── data/                   # Raw and processed datasets (git-ignored)
├── notebooks/              # Exploratory and analysis notebooks
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Ingestion, cleaning, feature engineering
│   ├── train.py            # Model training and MLflow logging
│   ├── evaluate.py         # Metrics, confusion matrix, SHAP analysis
│   └── predict.py          # Batch and single-record inference
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI prediction service
├── dashboard/
│   └── app.py              # Streamlit interactive dashboard
├── Dockerfile              # API container image (also reused by dashboard)
├── docker-compose.yml      # Orchestrates api + dashboard services
├── .dockerignore
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

You can run the project three ways. All three assume you have first trained the model so that `models/best_model.joblib` and `models/shap_explainer.joblib` exist:

```bash
python -m src.train          # produces both artifacts under models/
```

---

### Method 1 — Local (venv)

Best for active development and debugging.

```bash
# Setup
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (only the first time, or after changing the pipeline)
python -m src.train

# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard (after the API is up on :8000)
streamlit run dashboard/app.py
```

- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

Override the API URL the dashboard hits with `RETENTION_API_URL`, e.g.
`RETENTION_API_URL=http://localhost:9000 streamlit run dashboard/app.py`.

---

### Method 2 — Docker Compose (recommended)

Brings up both services with one command. The dashboard waits for the API's health check to pass before starting.

```bash
docker compose up --build
```

- API: `http://localhost:8000` (docs at `/docs`, status at `/health`)
- Dashboard: `http://localhost:8501`

Stop everything:

```bash
docker compose down
```

The `models/` directory on the host is mounted read-only into the API container, so re-running `python -m src.train` on the host automatically surfaces the new model on the next API restart.

---

### Method 3 — Individual Docker services

Run each service in its own container without Compose. Useful when deploying them to separate hosts.

```bash
# Build the image once
docker build -t customer-retention-ml:1.0 .

# API
docker run --rm -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    --name retention-api \
    customer-retention-ml:1.0

# Dashboard (in another terminal, after the API is reachable)
docker run --rm -p 8501:8501 \
    -e RETENTION_API_URL=http://host.docker.internal:8000 \
    --name retention-dashboard \
    customer-retention-ml:1.0 \
    streamlit run dashboard/app.py \
        --server.port=8501 --server.address=0.0.0.0 \
        --server.headless=true --browser.gatherUsageStats=false
```

On Linux replace `host.docker.internal` with `--network host` (and drop the `-p`) or with the host's bridge IP.

---

### Experiment tracking (optional)

After training, browse MLflow runs locally:

```bash
mlflow ui
```
