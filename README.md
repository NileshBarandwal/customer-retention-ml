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
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.train
```

MLflow will log runs to `mlruns/`. View the UI with:

```bash
mlflow ui
```

### 3. Start the prediction API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.
