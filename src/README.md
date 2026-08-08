# Customer Churn Prediction & LTV Engine

## Overview
This project predicts customer churn risk and customer lifetime value (LTV)
for a telecom company, using the Telco Customer Churn dataset. It includes
data ingestion, EDA, feature engineering, model training, and a REST API
serving real-time predictions.

## Tech Stack
- Python, Pandas, scikit-learn
- SQLite (via SQLAlchemy)
- FastAPI, Uvicorn
- Docker, Docker Compose
- Power BI (churn dashboard)

## Project Structure
- `src/load_data.py` — loads the raw CSV into the database
- `src/preprocessing.py` — cleans data and engineers features
- `src/train_model.py` — trains churn classification models
- `src/train_ltv.py` — trains the LTV regression model
- `src/shap_explain.py` — SHAP-based model explainability
- `src/api.py` — FastAPI service exposing predictions

## Running locally
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API docs.

## Running with Docker
```bash
docker compose up --build
```

## API Endpoints
- `GET /health` — service health check
- `POST /predict/churn` — returns churn probability
- `POST /predict/ltv` — returns predicted lifetime value

## Author
Karakana Sai Lahari