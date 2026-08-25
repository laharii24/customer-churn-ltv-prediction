# Customer Churn Prediction & LTV Engine — API

## Overview
This project predicts customer churn risk and customer lifetime value (LTV)
for a telecom company, using the Telco Customer Churn dataset. It covers data
ingestion, EDA, feature engineering, model training, a REST API serving
real-time predictions, and a two-page Power BI dashboard.

## Documentation
Full project write-up (Weeks 1–4): [`Customer_Churn_LTV_Report_Week1-4.docx`](./Customer_Churn_LTV_Report_Week1-4.docx)

## Tech Stack
- Python, Pandas, scikit-learn
- SQLite (via SQLAlchemy)
- FastAPI, Uvicorn
- Docker, Docker Compose
- Power BI (churn + LTV dashboard)

## Project Structure
- `src/load_data.py` — loads the raw CSV into the database
- `src/preprocessing.py` — cleans data and engineers features
- `src/eda.py` — exploratory data analysis
- `src/train_model.py` — trains churn classification models
- `src/train_ltv.py` — trains the LTV regression model
- `src/shap_explain.py` — SHAP-based model explainability
- `src/api.py` — FastAPI service exposing predictions
- `src/test_api.py` — automated tests for all API endpoints
- `churn_dashboard.pbix` — Power BI dashboard (Churn Analysis + Customer Value pages)
- `Dockerfile` / `docker-compose.yml` — containerized deployment

## Running locally
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or just
`http://127.0.0.1:8000/` — it redirects there automatically.

## Running with Docker
```bash
docker compose up --build
```
API available at `http://localhost:8000`.

## Running the tests
```bash
python src/test_api.py
```
Checks `/health`, `/predict/churn`, and `/predict/ltv` all return valid responses.

## API Endpoints

### `GET /health`
Returns `{"status": "ok"}` if the service is running.

### `POST /predict/churn`
**Request body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.50,
  "avg_monthly_usage_ratio": 1.2,
  "charge_per_tenure_month": 6.5
}
```
**Response:**
```json
{
  "churn_probability": 0.7421,
  "churn_prediction": "Yes"
}
```

### `POST /predict/ltv`
Same request body as above.

**Response:**
```json
{
  "predicted_ltv": 1683.40
}
```

## Power BI Dashboard
Two pages, built on the same underlying data:
- **Churn Analysis** — churn rate, total customers, average tenure, and churn
  broken down by contract type, tenure bucket, internet service, and payment method.
- **Customer Value (LTV)** — LTV (simple, active-customers-only, and
  churn-adjusted), revenue at risk, LTV by contract type, and cumulative LTV
  by tenure bucket.


## Sample Usage & Output

### 1. Start the API server
```bash
uvicorn src.api:app --reload
```

### 2. Run the test script (in a separate terminal)
```bash
python src/test_api.py
```

### 3. Example output
```
✅ Server is running. Proceeding with tests...

Churn prediction response: {'churn_probability': 0.8352, 'churn_prediction': 'Yes'}
```

This means the API correctly identified a high-risk customer profile as 83.5% likely to churn.

## Author
Karakana Sai Lahari