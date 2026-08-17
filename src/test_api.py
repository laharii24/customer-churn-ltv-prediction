"""
test_api.py

Week 4: Simple automated tests for the churn/LTV API.
Run the server first (uvicorn src.api:app --reload), then run this script
in a separate terminal:
    python src/test_api.py
"""

import requests

BASE_URL = "http://127.0.0.1:8080"

at_risk_customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 191.0,
    "avg_monthly_usage_ratio": 0.5, "charge_per_tenure_month": 47.75,
}


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print("PASS: /health returned 200")


def test_predict_churn():
    r = requests.post(f"{BASE_URL}/predict/churn", json=at_risk_customer)
    assert r.status_code == 200
    prob = r.json()["churn_probability"]
    print(f"PASS: /predict/churn returned churn_probability={prob}")


def test_predict_ltv():
    r = requests.post(f"{BASE_URL}/predict/ltv", json=at_risk_customer)
    assert r.status_code == 200
    ltv = r.json()["predicted_ltv"]
    print(f"PASS: /predict/ltv returned predicted_ltv={ltv}")


if __name__ == "__main__":
    test_health()
    test_predict_churn()
    test_predict_ltv()
    print("\nAll tests passed!")