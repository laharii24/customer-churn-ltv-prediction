"""
test_api.py

Week 4: Simple automated tests for the churn/LTV API.
Run the server first (uvicorn src.api:app --reload), then run this script
in a separate terminal:
    python src/test_api.py
"""

import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

# Check the server is running before doing anything else
try:
    health = requests.get(f"{BASE_URL}/health", timeout=3)
    health.raise_for_status()
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the API server.")
    print("   Please start it first in another terminal with:")
    print("   uvicorn src.api:app --reload")
    sys.exit(1)
except requests.exceptions.RequestException as e:
    print(f"❌ Server responded with an error: {e}")
    sys.exit(1)

print("✅ Server is running. Proceeding with tests...\n")

at_risk_customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 191.0,
    "avg_monthly_usage_ratio": 0.5, "charge_per_tenure_month": 47.75,
}

response = requests.post(f"{BASE_URL}/predict/churn", json=at_risk_customer, timeout=5)
response.raise_for_status()
print("Churn prediction response:", response.json())
print("API docs available at:", f"{BASE_URL}/docs") 