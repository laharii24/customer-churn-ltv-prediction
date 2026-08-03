"""
api.py

Week 3: FastAPI service exposing churn and LTV predictions.

Run locally:
    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Churn & LTV Prediction API",
    description="Predicts customer churn risk and lifetime value (LTV).",
    version="1.0.0",
)

# --- Load models once at startup, not on every request ---
churn_model = None
ltv_model = None
ltv_feature_columns = None


@app.on_event("startup")
def load_models():
    global churn_model, ltv_model, ltv_feature_columns
    churn_model = joblib.load("models/logistic_regression.joblib")
    ltv_model = joblib.load("models/ltv_model.joblib")
    ltv_feature_columns = joblib.load("models/ltv_feature_columns.joblib")
    print("Models loaded successfully.")


# --- Request / response schemas ---

class CustomerFeatures(BaseModel):
    """
    Raw customer fields, matching every column the churn/LTV models
    were trained on (see models/X_test.csv for the reference schema).
    """
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0, description="1 = senior citizen, 0 = not")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12, description="Months as a customer")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=845.50)
    avg_monthly_usage_ratio: float = Field(..., example=1.2, description="Engineered feature from preprocessing.py")
    charge_per_tenure_month: float = Field(..., example=6.5, description="Engineered feature from preprocessing.py")

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: str


class LTVResponse(BaseModel):
    predicted_ltv: float


def encode_input(features: CustomerFeatures) -> pd.DataFrame:
    """
    Convert incoming JSON into a single-row DataFrame matching the
    training feature format (same label-encoding style as preprocessing.py).
    """
    df = pd.DataFrame([features.dict()])
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes
    return df

from fastapi.responses import RedirectResponse


@app.get("/")
def root():
    """Redirect the base URL to the interactive API docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(features: CustomerFeatures):
    if churn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    df = encode_input(features)
    try:
        prob = churn_model.predict_proba(df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return ChurnResponse(
        churn_probability=round(float(prob), 4),
        churn_prediction="Yes" if prob >= 0.5 else "No",
    )


@app.post("/predict/ltv", response_model=LTVResponse)
def predict_ltv(features: CustomerFeatures):
    if ltv_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    df = encode_input(features)
    # Reindex to the exact column order the LTV model was trained on
    df = df.reindex(columns=ltv_feature_columns, fill_value=0)

    try:
        prediction = ltv_model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return LTVResponse(predicted_ltv=round(float(prediction), 2))