"""
train_ltv.py

Week 3: Customer Lifetime Value (LTV) regression model.
LTV = tenure (months) * MonthlyCharges
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from db import get_engine

MODEL_OUT = "models/ltv_model.joblib"


def load_clean_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM customers_raw", engine)
    return df


def build_ltv_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["LTV"] = df["tenure"] * df["MonthlyCharges"]
    return df


def select_features(df: pd.DataFrame):
    drop_cols = ["customerID", "Churn", "LTV"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes
    y = df["LTV"]
    return X, y


def main():
    print("Loading data...")
    df = load_clean_data()

    print("Building LTV target...")
    df = build_ltv_target(df)

    print("Selecting features...")
    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"LTV Model Evaluation:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R^2:  {r2:.4f}")

    joblib.dump(model, MODEL_OUT)
    print(f"Saved LTV model -> {MODEL_OUT}")

    joblib.dump(list(X.columns), "models/ltv_feature_columns.joblib")


if __name__ == "__main__":
    main()