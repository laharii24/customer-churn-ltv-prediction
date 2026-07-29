import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from db import get_engine

MODEL_DIR = "models"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_usage_ratio"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["charge_per_tenure_month"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    return df


def train_and_evaluate(df: pd.DataFrame):
    df = engineer_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            eval_metric="logloss", random_state=42
        ),
    }

    results = {}
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds))
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")

        results[name] = auc
        joblib.dump(model, f"{MODEL_DIR}/{name}.joblib")

    best_model_name = max(results, key=results.get)
    print(f"\nBest model by ROC-AUC: {best_model_name} ({results[best_model_name]:.4f})")

    X_test.to_csv(f"{MODEL_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{MODEL_DIR}/y_test.csv", index=False)

    return results, best_model_name


if __name__ == "__main__":
    engine = get_engine()
    data = pd.read_sql("SELECT * FROM customers_clean", engine)
    train_and_evaluate(data)