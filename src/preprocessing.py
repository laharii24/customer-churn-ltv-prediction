import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def prepare_dataset(df: pd.DataFrame):
    df = clean_data(df)
    df, encoders = encode_categoricals(df)
    return df, encoders


if __name__ == "__main__":
    from db import get_engine

    engine = get_engine()
    raw = pd.read_sql("SELECT * FROM customers_raw", engine)
    clean, _ = prepare_dataset(raw)
    clean.to_sql("customers_clean", engine, if_exists="replace", index=False)
    print(f"Cleaned data written to 'customers_clean' ({clean.shape[0]} rows)")