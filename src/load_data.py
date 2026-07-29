import pandas as pd
from db import get_engine

CSV_PATH = "data/telco_churn.csv"
TABLE_NAME = "customers_raw"


def load_csv_to_postgres(csv_path: str = CSV_PATH, table_name: str = TABLE_NAME):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path}")

    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Wrote data to table '{table_name}'.")


if __name__ == "__main__":
    load_csv_to_postgres()