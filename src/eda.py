import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_engine

OUTPUT_DIR = "reports"


def load_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM customers_raw", engine)


def run_eda(df: pd.DataFrame):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Shape:", df.shape)
    print("\nChurn distribution:\n", df["Churn"].value_counts(normalize=True))

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Churn by Contract Type")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/churn_by_contract.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack")
    plt.title("Tenure Distribution by Churn")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tenure_vs_churn.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges by Churn")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/charges_vs_churn.png")
    plt.close()

    print(f"\nPlots saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    data = load_data()
    run_eda(data)