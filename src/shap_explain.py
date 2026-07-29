import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

MODEL_DIR = "models"
REPORT_DIR = "reports"


def explain_model(model_path: str = f"{MODEL_DIR}/xgboost.joblib"):
    import os
    os.makedirs(REPORT_DIR, exist_ok=True)

    model = joblib.load(model_path)
    X_test = pd.read_csv(f"{MODEL_DIR}/X_test.csv")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_summary.png")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_feature_importance.png")
    plt.close()

    print(f"SHAP plots saved to '{REPORT_DIR}/'")


if __name__ == "__main__":
    explain_model()