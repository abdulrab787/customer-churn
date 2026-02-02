import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.preprocessing import load_data, prepare_data

def train_baseline_model():
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    # Baseline model
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",   # IMPORTANT for churn imbalance
        random_state=42
    )

    # Full pipeline: preprocessing + model
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Fit model
    full_pipeline.fit(df.drop(columns=["Churn"]), 
                      df["Churn"].map({"Yes": 1, "No": 0}))

    # Predictions
    y_pred = full_pipeline.predict(
        df.drop(columns=["Churn"])
    )

    y_proba = full_pipeline.predict_proba(
        df.drop(columns=["Churn"])
    )[:, 1]

    # Metrics
    acc = accuracy_score(
        df["Churn"].map({"Yes": 1, "No": 0}), y_pred
    )

    auc = roc_auc_score(
        df["Churn"].map({"Yes": 1, "No": 0}), y_proba
    )

    print("=== BASELINE LOGISTIC REGRESSION ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(
        df["Churn"].map({"Yes": 1, "No": 0}), y_pred
    ))

    # Save artifacts (very important for portfolio)
    joblib.dump(full_pipeline, "models/baseline_logreg_pipeline.pkl")

    print("\nModel saved to: models/baseline_logreg_pipeline.pkl")

    return full_pipeline
