import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

from src.preprocessing import load_data, prepare_data
from src.business_metrics import find_optimal_threshold
from src.mlflow_utils import start_experiment, log_model_and_metrics
from src.model_baseline import log_experiment


def train_lightgbm_model():
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    # Define model
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42
    )

    # Build pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", lgbm)
    ])

    # Start MLflow experiment
    start_experiment("Customer_Churn_Experiments")

    with mlflow.start_run():
        # Train
        full_pipeline.fit(X_train, y_train)

        # Predict
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Business threshold
        best_row, _ = find_optimal_threshold(
            y_true=y_test,
            y_proba=y_proba,
            retention_cost=100,
            acquisition_cost=500
        )

        # MLflow logging
        params = {
            "model": "LightGBM",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced"
        }

        metrics = {
            "accuracy": acc,
            "roc_auc": auc,
            "best_threshold": float(best_row["threshold"]),
            "business_cost": float(best_row["business_cost"])
        }

        log_model_and_metrics(full_pipeline, params, metrics)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/v2_lightgbm_{timestamp}.pkl"
        joblib.dump(full_pipeline, model_path)
        print(f"Model saved to: {model_path}")

        # Local experiment log
        log_experiment(
            exp_id=f"exp_{timestamp}",
            model_name="LightGBM",
            features="all_features_v1",
            acc=acc,
            auc=auc,
            notes=f"Business-optimized threshold={best_row['threshold']:.2f}"
        )

    print("LightGBM training completed successfully.")
    return full_pipeline