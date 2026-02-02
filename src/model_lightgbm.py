import os
import pandas as pd
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

from src.preprocessing import load_data, prepare_data
from src.model_baseline import log_experiment   # reuse your logger

def train_lightgbm_model():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    # Convert target
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # LightGBM baseline (industry defaults)
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

    # Full pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", lgbm)
    ])

    # Train
    full_pipeline.fit(X, y)

    # Predictions
    y_pred = full_pipeline.predict(X)
    y_proba = full_pipeline.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    print("\n=== LIGHTGBM BASELINE ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))

    # ----- MODEL VERSIONING -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/v2_lightgbm_{timestamp}.pkl"

    joblib.dump(full_pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # ----- LOG EXPERIMENT -----
    log_experiment(
        exp_id=f"exp_{timestamp}",
        model_name="LightGBM",
        features="all_features_v1",
        acc=acc,
        auc=auc,
        notes="LightGBM baseline with balanced classes"
    )

    return full_pipeline
if __name__ == "__main__":
    train_lightgbm_model()
    