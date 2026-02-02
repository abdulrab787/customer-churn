import os
import pandas as pd
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

from src.preprocessing import load_data, prepare_data
from src.model_baseline import log_experiment
from src.business_metrics import find_optimal_threshold


def train_lightgbm_model():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    # LightGBM baseline
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
    full_pipeline = lgbm
    

    # Train on TRAIN ONLY
    lgbm.fit(X_train, y_train)


    # Predictions on TEST ONLY
    y_pred = lgbm.predict(X_test)
    y_proba = lgbm.predict_proba(X_test)[:, 1]
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/v2_lightgbm_{timestamp}.pkl"

    joblib.dump(lgbm, model_path)

    # Business threshold optimization
    best_row, cost_df = find_optimal_threshold(
        y_true=y_test,
        y_proba=y_proba,
        retention_cost=100,
        acquisition_cost=500
    )

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== LIGHTGBM BASELINE ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/v2_lightgbm_{timestamp}.pkl"
    joblib.dump(full_pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # Log experiment
    log_experiment(
        exp_id=f"exp_{timestamp}",
        model_name="LightGBM",
        features="all_features_v1",
        acc=acc,
        auc=auc,
        notes=f"Business-optimized threshold={best_row['threshold']:.2f}"
    )

    return full_pipeline