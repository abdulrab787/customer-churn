import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import mlflow
from src.mlflow_utils import start_experiment, log_model_and_metrics
from src.preprocessing import load_data, prepare_data
from src.business_metrics import find_optimal_threshold
from src.model_baseline import log_experiment
from src.feature_engineering import FeatureEngineer


def train_ensemble_model():
    start_experiment("Customer_Churn_Experiments")

    with mlflow.start_run():

        # Load data
        df = load_data()
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

        # Convert full dataset for final training
        y = df["Churn"].map({"Yes": 1, "No": 0})
        X = df.drop(columns=["Churn"])

        # =========================
        # 1) LIGHTGBM MODEL
        # =========================
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

        lgbm_pipeline = Pipeline(steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", lgbm)
        ])

        lgbm_pipeline.fit(X, y)
        lgbm_proba = lgbm_pipeline.predict_proba(X)[:, 1]

        # =========================
        # 2) XGBOOST MODEL
        # =========================
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y) - y.sum()) / y.sum(),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )

        xgb_pipeline = Pipeline(steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", xgb)
        ])

        xgb_pipeline.fit(X, y)
        xgb_proba = xgb_pipeline.predict_proba(X)[:, 1]

        # =========================
        # 3) WEIGHTED ENSEMBLE
        # =========================
        best_acc = 0
        best_weight = None

        for w in np.linspace(0.2, 0.8, 7):
            ensemble_proba = (w * xgb_proba) + ((1 - w) * lgbm_proba)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)

            acc = accuracy_score(y, ensemble_pred)

            if acc > best_acc:
                best_acc = acc
                best_weight = w
                best_proba = ensemble_proba

        print(f"\n🔥 Best Ensemble Weight (XGB): {best_weight:.2f}")
        print(f"🔥 Best Ensemble Accuracy: {best_acc:.4f}")

        # Final predictions
        ensemble_pred = (best_proba >= 0.5).astype(int)
        auc = roc_auc_score(y, best_proba)

        print("\n=== ENSEMBLE PERFORMANCE ===")
        print(f"Accuracy: {best_acc:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(classification_report(y, ensemble_pred))

        # =========================
        # 4) BUSINESS THRESHOLD
        # =========================
        best_row, _ = find_optimal_threshold(
            y_true=y,
            y_proba=best_proba,
            retention_cost=100,
            acquisition_cost=500
        )

        # =========================
        # 5) SAVE ENSEMBLE ARTIFACT
        # =========================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_artifact = {
            "lgbm_pipeline": lgbm_pipeline,
            "xgb_pipeline": xgb_pipeline,
            "best_weight_xgb": best_weight
        }

        model_path = f"models/v3_ensemble_{timestamp}.pkl"
        joblib.dump(ensemble_artifact, model_path)
        print(f"Ensemble saved to: {model_path}")

        # =========================
        # 6) LOG TO EXPERIMENT TRACKER
        # =========================
        log_experiment(
            exp_id=f"exp_{timestamp}",
            model_name="XGB+LightGBM_Ensemble",
            features="all_features_v2",
            acc=best_acc,
            auc=auc,
            notes=f"Weighted ensemble, xgb_weight={best_weight:.2f}"
        )

        # =========================
        # 7) LOG TO MLFLOW
        # =========================
        params = {
            "model": "XGB_LightGBM_Ensemble",
            "xgb_weight": float(best_weight)
        }

        metrics = {
            "accuracy": float(best_acc),
            "roc_auc": float(auc),
            "best_threshold": float(best_row["threshold"]),
            "business_cost": float(best_row["business_cost"])
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(lgbm_pipeline, artifact_path="lgbm_model")
        mlflow.sklearn.log_model(xgb_pipeline, artifact_path="xgb_model")

        return ensemble_artifact
