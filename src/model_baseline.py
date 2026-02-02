import os
import pandas as pd
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.preprocessing import load_data, prepare_data

def log_experiment(exp_id, model_name, features, acc, auc, notes):
    log_path = "experiments/experiment_log.csv"

    new_row = pd.DataFrame([{
        "experiment_id": exp_id,
        "model": model_name,
        "features": features,
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "notes": notes
    }])

    if os.path.exists(log_path):
        new_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        new_row.to_csv(log_path, index=False)

def train_baseline_model():
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    full_pipeline.fit(X, y)

    y_pred = full_pipeline.predict(X)
    y_proba = full_pipeline.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    print(f"Baseline Accuracy: {acc:.4f}")
    print(f"Baseline ROC-AUC: {auc:.4f}")

    # ----- MODEL VERSIONING -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/v1_baseline_logreg_{timestamp}.pkl"

    joblib.dump(full_pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # ----- LOG EXPERIMENT -----
    log_experiment(
        exp_id=f"exp_{timestamp}",
        model_name="LogisticRegression",
        features="all_features_v1",
        acc=acc,
        auc=auc,
        notes="Baseline with balanced classes"
    )

    return full_pipeline
if __name__ == "__main__":
    train_baseline_model()
    
