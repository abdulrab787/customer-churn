import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import os


def train_and_evaluate(X_train, X_val, y_train, y_val, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)

        print(f"\n{name.upper()} ROC-AUC: {score:.4f}")
        print(classification_report(y_val, preds > 0.5))

        if score > best_score:
            best_score = score
            best_model = model

    joblib.dump(best_model, os.path.join(output_dir, "best_model.pkl"))
    print(f"\n✅ Best model saved with ROC-AUC: {best_score:.4f}")

    return best_model
