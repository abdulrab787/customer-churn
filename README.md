# 🚀 Customer Churn Prediction — End-to-End ML Project (MLOps + Business Impact)

## 📌 Project Overview

This project builds an **end-to-end machine learning system** to predict customer churn and optimize business decisions using **cost-sensitive modeling**. The pipeline follows industry best practices for reproducibility, experiment tracking, and model governance.

**Business Goal:**

> Identify customers likely to churn and prioritize retention actions in a cost-effective way.

---

## 🎯 Key Achievements

✅ Production-style ML pipeline (not just a notebook)
✅ Feature engineering based on business intuition
✅ Multiple models: Logistic Regression, LightGBM, XGBoost
✅ **Weighted Ensemble (XGB + LightGBM)** for best performance
✅ **Business-optimized decision threshold** (not default 0.5)
✅ Model versioning & artifact management
✅ **MLflow experiment tracking (MLOps-ready)**
✅ Reproducible project structure suitable for real companies

---

## 🗂️ Project Structure

```
customer-churn/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── v1_baseline_logreg_*.pkl
│   ├── v2_lightgbm_*.pkl
│   └── v3_ensemble_*.pkl
│
├── experiments/
│   └── experiment_log.csv
│
├── mlruns/                     # MLflow experiment logs
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Feature engineering + pipeline
│   ├── model_baseline.py      # Logistic Regression
│   ├── model_lightgbm.py      # LightGBM model
│   ├── model_ensemble.py      # XGB + LightGBM Ensemble
│   ├── business_metrics.py    # Cost-sensitive thresholding
│   └── mlflow_utils.py        # MLflow helpers
│
├── tests/
│   ├── test_baseline.py
│   ├── test_lightgbm.py
│   └── test_ensemble.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
│
├── .gitignore
└── README.md
```

---

## 🔍 Feature Engineering (Why This Matters)

We engineered domain-aware features such as:

* **Tenure bins:** capturing customer lifecycle risk
* **Spending behavior:** `AvgMonthlySpend`, `HighSpender`
* **ContractRisk score:** mapping contract types to risk levels
* **Family size interactions:** Partner + Dependents
* **Log transforms:** stabilizing skewed distributions

This alone gave a significant performance boost.

---

## 🧠 Modeling Approach

### 🔹 Baseline — Logistic Regression

* Balanced classes
* Full sklearn Pipeline (preprocessing + model)
* Accuracy: ~0.80
* ROC-AUC: ~0.85

### 🔹 LightGBM (Gradient Boosting)

* Handles nonlinear relationships
* Better performance than Logistic Regression
* Accuracy: ~0.84–0.86

### 🔹 Final Model — **XGBoost + LightGBM Ensemble**

* Weighted soft-voting ensemble
* Empirically tuned weight
* Best Accuracy: **~0.85–0.88**

---

## 💰 Business Impact (Cost-Sensitive Decisioning)

Instead of using the default **0.5 threshold**, we optimized decisions using a business cost model:

* **Retention cost:** $100 per customer
* **Acquisition cost:** $500 per lost customer

We selected a **business-optimal threshold (~0.42)** that minimizes total expected cost.

👉 This aligns the model with real business value, not just accuracy.

---

## 📊 Experiment Tracking (MLflow)

Every experiment logs:

* Model parameters
* Accuracy
* ROC-AUC
* Best business threshold
* Estimated business cost
* Trained model artifact

Run MLflow UI locally:

```bash
mlflow ui
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ▶️ How to Run the Project

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train baseline model

```bash
python tests/test_baseline.py
```

### 3) Train LightGBM model

```bash
python tests/test_lightgbm.py
```

### 4) Train Ensemble model

```bash
python tests/test_ensemble.py
```