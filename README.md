📉 Customer Churn Prediction (End-to-End ML Project)
📌 Project Overview

Customer churn is one of the biggest revenue risks for subscription-based businesses.
This project builds a production-style machine learning pipeline to predict customer churn and explain why customers leave — enabling targeted retention strategies.

Business question:

Which customers are likely to churn, and what actions can reduce churn?

🎯 Objectives

Predict customer churn with high recall

Understand key churn drivers

Build reusable ML pipelines

Create explainable and business-ready outputs

📂 Project Structure
customer-churn/
│
├── data/
│   ├── raw/                # Original dataset (Kaggle IBM Telco)
│   ├── processed/          # Cleaned & encoded data
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_shap_explainability.ipynb
│
├── src/
│   ├── preprocessing.py    # Reusable feature pipeline
│   ├── train.py             # Model training & selection
│   ├── evaluate.py          # Metrics
│   └── predict.py           # Inference script
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── eda_summary.md
│   └── shap_insights.md
│
├── README.md
├── requirements.txt
└── .gitignore

🔍 Exploratory Data Analysis (EDA)

Key insights:

Month-to-month contracts have the highest churn

High MonthlyCharges strongly increase churn risk

Short-tenure customers are most vulnerable

Fiber optic users churn more than DSL customers

📄 Full analysis: notebooks/01_eda.ipynb

🧪 Feature Engineering

Cleaned TotalCharges

Dropped non-predictive IDs

One-hot encoded categorical features

Scaled numerical features

Built reusable preprocessing pipeline

📄 Code: src/preprocessing.py

🤖 Modeling

Models trained and compared:

Logistic Regression (baseline)

Random Forest

XGBoost (best model)

Evaluation metrics:

ROC-AUC (ranking churners)

Recall (don’t miss churners)

Precision (optimize retention budget)

📄 Training pipeline: src/train.py

🔎 Explainability (SHAP)

SHAP was used to explain model predictions.

Top churn drivers:

Month-to-month contracts

High monthly charges

Short tenure

Fiber optic internet

📄 Insights: reports/shap_insights.md

📈 Business Impact

How this model can be used:

Target high-risk customers with retention offers

Incentivize long-term contracts

Reduce churn-related revenue loss

Improve customer onboarding strategy

🚀 How to Run Locally
git clone https://github.com/abdulrab787/customer-churn.git
cd customer-churn
pip install -r requirements.txt

Run pipeline
python src/train.py

🛠 Tech Stack

Python, Pandas, NumPy

Scikit-learn, XGBoost

SHAP (Explainability)

VS Code, Git, GitHub

Jupyter Notebook

🧠 What This Project Demonstrates

End-to-end ML thinking

Business-oriented modeling

Clean project structure

Reproducible pipelines

Explainable AI

Professional Git workflow

👤 Author

Abdurrab Nizamuddeen
Aspiring Data Analyst | Machine Learning | Analytics
📫 GitHub: https://github.com/abdulrab787