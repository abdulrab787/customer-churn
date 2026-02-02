import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import RAW_DATA_PATH, TARGET

# Feature engineering functions
def add_tenure_features(df):
    df["Tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "6-12m", "12-24m", "24-48m", "48m+"]
    )
    return df

def add_spending_features(df):
    spend_cols = ["MonthlyCharges", "TotalCharges"]
    
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighSpender"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    
    return df

def add_contract_risk_score(df):
    contract_map = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1
    }
    df["ContractRisk"] = df["Contract"].map(contract_map)
    return df

def add_family_features(df):
    df["FamilySize"] = df["Dependents"].map({"Yes": 1, "No": 0}) + \
                       df["Partner"].map({"Yes": 1, "No": 0})
    return df

def add_log_transform(df):
    skewed_cols = ["MonthlyCharges", "TotalCharges"]
    for col in skewed_cols:
        df[col] = np.log1p(df[col])
    return df



def load_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Convert TotalCharges safely
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID if exists
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df

def get_feature_types(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if TARGET in categorical_cols:
        categorical_cols.remove(TARGET)

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return categorical_cols, numerical_cols

def build_preprocessing_pipeline(categorical_cols, numerical_cols):
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ]
    )

    return preprocessor

def prepare_data(df):
    df = df.copy()
    df = clean_data(df)
    
    # FEATURE ENGINEERING STEPS
    df = add_tenure_features(df)
    df = add_spending_features(df)
    df = add_contract_risk_score(df)
    df = add_family_features(df)
    df = add_log_transform(df)

    # CLEANING STEPS
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # Identify columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # Preprocessing pipelines
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
