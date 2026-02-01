import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import RAW_DATA_PATH, TARGET

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
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"Yes": 1, "No": 0})

    categorical_cols, numerical_cols = get_feature_types(df)

    preprocessor = build_preprocessing_pipeline(categorical_cols, numerical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor
