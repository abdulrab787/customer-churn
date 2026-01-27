import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


NUM_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID (not predictive)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def build_preprocessor():
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
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES)
        ]
    )

    return preprocessor


def preprocess_and_split(
    input_path,
    output_dir="data/processed",
    test_size=0.2,
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_path)
    df = clean_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))

    return X_train_processed, X_val_processed, y_train, y_val
