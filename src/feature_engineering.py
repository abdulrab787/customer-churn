from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        # Clean TotalCharges
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

        # Avoid division by zero
        tenure_safe = X["tenure"].replace(0, 1)

        # Feature 1: Avg monthly spend
        X["AvgMonthlySpend"] = X["TotalCharges"] / tenure_safe

        # Feature 2: High spender flag
        X["HighSpender"] = (X["AvgMonthlySpend"] > 70).astype(int)

        # Feature 3: Contract risk score
        contract_map = {
            "Month-to-month": 2,
            "One year": 1,
            "Two year": 0
        }
        X["ContractRisk"] = X["Contract"].map(contract_map).fillna(1)
        
        # Feature 4: Family size
        # Partner + Dependents → family size indicator
        X["FamilySize"] = (
            X["Partner"].map({"Yes": 1, "No": 0}).fillna(0)
            + X["Dependents"].map({"Yes": 1, "No": 0}).fillna(0)
        )

        return X