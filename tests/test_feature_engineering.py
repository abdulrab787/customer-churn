import pandas as pd
from src.feature_engineering import FeatureEngineer


def test_feature_engineer_creates_expected_columns():
    """
    Ensures FeatureEngineer:
    - creates AvgMonthlySpend, HighSpender, ContractRisk
    - handles string/empty TotalCharges
    - handles tenure=0 safely
    - outputs correct numeric types
    """

    df = pd.DataFrame({
        "tenure": [0, 10, 24],
        "TotalCharges": ["", "200.0", "1500"],
        "Contract": ["Month-to-month", "One year", "Two year"]
    })

    fe = FeatureEngineer()
    out = fe.fit_transform(df)

    # --- Column existence ---
    assert "AvgMonthlySpend" in out.columns
    assert "HighSpender" in out.columns
    assert "ContractRisk" in out.columns

    # --- Type checks ---
    assert pd.api.types.is_numeric_dtype(out["AvgMonthlySpend"])
    assert pd.api.types.is_integer_dtype(out["HighSpender"])
    assert pd.api.types.is_integer_dtype(out["ContractRisk"])

    # --- Logic checks ---
    # Row 0: tenure=0 → protected to 1, TotalCharges="" → 0
    assert out.loc[0, "AvgMonthlySpend"] == 0
    assert out.loc[0, "HighSpender"] == 0
    assert out.loc[0, "ContractRisk"] == 2  # Month-to-month = highest churn risk

    # Row 1: 200 / 10 = 20
    assert out.loc[1, "AvgMonthlySpend"] == 20
    assert out.loc[1, "HighSpender"] == 0
    assert out.loc[1, "ContractRisk"] == 1  # One year

    # Row 2: 1500 / 24 = 62.5
    assert out.loc[2, "AvgMonthlySpend"] == 62.5
    assert out.loc[2, "HighSpender"] == 0
    assert out.loc[2, "ContractRisk"] == 0  # Two year = lowest churn risk