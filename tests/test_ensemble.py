from src.model_ensemble import train_ensemble_model
from src.preprocessing import load_data, prepare_data


def test_ensemble_training():
    """
    Ensures the ensemble:
    - trains end‑to‑end without errors
    - returns the correct artifact structure
    - both pipelines can predict on new data
    """

    # Train ensemble
    artifact = train_ensemble_model()

    # Validate returned structure
    assert isinstance(artifact, dict)
    assert "lgbm_pipeline" in artifact
    assert "xgb_pipeline" in artifact
    assert "best_weight_xgb" in artifact

    lgbm_pipe = artifact["lgbm_pipeline"]
    xgb_pipe = artifact["xgb_pipeline"]

    # Load fresh data for prediction test
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    # Ensure both models can predict
    lgbm_preds = lgbm_pipe.predict(X_test)
    xgb_preds = xgb_pipe.predict(X_test)

    # Validate prediction lengths
    assert len(lgbm_preds) == len(y_test)
    assert len(xgb_preds) == len(y_test)

    # Validate predictions are binary
    assert set(lgbm_preds).issubset({0, 1})
    assert set(xgb_preds).issubset({0, 1})