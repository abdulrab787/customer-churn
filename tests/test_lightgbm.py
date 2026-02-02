from src.model_lightgbm import train_lightgbm_model

def test_lightgbm_training():
    model = train_lightgbm_model()
    assert model is not None
    print("LightGBM training completed successfully.")
