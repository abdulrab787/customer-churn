from src.model_baseline import train_baseline_model

def test_baseline_training():
    model = train_baseline_model()
    assert model is not None
    print("Baseline training completed successfully.")
