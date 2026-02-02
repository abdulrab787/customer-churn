from src.preprocessing import load_data, prepare_data

def test_preprocessing_shapes():
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert preprocessor is not None

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Preprocessor built successfully!")
test_preprocessing_shapes()