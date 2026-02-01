from src.preprocessing import load_data, prepare_data

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Preprocessor built successfully!")