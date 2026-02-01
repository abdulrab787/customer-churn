import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")

TARGET = "Churn"
