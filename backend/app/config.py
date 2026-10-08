from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "fraud_oracle.csv"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "fraud_predictions.db"
TARGET = "FraudFound_P"
ID_COLUMNS = ["PolicyNumber", "RepNumber"]
RANDOM_STATE = 42
