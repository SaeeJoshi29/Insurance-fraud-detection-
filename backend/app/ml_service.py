import json
from pathlib import Path
import joblib
import pandas as pd
from .config import MODELS_DIR

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "KNN": "knn.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
}

class MLService:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.schema = {}
        self.load()

    def load(self):
        metrics_path = MODELS_DIR / "metrics.json"
        schema_path = MODELS_DIR / "schema.json"
        if not metrics_path.exists() or not schema_path.exists():
            raise RuntimeError("Models are not trained yet. Run: python train.py")
        self.metrics = json.loads(metrics_path.read_text())
        self.schema = json.loads(schema_path.read_text())
        for name, filename in MODEL_FILES.items():
            path = MODELS_DIR / filename
            if not path.exists():
                raise RuntimeError(f"Missing trained model: {path}")
            self.models[name] = joblib.load(path)

    def predict(self, model_name: str, claim: dict):
        if model_name not in self.models:
            raise ValueError(f"Unknown model. Choose one of: {', '.join(self.models)}")
        frame = pd.DataFrame([claim])
        # Keep exactly the feature columns used during training.
        frame = frame.reindex(columns=self.schema["model_features"])
        model = self.models[model_name]
        pred = int(model.predict(frame)[0])
        if hasattr(model, "predict_proba"):
            fraud_prob = float(model.predict_proba(frame)[0][1])
        else:
            # All four configured models expose predict_proba; this is defensive only.
            score = float(model.decision_function(frame)[0])
            fraud_prob = 1.0 / (1.0 + __import__("math").exp(-score))
        fraud_prob = max(0.0, min(1.0, fraud_prob))
        genuine_prob = 1.0 - fraud_prob
        risk = risk_level(fraud_prob)
        return pred, fraud_prob, genuine_prob, risk

def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.70:
        return "Medium"
    return "High"
