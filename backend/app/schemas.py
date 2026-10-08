from typing import Any, Dict
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    model: str = Field(..., description="One of Logistic Regression, KNN, Random Forest, SVM")
    claim: Dict[str, Any]

class PredictionResponse(BaseModel):
    model: str
    prediction: int
    prediction_label: str
    fraud_probability: float
    genuine_probability: float
    risk_level: str
    timestamp: str
