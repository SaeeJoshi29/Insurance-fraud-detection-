import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .database import SessionLocal, init_db, Prediction
from .ml_service import MLService
from .schemas import PredictionRequest, PredictionResponse
from .config import MODELS_DIR

app = FastAPI(title="Insurance Claim Fraud Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
init_db()
ml = MLService()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(ml.models.keys())}

@app.get("/api/schema")
def schema():
    return ml.schema

@app.get("/api/models")
def models():
    return {"models": list(ml.models.keys())}

@app.get("/api/performance")
def performance():
    return ml.metrics

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)):
    rows = db.query(Prediction).all()
    total = len(rows)
    fraud = sum(r.prediction == 1 for r in rows)
    genuine = total - fraud
    return {
        "total_claims_analyzed": total,
        "genuine_claims": genuine,
        "fraud_claims": fraud,
        "fraud_percentage": round((fraud / total * 100) if total else 0, 2),
        "model_performance": ml.metrics.get("models", {}),
    }

@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    try:
        pred, fraud_prob, genuine_prob, risk = ml.predict(request.model, request.claim)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    now = datetime.now(timezone.utc)
    row = Prediction(
        claim_json=json.dumps(request.claim),
        selected_model=request.model,
        prediction=pred,
        prediction_label="Potential Fraud" if pred == 1 else "Genuine",
        fraud_probability=fraud_prob,
        genuine_probability=genuine_prob,
        risk_level=risk,
        created_at=now.replace(tzinfo=None),
    )
    db.add(row)
    db.commit()
    return PredictionResponse(
        model=request.model,
        prediction=pred,
        prediction_label=row.prediction_label,
        fraud_probability=round(fraud_prob * 100, 2),
        genuine_probability=round(genuine_prob * 100, 2),
        risk_level=risk,
        timestamp=now.isoformat(),
    )

@app.get("/api/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(min(limit, 200)).all()
    return [{
        "id": r.id, "model": r.selected_model, "prediction": r.prediction_label,
        "fraud_probability": round(r.fraud_probability * 100, 2),
        "genuine_probability": round(r.genuine_probability * 100, 2),
        "risk_level": r.risk_level, "timestamp": r.created_at.isoformat() + "Z"
    } for r in rows]
