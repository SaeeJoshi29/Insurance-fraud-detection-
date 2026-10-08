from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    claim_json = Column(Text, nullable=False)
    selected_model = Column(String(50), nullable=False)
    prediction = Column(Integer, nullable=False)
    prediction_label = Column(String(50), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    genuine_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

def init_db():
    Base.metadata.create_all(bind=engine)
