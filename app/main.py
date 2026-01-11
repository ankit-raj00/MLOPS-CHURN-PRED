import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from src.pipeline.prediction_pipeline import PredictionPipeline

# Monitoring
from prometheus_fastapi_instrumentator import Instrumentator
from app.monitoring import churn_prediction_total, prediction_latency_seconds, churn_probability_histogram

# --- Global Pipeline ---
pipeline = None

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    
    try:
        pipeline = PredictionPipeline()
        pipeline.load_resources()
        print("✅ Prediction Pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
    yield

# --- API App ---
app = FastAPI(title="Churn Prediction API", lifespan=lifespan)

# --- Instrumentation ---
Instrumentator().instrument(app).expose(app)

# --- Pydantic Schema ---
class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    payment_method: str
    internet_service: str
    tech_support: str
    online_security: str
    support_calls: int

# --- Endpoints ---
@app.get("/")
def home():
    return {"message": "Churn Prediction API (Unified Pipeline) is Live."}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    global pipeline
    if not pipeline:
         raise HTTPException(status_code=503, detail="Pipeline not loaded.")
    
    try:
        # Measure Latency
        with prediction_latency_seconds.time():
            # Just pass dictionary. Pipeline handles everything.
            # Returns (prediction, probability)
            churn_val, churn_prob = pipeline.predict(customer.model_dump())
            
        result = "Churn" if churn_val == 1 else "No Churn"
        
        # Log Metrics
        churn_prediction_total.labels(
            prediction_class=result, 
            model_version="v1"  # Ideally dynamically fetched
        ).inc()
        
        churn_probability_histogram.observe(churn_prob)
        
        return {
            "prediction": result,
            "probability": float(churn_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
