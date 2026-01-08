import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from src.pipeline.prediction_pipeline import PredictionPipeline

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
        # Just pass dictionary. Pipeline handles everything.
        churn_prob = pipeline.predict(customer.model_dump())
        result = "Churn" if churn_prob == 1 else "No Churn"
        
        return {
            "prediction": result,
            "raw_value": float(churn_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train")
def trigger_training():
    try:
        exit_code = os.system("dvc repro --force")
        if exit_code == 0:
            return {"status": "Success", "message": "Pipeline triggered."}
        else:
            raise HTTPException(status_code=500, detail="DVC Pipeline failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
