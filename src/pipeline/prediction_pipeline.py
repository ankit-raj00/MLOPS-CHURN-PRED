import os
import joblib
import pandas as pd
import mlflow
from src.exception import ChurnException
import sys
from src.utils.transformers import FeaturePreprocessor
from src.utils.common import read_yaml
from src.constants import PARAMS_FILE_PATH
from dotenv import load_dotenv

load_dotenv()

class PredictionPipeline:
    def __init__(self):
        self.model = None
        # Load params to get model name
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.model_name = self.params.mlflow_config.model_name
        
        # Optional: Set URI if provided in env, else rely on default
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def load_resources(self):
        """Loads the Unified Pipeline Model"""
        try:
            if self.model is None:
                try:
                    target_stage = self.params.model_deployment.target_stage
                    # Load Production Pipeline ONLY (Strict Mode)
                    print(f"Loading Pipeline ({self.model_name}) from alias '@{target_stage}'...")
                    self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@{target_stage}")
                    
                except Exception as e:
                    print(f"❌ Model not found in Registry: {e}")
                    raise Exception(f"No model found with alias '@{target_stage}'. Service Unavailable.")
                    
        except Exception as e:
            raise ChurnException(e, sys)

    def predict(self, data: dict):
        try:
            self.load_resources()
            
            # Create DataFrame from input
            # Note: We pass Raw customer data. The Pipeline handles encoding.
            input_df = pd.DataFrame([data])
            
            # Predict
            prediction = self.model.predict(input_df)
            
            # Get Probability (Safe check if model supports it)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(input_df)[0][1] # Probability of class 1 (Churn)
            else:
                proba = 0.0 # Fallback
                
            return prediction[0], proba
            
        except Exception as e:
            raise ChurnException(e, sys)
