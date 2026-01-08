import pandas as pd
import os
from pathlib import Path
import mlflow
import mlflow.lightgbm
from urllib.parse import urlparse
from src.logger import logger
from src.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import joblib
from src.utils.common import save_json
from src.exception import ChurnException
import sys
import numpy as np

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)
            pipeline = joblib.load(self.config.model_path)
            
            # Read Run ID
            run_id_path = os.path.join(os.path.dirname(self.config.model_path), "run_id.txt")
            with open(run_id_path, "r") as f:
                run_id = f.read().strip()

            test_x = test_data.iloc[:, :-1]
            test_y = test_data.iloc[:, -1]
            
            # Note: 'test_data' is ALREADY transformed (from Stage 02). 
            # 'pipeline' expects RAW data.
            # So we extract the trained model step to evaluate on transformed data.
            model_step = pipeline.named_steps['model']
            predicted_qualities = model_step.predict(test_x)

            acc = accuracy_score(test_y, predicted_qualities)
            f1 = f1_score(test_y, predicted_qualities)
            recall = recall_score(test_y, predicted_qualities)
            precision = precision_score(test_y, predicted_qualities)
            
            scores = {
                "accuracy": acc, 
                "f1_score": f1,
                "recall": recall,
                "precision": precision
            }
            
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            # MLflow Logging
            # MLflow Logging
            # Remove hardcoded URI, rely on env var or system default
            # mlflow.set_tracking_uri("sqlite:///mlflow.db") # Handled by env var
            
            mlflow.set_experiment(self.config.mlflow_config['experiment_name'])
            
            # Resume the Training Run
            with mlflow.start_run(run_id=run_id):
                # 0. Log Model (ALWAYS) - Single Source of Truth
                model_info = mlflow.sklearn.log_model(pipeline, name="model")
                model_uri = model_info.model_uri

                mlflow.log_metrics(scores)
                
                # --- CHAMPION / CHALLENGER LOGIC ---
                model_name = self.config.mlflow_config['model_name']
                target_metric = self.config.mlflow_config['target_metric']
                
                # Default to current score as baseline
                production_score = 0.0
                
                client = mlflow.tracking.MlflowClient()
                
                try:
                    # 1. Search for Production Model
                    # 1. Load Production Model (Directly via Alias)
                    # matches API behavior
                    logger.info(f"Loading Production model from: models:/{model_name}/Production")
                    prod_model_uri = f"models:/{model_name}/Production"
                    prod_model = mlflow.sklearn.load_model(prod_model_uri)
                    
                    # 2. Predict on Current Test Data
                    if hasattr(prod_model, 'named_steps'):
                        prod_estimator = prod_model.named_steps['model']
                    else:
                         prod_estimator = prod_model
                    
                    prod_preds = prod_estimator.predict(test_x)
                    
                    # 3. Calculate Score
                    if target_metric == "f1_score":
                        production_score = f1_score(test_y, prod_preds)
                    else:
                        production_score = accuracy_score(test_y, prod_preds)
                        
                    logger.info(f"Re-evaluated Production Score ({target_metric}): {production_score}")
                
                except Exception as e:
                    logger.warning(f"Could not re-evaluate production model: {e}")

                # Comparison
                current_score = scores[target_metric]
                # Champion/Challenger Comparison
                if current_score > production_score:
                    logger.info(f"New Model ({current_score}) > Production ({production_score}). Registering...")
                    
                    # Register Model (Point to the artifact we just logged above)
                    model_version = mlflow.register_model(model_uri, model_name)
                    
                    # Promote to Staging Stage (Manual approval needed for Production)
                    # (This moves the version to Staging)
                    client = mlflow.tracking.MlflowClient()
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Staging",
                        archive_existing_versions=True
                    )
                         
                    logger.info(f"Model Version {model_version.version} registered and promoted to Staging.")
                else:
                    logger.info(f"New Model ({current_score}) < Production ({production_score}). Discarding...")

        except Exception as e:
            raise ChurnException(e, sys)
