import pandas as pd
import os
from src.logger import logger
from src.entity.config_entity import ModelTrainerConfig
import joblib
from lightgbm import LGBMClassifier
import mlflow
import mlflow.lightgbm
from src.exception import ChurnException
import sys

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
            
            # Assuming last column is target as per transformation
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]
            test_x = test_data.iloc[:, :-1]
            test_y = test_data.iloc[:, -1]

            # mlflow.set_tracking_uri("sqlite:///mlflow.db") # Handled by env var
            mlflow.set_experiment(self.config.mlflow_config['experiment_name'])

            with mlflow.start_run():
                model = LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    class_weight=self.config.class_weight,
                    random_state=self.config.random_state,
                    verbosity=self.config.verbosity
                )
                
                model.fit(train_x, train_y)

                # --- NEW: Pipeline Construction ---
                from sklearn.pipeline import Pipeline
                
                # Load Preprocessor
                preprocessor_path = "artifacts/data_transformation/preprocessor.pkl"
                if os.path.exists(preprocessor_path):
                    preprocessor = joblib.load(preprocessor_path)
                else:
                    raise Exception(f"Preprocessor not found at {preprocessor_path}")

                final_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])

                # Save PIPELINE, not just model
                joblib.dump(final_pipeline, os.path.join(self.config.root_dir, self.config.model_name))
                
                # Retrieve parameters from config
                mlflow.log_params({
                    "n_estimators": self.config.n_estimators,
                    "learning_rate": self.config.learning_rate,
                    "class_weight": self.config.class_weight
                })

                # Log PIPELINE
                # We use sklearn flavor now because it's a Pipeline
                # Log model (Disabled here to prevent duplicate artifacts - moved to ModelEvaluation)
            # mlflow.sklearn.log_model(final_pipeline, name="model")
                
                # Save Run ID for Evaluation Step
                run_id = mlflow.active_run().info.run_id
                with open(os.path.join(self.config.root_dir, "run_id.txt"), "w") as f:
                    f.write(run_id)

                logger.info(f"Model trained and saved at: {os.path.join(self.config.root_dir, self.config.model_name)}")
                logger.info(f"Run ID saved to: {os.path.join(self.config.root_dir, 'run_id.txt')}")
                
        except Exception as e:
            raise ChurnException(e, sys)
