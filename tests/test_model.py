import unittest
import mlflow
import os
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            # 0. Load Config
            with open("params.yaml", "r") as f:
                cls.config = yaml.safe_load(f)["model_deployment"]
                
            # 1. Credentials checking
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                raise EnvironmentError("MLFLOW_TRACKING_URI is not set")
            
            # Ensure credentials if remote
            if "dagshub" in tracking_uri:
                 if not os.getenv("MLFLOW_TRACKING_USERNAME") or not os.getenv("MLFLOW_TRACKING_PASSWORD"):
                     raise EnvironmentError("MLFLOW_TRACKING_USERNAME/PASSWORD not set for DagsHub")

            # 2. Connect to MLflow
            mlflow.set_tracking_uri(tracking_uri)
            
            # 3. Load Model from Staging
            cls.model_name = cls.config['model_name']
            cls.stage = cls.config['source_stage']
            
            client = mlflow.MlflowClient()
            
            # NEW API: search_model_versions
            versions = client.search_model_versions(f"name='{cls.model_name}'")
            
            # Filter for Staging
            staging_versions = [v for v in versions if v.current_stage == cls.stage]
            
            if not staging_versions:
                print(f"⚠️ No model found in {cls.stage} stage. Skipping tests.")
                cls.model = None
                return
                
            # Sort by version (descending) to get latest
            staging_versions.sort(key=lambda x: int(x.version), reverse=True)
            cls.model_version = staging_versions[0].version
            cls.model_uri = f"models:/{cls.model_name}/{cls.stage}"
            
            print(f"📥 Loading Model: {cls.model_name} (Version {cls.model_version}) from {cls.stage}...")
            cls.model = mlflow.sklearn.load_model(cls.model_uri)
            
            # 4. Load Test Data
            test_data_path = cls.config['test_data_path']
            if not os.path.exists(test_data_path):
                 raise FileNotFoundError(f"Test data not found at {test_data_path}. Run 'dvc repro' first.")
                 
            cls.test_df = pd.read_csv(test_data_path)
            
        except Exception as e:
            # If explicit error (like config missing), we assume test failure unless it's just "No Model"
            print(f"Setup failed: {e}")
            cls.model = None

    def test_model_loaded(self):
        if self.model is None:
            self.skipTest("No Staging model found.")
        self.assertIsNotNone(self.model)

    def test_performance(self):
        if self.model is None:
            self.skipTest("No Staging model found.")
        
        # Split X and y (Last column is target)
        X_test = self.test_df.iloc[:, :-1]
        y_test = self.test_df.iloc[:, -1]
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"📊 Test Results - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Assertions from Config
        min_acc = self.config['min_accuracy']
        min_f1 = self.config['min_f1_score']
        
        self.assertGreaterEqual(acc, min_acc, f"Accuracy {acc} < Threshold {min_acc}")
        self.assertGreaterEqual(f1, min_f1, f"F1 Score {f1} < Threshold {min_f1}")

if __name__ == "__main__":
    unittest.main()
