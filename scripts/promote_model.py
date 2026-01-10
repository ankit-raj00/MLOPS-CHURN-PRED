import mlflow
import os
import sys
import yaml

def promote_model():
    # 0. Load Config
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["model_deployment"]

    model_name = config['model_name']
    source_stage = config['source_stage']
    target_stage = config['target_stage']
    
    # Credentials check
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("❌ Error: MLFLOW_TRACKING_URI not set.")
        sys.exit(1)
        
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    
    print(f"🔍 Searching for model with alias '{source_stage}'...")
    
    try:
        source_model = client.get_model_version_by_alias(model_name, source_stage)
        version_to_promote = source_model.version
    except:
        print(f"⚠️ No model found with alias '{source_stage}'. Skipping promotion.")
        return # Exit gracefully
        
    # Check current Production Version via Alias
    try:
        production_model = client.get_model_version_by_alias(model_name, target_stage)
        if production_model.version == version_to_promote:
            print(f"ℹ️ Model Version {version_to_promote} is ALREADY tagged as '{target_stage}'. Skipping promotion.")
            return
    except:
        # No production model exists yet, that's fine
        pass

    print(f"🚀 Promoting Model Version {version_to_promote} to alias '{target_stage}'...")
    
    # Assign Alias
    client.set_registered_model_alias(model_name, target_stage, version_to_promote)
    
    print(f"✅ Success! Version {version_to_promote} is now aliased as '{target_stage}'.")

if __name__ == "__main__":
    promote_model()
