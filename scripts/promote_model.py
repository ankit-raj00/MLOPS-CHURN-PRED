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
    
    print(f"🔍 Searching for latest model in '{source_stage}' stage...")
    
    # NEW API: search_model_versions
    versions = client.search_model_versions(f"name='{model_name}'")
    source_versions = [v for v in versions if v.current_stage == source_stage]
    
    if not source_versions:
        print(f"⚠️ No model found in {source_stage}. Skipping promotion.")
        return # Exit gracefully
        
    # Sort descending
    source_versions.sort(key=lambda x: int(x.version), reverse=True)
    version_to_promote = source_versions[0].version
    
    # Check current Production Version
    production_versions = [v for v in versions if v.current_stage == target_stage]
    if production_versions:
        # Sort descending to be safe (though usually only 1 prod)
        production_versions.sort(key=lambda x: int(x.version), reverse=True)
        current_prod_version = production_versions[0].version
        if current_prod_version == version_to_promote:
            print(f"ℹ️ Model Version {version_to_promote} is ALREADY in {target_stage}. Skipping promotion.")
            return

    print(f"🚀 Promoting Model Version {version_to_promote} from {source_stage} to {target_stage}...")
    
    # Transition
    client.transition_model_version_stage(
        name=model_name,
        version=version_to_promote,
        stage=target_stage,
        archive_existing_versions=True # Safely archive old Production model
    )
    
    print(f"✅ Success! Version {version_to_promote} is now in {target_stage}.")

if __name__ == "__main__":
    promote_model()
