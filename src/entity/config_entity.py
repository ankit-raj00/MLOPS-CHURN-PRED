from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_file_path: Path
    raw_data_path: Path

@dataclass
class MongoDBEnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    valid_name: str
    n_estimators: int
    learning_rate: float
    class_weight: str
    random_state: int
    verbosity: int
    mlflow_config: dict

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    mlflow_config: dict
