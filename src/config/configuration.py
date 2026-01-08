from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
from box import ConfigBox

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_file_path=config.data_file_path,
            raw_data_path=config.raw_data_path,
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            transformed_train_path=config.transformed_train_path,
            transformed_test_path=config.transformed_test_path,
            preprocessor_path=config.preprocessor_path
        )
        
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.LightGBM
        mlflow_config = self.params.mlflow_config
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            valid_name=params.valid_name,
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            class_weight=params.class_weight,
            random_state=params.random_state,
            verbosity=params.verbosity,
            mlflow_config=mlflow_config
        )
        
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        mlflow_config = self.params.mlflow_config
        
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name,
            mlflow_config=mlflow_config
        )
        
        return model_evaluation_config
