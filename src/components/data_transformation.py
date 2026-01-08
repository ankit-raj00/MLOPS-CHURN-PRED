import sys
from src.exception import ChurnException
from src.logger import logger
from src.entity.config_entity import DataTransformationConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        try:
            # Load Data
            df = pd.read_csv(self.config.data_path)
            logger.info("Loaded data for transformation")

            # Drop ID
            if 'customer_id' in df.columns:
                df.drop("customer_id", axis=1, inplace=True)

            # --- PREPROCESSING START (Consistent with Analysis) ---
            # 1. Fill NA
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna("Unknown")
                else:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())

            # 2. Preprocessing
            from src.utils.transformers import FeaturePreprocessor
            preprocessor = FeaturePreprocessor()
            df_processed = preprocessor.fit_transform(df)
            
            # Save Preprocessor object (for Pipeline construction later)
            joblib.dump(preprocessor, self.config.preprocessor_path)
            logger.info("Preprocessing complete and objects saved")
            
            # Update df to processed version
            df = df_processed

            # 3. Train Test Split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save
            train_df.to_csv(self.config.transformed_train_path, index=False)
            test_df.to_csv(self.config.transformed_test_path, index=False)
            
            logger.info(f"Train data saved at: {self.config.transformed_train_path}")
            logger.info(f"Test data saved at: {self.config.transformed_test_path}")

        except Exception as e:
            raise ChurnException(e, sys)
