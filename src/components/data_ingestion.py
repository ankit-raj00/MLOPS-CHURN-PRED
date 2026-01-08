import os
import sys
from src.exception import ChurnException
from src.logger import logger
from src.entity.config_entity import DataIngestionConfig
from src.connection.mongodb_client import MongoDBClient
from src.constants import DATABASE_NAME, COLLECTION_NAME
import pandas as pd
from pymongo import MongoClient

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
            self.mongodb_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise ChurnException(e, sys)

    def export_collection_as_dataframe(self):
        try:
            database = self.mongodb_client.database
            collection = database[COLLECTION_NAME]
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            
            df.replace({"na": pd.NA}, inplace=True)
            return df

        except Exception as e:
            raise ChurnException(e, sys)

    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_dataframe()
            logger.info("Exported collection as dataframe")
            
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Saved raw data at: {self.config.raw_data_path}")
            
            # For now, raw and ingested path are same, but usually we might do train/test split here or simple copy
            df.to_csv(self.config.data_file_path, index=False)
            logger.info(f"Saved ingested data at: {self.config.data_file_path}")

        except Exception as e:
            raise ChurnException(e, sys)
