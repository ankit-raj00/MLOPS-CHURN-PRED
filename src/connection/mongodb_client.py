import pymongo
import os
from src.exception import ChurnException
from src.logger import logger
import sys
from src.entity.config_entity import MongoDBEnvironmentVariable

class MongoDBClient:
    client = None

    def __init__(self, database_name:str) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = MongoDBEnvironmentVariable().mongo_db_url
                if mongo_db_url is None:
                    raise Exception(f"Environment key: MONGO_DB_URL is not set.")
                
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
            
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logger.info(f"Connected to MongoDB database: {self.database_name}")
            
        except Exception as e:
            raise ChurnException(e, sys)
