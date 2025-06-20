import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts',"train.csv")
    test_data_path = os.path.join('artifacts',"test.csv")
    raw_data_path = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def InitiateDataIngestion(self,data_path):
        logging.info('Starting Data Ingestion')
        try:
            df = pd.read_csv(data_path)
            df = df.dropna(axis=0)
            df = df.drop_duplicates(keep='first')
            logging.info("Successfully Imported the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split Initiated')
            train_set,test_set = train_test_split(df,test_size=0.15,random_state=21)

            logging.info('Saving train set and test set')

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)  
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.InitiateDataIngestion()

    data_transformation = DataTransformation()
    train,test,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    # pd.DataFrame(train).to_csv('train.csv')
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train,test))