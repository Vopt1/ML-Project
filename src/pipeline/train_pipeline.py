from pathlib import Path
from src.exception import CustomException
import sys
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def initiate_training():

        while True:
            file_path = Path(input("Enter the file path: ")).expanduser().resolve()

            if file_path.exists() and file_path.is_file():
                print('File found beignning Data ingestion and model training')
                break
            else:
                print('Either file path incorrect or file does not exist\nCheck and reenter')
        
        print('Beiginning Data Ingestion')
        obj = DataIngestion()
        train_data,test_data = obj.InitiateDataIngestion(file_path)
        print('Data Ingestion completed')

        print('Beiginning Data Transformations')
        data_transformation = DataTransformation()
        train,test,_ = data_transformation.initiate_data_transformation(train_data,test_data)
        print('Data Transformations Completed')

        print('Beiginning Model Selection and training')
        modeltrainer = ModelTrainer()
        print(f'The best score was: {modeltrainer.initiate_model_trainer(train,test)}')
        print('Model Selection and training completed and models saved')

if __name__ == '__main__':
    TrainPipeline.initiate_training()