import sys
import pandas as pd
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
from src.components.data_preprocess import DataPreprocessor
from datetime import datetime

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            logging.info(f"Input features:\n{features}")
            preprocessor = load_object(file_path=preprocessor_path)
            preprocess = DataPreprocessor()
            features = preprocess.initiate_preprocessing(features)
            data_scaled = preprocessor.transform(features)
            logging.info(f"Transformed data shape: {data_scaled.shape}")
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys) from None
        
class CustomData:
    def __init__(self,
                 Airline:str,
                 Date_of_Journey:str,
                 Source:str,
                 Destination:str,
                 Dep_Time:str,
                 Arrival_Time:str,
                 Duration:str,
                 Total_Stops:str,
                 Additional_Info:str):
        self.Airline = Airline
        self.Date_of_Journey = Date_of_Journey
        self.Source = Source
        self.Destination = Destination
        self.Dep_Time = Dep_Time
        self.Arrival_Time = Arrival_Time
        self.Duration = Duration
        self.Total_Stops = Total_Stops
        self.Additional_Info = Additional_Info

    def get_data_as_data_frame(self):
        try:
            parsed_date = datetime.strptime(self.Date_of_Journey,'%Y-%m-%d')
            formated_time = datetime.strftime(parsed_date,'%d/%m/%Y')
            custom_data_input_dict = {
                "Airline": self.Airline,
                "Date_of_Journey": formated_time,
                "Source": self.Source,
                "Destination": self.Destination,
                "Dep_Time": self.Dep_Time,
                "Arrival_Time": self.Arrival_Time,
                "Duration": self.Duration,
                "Total_Stops": self.Total_Stops,
                "Additional_Info": self.Additional_Info
            }

            return pd.DataFrame(custom_data_input_dict,index=[0])

        except Exception as e:
            raise CustomException(e, sys)