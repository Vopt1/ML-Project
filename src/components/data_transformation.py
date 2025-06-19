import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder,RobustScaler,StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()



    def get_data_transformer_obj(self):
        try:

            drop_cols=['Date_of_Journey','Source','Destination','Dep_Time','Arrival_Time']
            cat_col=['Airline', 'Additional_Info']
            num_col=['Total_Stops','Day_of_journey','Month_of_journey','Dep_hour','Dep_minute','Arrival_hour','Arrival_minute','Duration_mins','Arrival_time_sinceMidnight','Dep_time_sinceMidnight','Distance_km']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",RobustScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse_output=False))
                ]
            )

            custom_pipeline= Pipeline(
                steps=[
                    ("Distance Calculation",CityDistanceTransformer(col1='Source',col2='Destination',newcol='Distance_km',user_agent='Small_data_science_project')),
                    ("All date time Transformations",Transform_datetime(date_col='Date_of_Journey',dep_col='Dep_Time',arr_col='Arrival_Time',duration_col='Duration')),
                    ("Preparing for OHE",CustomTransformer(air_col='Airline',tot='Total_Stops',Add_info='Additional_Info')),
                ]
            )

            main_processor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_col),
                ("cat", cat_pipeline, cat_col)
            ],
            remainder='passthrough'
            )

            preproccesor=Pipeline(
                steps=[
                    ('preprocessing',custom_pipeline),
                    ("Main_Processing", main_processor)
                ]
            )
            

            return preproccesor
        except Exception as e:
            raise CustomException(e,sys) from None
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column_name="Price"

            input_feature_train_df=train_df.drop(columns=[target_column_name,],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys) from None

class CityDistanceTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,col1='Source',col2='Destination',newcol='Distance_km',user_agent='Small_data_science_project'):
        """
        Parameters:
        - city_col1: Name of first city column (default: 'Source')
        - city_col2: Name of second city column (default: 'Destination')
        - new_col: Name for new distance column (default: 'Distance_km')
        - user_agent: Unique identifier for geocoding API 
        """
        self.col1 = col1
        self.col2 = col2
        self.newcol = newcol
        self.user_agent = user_agent
        self.city_coords = {}

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X,y)

    def geocode(self,city_name:str):
        geolocator = Nominatim(user_agent=self.user_agent,timeout=10)
        try:
            location = geolocator.geocode(city_name)
            return location
        except Exception as e:
            raise CustomException(e,sys) from None
        
    def fit(self,X,y=None):
        cities = set(X[self.col1]).union(set(X[self.col2]))
        
        for city in cities:
            if city not in self.city_coords:
                self.city_coords[city] = self.geocode(city_name=city)
        logging.info('All cities coordinates found and stored')
        return self
    
    def distance(self,row):
        source = row[self.col1]
        destination = row[self.col2]
        coords1 = self.city_coords.get(source)
        coords2 = self.city_coords.get(destination)

        if coords1 and coords2:
            return geodesic((coords1.latitude,coords1.longitude),(coords2.latitude,coords2.longitude)).km
        return np.nan
    
    
    def transform(self,X,y=None):
        X = X.copy()

        X[self.newcol] = X.apply(self.distance,axis=1).astype(float)
        logging.info('All distances calculated and updated')
        return X

class Transform_datetime(BaseEstimator,TransformerMixin):
    def __init__(self,date_col='Date_of_Journey',dep_col='Dep_Time',arr_col='Arrival_Time',duration_col='Duration'):
        """
        Parameters:
        - date_col: Name of column containing Date of journey (default: 'Date_of_Journey')
        - dep_col: Name of column containing departure time (default: 'Dep_Time')
        - new_col: Name of column containing arrival time (default: 'Arrival_Time')
        - duration_col: Name of column containing duration of flight(default: 'Duration') 
        """
        
        self.date_col=date_col
        self.dep_col=dep_col
        self.arr_col=arr_col
        self.duration_col=duration_col

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X,y)

    def fit(self,X,y=None):
        return self
    
    def transform(self,df,y=None):
        df = df.copy()

        # code for transforming date of journey
        df['Day_of_journey']  = df[self.date_col].str.split('/').str[0]
        df['Month_of_journey']  = df[self.date_col].str.split('/').str[1]
        df['Year_of_journey']  = df[self.date_col].str.split('/').str[2]

        df['Day_of_journey'] = df['Day_of_journey'].astype(int)
        df['Month_of_journey'] = df['Month_of_journey'].astype(int)
        df['Year_of_journey'] = df['Year_of_journey'].astype(int)
        logging.info('Date of Journey transformation completed')

        # code for transforming departure time
        df['Dep_hour'] = df[self.dep_col].str.split(':').str[0]
        df['Dep_minute'] = df[self.dep_col].str.split(':').str[1]
        df['Dep_hour'] = df['Dep_hour'].astype(int)
        df['Dep_minute'] = df['Dep_minute'].astype(int)
        df['Dep_time_sinceMidnight'] = df['Dep_hour']*60 + df['Dep_minute']
        logging.info('Departure time transformation completed')

        # code for transforming arrival time
        df[self.arr_col] = df[self.arr_col].str.split(" ").str[0]
        df['Arrival_hour'] = df[self.arr_col].str.split(":").str[0]
        df['Arrival_minute'] = df[self.arr_col].str.split(":").str[1]
        df['Arrival_hour'] = df['Arrival_hour'].astype(int)
        df['Arrival_minute'] = df['Arrival_minute'].astype(int)
        df['Arrival_time_sinceMidnight'] = df['Arrival_hour']*60+df['Arrival_minute']
        logging.info('Arrival time transformation completed')

        #code for transforming duration
        duration = pd.to_timedelta(df['Duration'])
        for i  in range(0,len(duration)):
            df.loc[i,'Duration_mins'] = duration[i].seconds //60
        logging.info('Duration transformation completed')
        return df.drop(columns=['Date_of_Journey', 'Source', 'Destination', 'Dep_Time',
                           'Arrival_Time', 'Duration', 'Year_of_journey','Route'], errors='ignore')
    
class CustomTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,air_col='Airline',tot='Total_Stops',Add_info='Additional_Info'):
        self.air_col=air_col
        self.tot=tot
        self.Add_info=Add_info

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X,y)

    def fit(self,X,y=None):
        return self

    def airline(self,x):
        if 'Trujet' in x or 'GoAir' in x or 'Multiple carriers' in x:
            return 'Multiple carriers'
        elif 'Jet Airways' in x:
            return 'Jet Airways'
        elif 'Vistara' in x:
            return 'Vistara'
        else: 
            return x

    def transform(self, X, y = None):
        X['Premium'] = X[self.air_col].apply(lambda x: 1 if 'Premium' in x or 'Business' in x else 0)
        X[self.air_col] = X[self.air_col].apply(self.airline)
        X['Total_Stops'] = X['Total_Stops'].map({'non-stop':0,'2 stops':2,'1 stop':1,'3 stops':3,'4 stops':4})
        X['Total_Stops'] = X['Total_Stops'].astype(int)
        X['Additional_Info'] = X['Additional_Info'].apply(lambda x: 'No Info' if 'No info' in x else x)
        logging.info('Transformation of Airline names,Additional Info and Total stops completed')

        return X