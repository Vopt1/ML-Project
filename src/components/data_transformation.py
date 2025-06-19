import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_preprocess import DataPreprocessor

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()



    def get_data_transformer_obj(self):
        try:

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

            preproccesor = ColumnTransformer(
                transformers=[
                    ('Categorical',cat_pipeline,cat_col),
                    ('Numerical',num_pipeline,num_col)
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

            logging.info('Preparing Data for transformations')
            data_preprocessor = DataPreprocessor()
            input_feature_train_df=data_preprocessor.initiate_preprocessing(input_feature_train_df)
            input_feature_test_df=data_preprocessor.initiate_preprocessing(input_feature_test_df)
            logging.info('Data ready for column transformations')

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_obj()

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("Training data transformed")
            logging.error(f'{hasattr(preprocessing_obj, '_is_fitted')}')
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            input_feature_test_df.to_csv('train.csv',index=False)

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