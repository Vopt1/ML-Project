from src.exception import CustomException
from src.logger import logging
import os
import pickle
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from typing import Dict,Any,Union
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore',category=ConvergenceWarning)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        if obj is None:
            raise ValueError("Cannot save None object!")
            
        try:
            _ = dill.dumps(obj)
        except Exception as e:
            raise ValueError(f"Object not serializable: {str(e)}")

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            raise RuntimeError("Saved file is empty!")

    except Exception as e:
        raise CustomException(e, sys)
    
class ModelEvaluation:
    def __init__(self):
        pass

    def model_evalute(self,X_train:Union[np.ndarray,pd.DataFrame],y_train:Union[np.ndarray,pd.DataFrame],
                    X_test:Union[np.ndarray,pd.DataFrame],y_test:Union[np.ndarray,pd.DataFrame],
                    models_params:Dict[str,Dict[Any,Any]],verbose:bool = False):  
        try:
            report = {}

            for model_name,mp in models_params.items():
                if verbose:
                    print(f"Starting model tuning for {model_name}")
                model = mp['model']
                param_grid = mp['params']
                grid = GridSearchCV(model,param_grid=param_grid,scoring='r2',n_jobs=-1,cv=5,refit=True)
                grid.fit(X_train,y_train)
                if isinstance(X_train,pd.DataFrame):
                    N = len(X_test)
                    P = len(X_test.columns)
                else:
                    N = X_test.shape[0]
                    P = X_test.shape[1]
                
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

                score = self.adjusted_r2_score(y_test,y_pred,N=N,P=P)
                report[model_name] = {'estimator':best_model,
                                    'score':score}
            
            return report
        
        except Exception as e:
            logging.error(f'{e} error in utils evaluate during model evaluation')
            raise CustomException(e,sys)
        
    def adjusted_r2_score(self,
                    y_true:Union[pd.DataFrame,np.ndarray],
                    y_pred:Union[pd.DataFrame,np.ndarray],
                    N:int,
                    P:int):
        '''This function calculates and returns the adjusted r2 score
            - y_true: the correct values
            - y_pred: values predicted by model
            - N: Total no of samples
            - P: Total no of features'''
        if len(y_true)!=len(y_pred):
            raise ValueError(f"Number of predictions (len(y_pred)={len(y_pred)}) does not match number of true labels (len(y_true)={len(y_true)}).")
        R2 = r2_score(y_true=y_true,y_pred=y_pred)
        adjusted_r2 = 1-(((1-R2)*(N-1))/(N-P-1))
        return adjusted_r2
    
def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")
            
        if os.path.getsize(file_path) == 0:
            raise ValueError("File is empty")
            
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
            
        if obj is None:
            raise ValueError("Loaded None object")
            
        return obj
    except Exception as e:
        raise CustomException(e, sys)