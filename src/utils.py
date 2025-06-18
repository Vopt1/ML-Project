from src.exception import CustomException
from src.logger import logging
import os
import pickle
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from typing import Dict,Any,Union
import numpy as np
import pandas as pd


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def model_evalute(X_train:Union[np.ndarray,pd.DataFrame],y_train:Union[np.ndarray,pd.DataFrame],
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
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            score = r2_score(y_test,y_pred)
            report[model_name] = {'estimator':best_model,
                                  'score':score}
        
        return report
    
    except Exception as e:
        logging.info(f'{e} error in utils evaluate during model evaluation')
        raise CustomException(e,sys)