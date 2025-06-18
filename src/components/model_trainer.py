import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,model_evalute

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
        trained_model_path=os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Spliting train and test array')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'Ridge Regression': {
                    'model': Ridge(random_state=45),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0],
                        'solver': ['auto', 'svd', 'cholesky']
                    }
                },
                'Lasso Regression': {
                    'model': Lasso(random_state=45),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0],
                        'selection': ['cyclic', 'random']
                    }
                },
                'Decision Tree': {
                    'model': DecisionTreeRegressor(random_state=45),
                    'params': {
                        'max_depth': [None, 5, 10],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=45),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_features': ['sqrt', 'log2']
                    }
                },
                'AdaBoost': {
                    'model': AdaBoostRegressor(random_state=45),
                    'params': {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.01, 0.1, 1.0]
                    }
                },
                'Gradient Boost': {
                    'model': GradientBoostingRegressor(random_state=45),
                    'params': {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5]
                    }
                },
                'XG Boost': {
                    'model': XGBRegressor(random_state=45),
                    'params': {
                        'n_estimators':[338,340,342],
                        'learning_rate':[0.1],
                        'max_depth':[None],
                        'gamma':[1],
                        'subsample':[0.8],
                        'random_state':[45],
                        'colsample_bytree':[0.9],
                        'minimum_child_weight':[1,2]
                    }
                }
            }

            model_report:dict=model_evalute(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models_params=models)

            best_model_name = max(model_report, key=lambda k: model_report[k]['score'])
            best_model_info = model_report[best_model_name]

            if best_model_info['score']<0.6:
                logging.info('No model scored above 0.6')
                raise CustomException('No best model found',sys)
            
            logging.info('Found the best model')

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model_info['estimator']
            )

            return best_model_info['score']

        except Exception as e:
            raise CustomException(e,sys) from None