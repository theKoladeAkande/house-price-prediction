#predict.py
import pandas as pd

from regression_model.config import config
from regression_model.processing.data_managment import load_pipeline

pipeline_file_name ='xgboost_regression_model.pkl'

loaded_pipeline = load_pipeline(file_name=pipeline_file_name)

def model_predict(*, input_data):
    """makes predictions via the loaded pipeline"""
    test_data = pd.read_json(input_data)
    output = loaded_pipeline.predict(test_data[config.FEATURES])
    response = {'predictions': output}

    return response