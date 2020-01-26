#data_management.py
import pandas as pd 
from sklearn.pipeline import Pipeline
import joblib

from regression_model.config import config

def load_dataset(*, filename: str):
    """loads data """
    _data = pd.read_csv(f'{config.DATASET_DIR}/{filename}')
    return _data

def save_pipeline(*, pipeline_to_persist):
    """ persist pipeline for reproducibility"""
    save_file_name = 'xgboost_regression_model.pkl'
    save_path = config.TRAINED_MODELS_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print(f'Saved pipeline {save_file_name}')

def load_pipeline(*, file_name: str):
    file_path = config.TRAINED_MODELS_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path) 
    return saved_pipeline