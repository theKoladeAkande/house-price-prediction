#data_management.py
import pandas as pd 
from sklearn.pipeline import Pipeline
import joblib

from regression_model import __version__  as _version 
from regression_model.config import config
from regression_model.config import logging_config


_logger = logging_config.get_logger(__name__)

def load_dataset(*, filename: str):
    """loads data """
    _data = pd.read_csv(f'{config.DATASET_DIR}/{filename}')
    return _data

def save_pipeline(*, pipeline_to_persist):
    """ persist pipeline for reproducibility"""
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODELS_DIR / save_file_name
    
    remove_old_pipeline(files_to_keep=save_file_name)
    
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f'save_file : {save_file_name}')
    print(f'Saved pipeline {save_file_name}')

def load_pipeline(*, file_name: str):
    file_path = config.TRAINED_MODELS_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path) 
    return saved_pipeline

def remove_old_pipeline(*, files_to_keep):

    for model_file in config.TRAINED_MODELS_DIR.iterdir():
        if model_file.name not in [files_to_keep, '__init__.py']:
            model_file.unlink()