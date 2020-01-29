#predict.py
import pandas as pd

from regression_model.config import config
from regression_model.processing.data_managment import load_pipeline
from regression_model.processing.validation import validate_inputs
from regression_model import __version__ as _version
from regression_model.config.logging_config import get_logger

_logger = get_logger(logger_name=__name__)

pipeline_file_name =f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'

loaded_pipeline = load_pipeline(file_name=pipeline_file_name)

def model_predict(*, input_data):
    """makes predictions via the loaded pipeline"""
    raw_test_data = pd.read_json(input_data)
    test_data = validate_inputs(raw_test_data)
    output = loaded_pipeline.predict(test_data[config.FEATURES])
    output = output.astype(float)
    results = {'predictions': output, 'version': _version}

    _logger.info(f'making prediction with model version {_version}'
             f'inputs used for prediction: {test_data}'
             f'prediction results: {results}')

    return results