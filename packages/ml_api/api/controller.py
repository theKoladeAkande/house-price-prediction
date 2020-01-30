from flask import Blueprint, request, jsonify

from regression_model.predict import model_predict
from regression_model import __version__ as model_version

from api import __version__ as _version
from api.config import get_logger
from api.validation import validate_inputs

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health staus ok')
        return 'ok'

@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'api_version': _version,
                    'model_version': model_version})


@prediction_app.route('/v1/predict/regression', methods=['POST'])
def predict():
    if request.method == 'POST':
        not_validated_json_data = request.get_json()
        #_logger.info(f'inputs: {not_validated_json_data}')


        json_data,errors = validate_inputs(input_data=not_validated_json_data)

        result = model_predict(input_data=json_data)
        _logger.info(f'outputs: {result}')

        prediction = result.get('predictions').tolist()
        version = result.get('version')
        
        _logger.info(f'predictions: {prediction}')

        return jsonify({'predictions': prediction,
                'version': version,
                'errors': errors})
