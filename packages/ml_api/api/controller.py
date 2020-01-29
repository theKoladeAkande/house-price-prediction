from flask import Blueprint, request, jsonify
from regression_model.predict import model_predict


from api.config import get_logger

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health staus ok')
        return 'ok'

@prediction_app.route('/v1/predict/regression', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'inputs: {json_data}')

        result = model_predict(input_data=json_data)
        _logger.info(f'outputs: {result}')

        prediction = result.get('predictions')[0]
        version = result.get('version')
        
        _logger.info(f'predictions: {prediction}')

        return jsonify({'predictions': prediction,
                'version': version})
