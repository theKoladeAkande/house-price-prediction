import json
from math import ceil


from regression_model.config import config 
from regression_model.processing.data_managment import load_dataset
from regression_model import __version__ as _version




def test_endpoint_health_returns_200(flask_test_client):

    response = flask_test_client.get('/health')

    assert response.status_code == 200

def test_endpoint_prediction_returns_predictions(flask_test_client):
    test_data = load_dataset(filename=config.TEST_DATA_FILE)
    json_test_data = test_data[0:1].to_json(orient='records')


    response = flask_test_client.post('/v1/predict/regression',
                                      json=json_test_data)


    assert response.status_code == 200

    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    version = response_json['version']

    assert ceil(prediction) == 124265
    assert version == _version