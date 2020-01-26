from math import ceil
import numpy as np
from regression_model.processing.data_managment import load_dataset
from regression_model.predict import model_predict

def test_make_single_prediction():
    test_data = load_dataset(filename='test.csv')
    single_data = test_data[0:1].to_json(orient='records')

    subject = model_predict(input_data=single_data)

    assert subject is not None
    assert isinstance(subject.get('predictions')[0], np.float32)
    assert ceil(subject.get('predictions')[0]) == 124265



def test_make_multiple_predictions():
    test_data = load_dataset(filename='test.csv')
    test_data_length = len(test_data)
    multiple_test_data = test_data.to_json(orient='records')

    subject = model_predict(input_data=multiple_test_data)


    assert subject is not None
    assert len(subject.get('predictions')) == 1456


    assert len(subject.get('predictions')) != test_data_length
