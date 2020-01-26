#train_pipeline.py

from sklearn.model_selection import train_test_split

import pipeline
from regression_model.processing.data_managment import load_dataset, save_pipeline
from regression_model.config import config

def train_model():
    """ runs the training piple line"""

    data = load_dataset(filename=config.TRAIN_DATA_FILE)
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        data[config.FEATURES], 
                                        data[config.TARGET],
                                        test_size=0.15,
                                        random_state=42,
                                        shuffle=True
                                           )
    
    pipeline.house_price_pipeline.fit(X_train[config.FEATURES],y_train)

    save_pipeline(pipeline_to_persist=pipeline.house_price_pipeline)
    
    print("Training model...")


if __name__ == '__main__':
    train_model()