#train_pipeline.py

import pathlib

import joblib

import pandas as  pd

from sklearn.model_selection import train_test_split

import pipeline

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

DATASET_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODELS_DIR = PACKAGE_ROOT / 'trained_models'

TRAIN_DATA_FILE = DATASET_DIR / 'train.csv'
TEST_DATA_FILE = DATASET_DIR / 'test.csv'

TARGET = 'SalePrice'

FEATURES = ['OverallQual',
            'GarageCars',
            'ExterQual',
            'BsmtQual',
            'GrLivArea',
            'Neighborhood',
            'FullBath',
            'KitchenQual',
            'KitchenAbvGr',
            '2ndFlrSF',
            'PoolArea',
            '1stFlrSF',
            'GarageCond',
            'TotRmsAbvGrd',
            'GarageQual',
            'BsmtFinSF1',
            'TotalBsmtSF',
            'GarageFinish',
            'CentralAir',
            'FireplaceQu',
            'LandContour',
            'LotFrontage',
#temporal feature to generate 'hasFireplaces' feature            
            'Fireplaces']

def save_pipeline(*, pipeline_to_persist):
    """ persist pipeline for reproducibility"""
    save_file_name = 'xgboost_regression_model.pkl'
    save_path = TRAINED_MODELS_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    
    print('Pipeline persisted')



def train_model():
    """ runs the training piple line"""

    data = pd.read_csv(TRAIN_DATA_FILE)
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        data[FEATURES], data[TARGET],
                                        test_size=0.15,
                                        random_state=42,
                                        shuffle=True
                                           )
    
    pipeline.house_price_pipeline.fit(X_train[FEATURES],y_train)

    save_pipeline(pipeline_to_persist=pipeline.house_price_pipeline)
    
    print("Training model...")


if __name__ == '__main__':
    train_model()