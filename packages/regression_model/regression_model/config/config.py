#connfig.py
import pathlib
import regression_model



PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent

DATASET_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODELS_DIR = PACKAGE_ROOT / 'trained_models'

XGBOOST_PARAMS_DIR = PACKAGE_ROOT / 'xgboost_params'

XGBOOST_PARAMS_FILE = XGBOOST_PARAMS_DIR  / 'xgb_regression_params.npy'


TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'

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