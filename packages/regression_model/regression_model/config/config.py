#connfig.py
""" contains constant needed for functional pipeline"""
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

PIPELINE_NAME = 'xgboost_model_pipeline'

# the categorical feature hasFirePlace not included is a generated feature
CATEGORICAL_FEATURES = ['ExterQual',
                        'BsmtQual',
                        'Neighborhood',
                        'KitchenQual',
                        'GarageCond',
                        'GarageQual',
                        'GarageFinish',
                        'CentralAir',
                        'FireplaceQu',
                        'LandContour']

#categorical features with na
CATEGORICAL_FEATURES_WITH_NA = ['BsmtQual', 
                                'GarageCond', 
                                'GarageQual', 
                                'GarageFinish', 
                                'FireplaceQu']

CATEGORICAL_NA_NOT_ALLOWED = [ 'CentralAir', 
                               'ExterQual', 
                               'KitchenQual', 
                               'LandContour',
                               'Neighborhood' ]

# Numerical features with na
NUMERICAL_FEATURES_WITH_NA = ['LotFrontage']

NUMERICAL_NA_NOT_ALLOWED = ['1stFlrSF',
                            '2ndFlrSF',
                            'BsmtFinSF1',
                            'FullBath',
                            'GarageCars',
                            'GrLivArea',
                            'KitchenAbvGr',
                            'OverallQual',
                            'PoolArea',
                            'TotRmsAbvGrd',
                            'TotalBsmtSF']

FEATURES_FOR_FEATURE_GENERATION = ['Fireplaces']

#Features to drop
DROP_FEATURES = ['Fireplaces']
