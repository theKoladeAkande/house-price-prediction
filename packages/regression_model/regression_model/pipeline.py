#pipeline.py
"""Module to construct pipline """
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from regression_model.config import config
from regression_model import preprocessors as pps



#load xgboost persisted parameters with numpy
xgboost_params = np.load(config.XGBOOST_PARAMS_FILE, allow_pickle=True)
#load convert from array to dictionary
xgboost_params_dict = xgboost_params.tolist()

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

# Numerical features with na
NUMERICAL_FEATURES_WITH_NA = ['LotFrontage']

FEATURES_FOR_FEATURE_GENERATION = ['Fireplaces']

#Features to drop
DROP_FEATURES = ['Fireplaces']




house_price_pipeline = Pipeline(
        [
        ('categorical_nan_imputer', 
        pps.CategoricalNaNImputer(variables=CATEGORICAL_FEATURES_WITH_NA,category='MissingValue')),
        ('numerical_na_imputer',
        pps.NumeriaclNaNImputer(variables=NUMERICAL_FEATURES_WITH_NA)),
        ('rare_label_encoder',
        pps.RareLabelCategoryImputer(variables=CATEGORICAL_FEATURES,tol=0.01)),
        ('cateogrical_encoder',
        pps.CategoricalMonotonicEncoder(variables=CATEGORICAL_FEATURES)),
        ('binary_feature_generator',
        pps.BinaryFeatureGenerator(variables=FEATURES_FOR_FEATURE_GENERATION)),
        ('drop_features',
        pps.DropFeatures(drop_features=DROP_FEATURES)),
        ('xgboost_regression_model',
        XGBRegressor(seed=42, objective='reg:squarederror', **xgboost_params_dict)) 
        ])
