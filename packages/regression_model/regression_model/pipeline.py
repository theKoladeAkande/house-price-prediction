#pipeline.py
"""Module to construct pipline """
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from regression_model.config import config
from regression_model.processing import preprocessors as pps
from regression_model.processing import features


#load xgboost persisted parameters with numpy
xgboost_params = np.load(config.XGBOOST_PARAMS_FILE, allow_pickle=True)
#load convert from array to dictionary
xgboost_params_dict = xgboost_params.tolist()

house_price_pipeline = Pipeline(
        [
        ('categorical_nan_imputer', 
        pps.CategoricalNaNImputer(variables=config.CATEGORICAL_FEATURES_WITH_NA,
                                                category='MissingValue')),
        ('numerical_na_imputer',
        pps.NumeriaclNaNImputer(variables=config.NUMERICAL_FEATURES_WITH_NA)),
        ('rare_label_encoder',
        pps.RareLabelCategoryImputer(variables=config.CATEGORICAL_FEATURES, tol=0.01)),
        ('cateogrical_encoder',
        pps.CategoricalMonotonicEncoder(variables=config.CATEGORICAL_FEATURES)),
        ('binary_feature_generator',
        features.BinaryFeatureGenerator(variables=config.FEATURES_FOR_FEATURE_GENERATION)),
        ('drop_features',
        pps.DropFeatures(drop_features=config.DROP_FEATURES)),
        ('xgboost_regression_model',
        XGBRegressor(seed=42, objective='reg:squarederror', **xgboost_params_dict)) 
        ])
