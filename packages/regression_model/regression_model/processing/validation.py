#validation.py
import pandas as pd
from regression_model.config import config


def validate_inputs(input_data: pd.DataFrame):
    """"Validate the input data to ensure right data for the preprocessing"""
   
    validated_data = input_data.copy()

    #Validate for categorical data with NA which wasn't in training
    if validated_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data =  validated_data.dropna(
                            axis=0,
                            subset=config.CATEGORICAL_NA_NOT_ALLOWED
        )
    

    #Validate for numerical data with NA which wasn't in training 
    if validated_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
                            axis=0,
                            subset=config.NUMERICAL_NA_NOT_ALLOWED
        )
    
    return validated_data