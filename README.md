# A deployed machine learning pipeline for predicting house prices.
A machine learning pipeline for predicting house prices this pipeline trains, tests, publishes and serves the model via Rest Api.

## Research 
**file path** *house-price-prediction/ml_research_jupyter_ notebooks/*

Exploratory Data analysis, feature engineering, feature selection, hyperparameter optimization and
model building with xgboost all carried out in a research notebook enviroment.
Essential parameters for the model were persisted to ensure reproducibility.

## The Pipeline
Machine learning pipline is built with the third part scikit-learn's Pipeline module.

The pipeline involves: 

#### Processing
**file path** *house-price-prediction/packages/regression_model/regression_model/processing*

Custom transformer were built using subclassing sklearn's BaseEstimator, Transformermxin.
These transformers handles preprocessing operations on the data including feature engineering, preparing the data for model building.
Data Validations were also added to process data for errors before loading in the data, also the processing aspect includes
saving the pipeline and loading the pipeline.

#### Connected Pipeline
**file path** *house-price-prediction/packages/regression_model/regression_model/pipeline.py*

All the transformers and processing functionality are connected to make up the full pipeline

#### Training
**file path** *house-price-prediction/packages/regression_model/regression_model/train_pipeline.py /

The data is loaded and is passed through the pipeline for transformation which is then trained on the model using persisted 
parameters. The resulting pipeline is persisted for use in prediction


#### Prediction
**file path** house-price-prediction/packages/regression_model/regression_model/predict.py /

The data can be passed to the saved pipeline to obtain predictions


#### Tests.
**file path** *house-price-prediction/packages/regression_model/regression_model/tests/*

Unit tests written with pytest to ensure pipeline's functionality.


## Api
**file path** *house-price-prediction/packages/regression_model/regression_model/ml_api*

Api is built using flask web framework, schema validation with marshmallow on the API's end with marsmallow


## Deployment Practices:
* Package published to gemfury 
* Continous Integration platform (Circle ci) used to automate the process of training, publishing and serving the model.
* Deployment on heroku
