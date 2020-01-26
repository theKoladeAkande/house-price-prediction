#features.py 
""" Handle Feature Engineering"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BinaryFeatureGenerator(BaseEstimator, TransformerMixin):
    """generates new a new feature specifying the exixtence of a property as 0
       and non-existence as 1 """

    def __init__(self, variables: list = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        """Nothing to be estimated from trainset,
           method implemented to accomodate sklearn pipeline"""

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """generates a binary feature indicator """

        X = X.copy()

        for feature in self.variables:
            X['has'+feature] = X[feature].apply(lambda x: 1 if x > 0 else 0)
        return X
