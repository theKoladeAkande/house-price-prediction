#preprocessors.py

""" Transformers for preprocessing"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CategoricalNaNImputer(BaseEstimator, TransformerMixin):
    """ Creates another category in categorical features by filling
        the NaN values with a new category """

    def __init__(self, *, variables: list = None, category: str = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.category = category

    def fit(self, X, y=None):
        """Nothing to be estimated from trainset,
           method implemented to accomodate sklearn pipeline"""

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """fills the missing value of categotrical features
           with the specified category"""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.category)
        return X


class NumeriaclNaNImputer(BaseEstimator, TransformerMixin):
    """Fills the missing values in numerical features with the estimated median 
       from training data"""
    def __init__(self, *, variables: list = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        """Estimates the median from the training data"""
        self.median_dict = {}

        for feature in self.variables:
            self.median_dict[feature] = X[feature].median() 
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """fills the missing value with the estimated median from the training data"""
        X = X.copy()

        for feature in self.variables:
            median = self.median_dict[feature]
            X[feature] = X[feature].fillna(median)
        return X


class RareLabelCategoryImputer(BaseEstimator, TransformerMixin):
    """Replaces observations in categorical feature with the value "rare", with occurences 
       less than a certain percentage """

    def __init__(self, *, variables: list = None, tol: float = 0.01):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        
        self.tol = tol
    
    def fit(self, X: pd.DataFrame, y=None):
        """gets frequent labels from train data"""
        self.frequent_labels_dict = {}

        for feature in self.variables:
           temp = pd.Series(X[feature].value_counts()/ len(X))
           frequent_labels = list(temp[temp > self.tol].index)
           self.frequent_labels_dict[feature] = frequent_labels
 
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """creates new category label rare for observations not conatined in 
           frequent_labels_dict"""

        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.frequent_labels_dict[feature]),X[feature], 'Rare')
        return X


class BinaryFeatureGenerator(BaseEstimator, TransformerMixin):
    """generates new a new feature specifying the exixtence of a property as 0
       and non-existence as 1 """

    def __init__(self, variables: list = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables =variables

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


class CategoricalMonotonicEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variable via monotonic ordering"""

    def __init__(self, *, variables: list = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        """estimates monotonic ordering on the training data"""
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns)+['target']

        self.monotonic_labels_dict = {}

        for feature in self.variables:
            order = temp[[feature, 'target']].groupby(feature).median().sort_values('target').index
            monotonic_label = {keys:idx for idx, keys in enumerate(order, 0)}

            self.monotonic_labels_dict[feature] = monotonic_label

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """" encodes categorical features with monotonic label estimated from
             train data """
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.monotonic_labels_dict[feature])

        if X[self.variables].isnull().any().any():
            counts_null = X[self.variables].isnull().any()
            vars_na = {key: vars for key, vars in counts_null.items() if vars is True}
            raise ValueError(
                            f'Error in encoding features NaN Values introduced for'
                            f'features {vars_na.keys()}')
        return X


class DropFeatures(BaseEstimator, TransformerMixin):
    """Drop Features not needed"""
    def __init__(self, *, drop_features: list = None):
        if not isinstance(drop_features, list):
            self.drop_features = [drop_features]
        else:
            self.drop_features = drop_features

    def fit(self, X, y=None):
        """Nothing to be estimated from training data, 
            fit implemented to accomodate sklearn pipeline"""

        return self


    def transform(self, X: pd.DataFrame, y=None):
        """ drops specified features"""
        X = X.copy()

        X = X.drop(self.drop_features, axis=1)

        return X
