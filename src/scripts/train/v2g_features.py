"""Polynomial terms with binary variables restricted to exponent one."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

class BinaryAwarePolynomialFeatures(TransformerMixin, BaseEstimator):
    """Represent binary aware polynomial features and its associated operations."""
    def __init__(self, degree=3, binary_indices=(5,)):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        degree : object
            Degree used by this operation.
        binary_indices : object
            Binary indices used by this operation."""
        self.degree = degree
        self.binary_indices = binary_indices

    def fit(self, X, y=None):
        """Fit the estimator or feature transformer using the supplied training data.
        
        Parameters
        ----------
        X : object
            Input feature matrix.
        y : object
            Target values corresponding to the input samples.
        
        Returns
        -------
        object
            Result computed as ``self``."""
        self.poly_ = PolynomialFeatures(degree=self.degree, include_bias=False).fit(X)
        self.n_features_in_ = self.poly_.n_features_in_
        self.keep_ = np.all(self.poly_.powers_[:, list(self.binary_indices)] <= 1, axis=1)
        self.powers_ = self.poly_.powers_[self.keep_]
        return self

    def transform(self, X):
        """Transform the input data into the feature representation used by the model.
        
        Parameters
        ----------
        X : object
            Input feature matrix.
        
        Returns
        -------
        object
            Result computed as ``self.poly_.transform(X)[:, self.keep_]``."""
        check_is_fitted(self, 'keep_')
        return self.poly_.transform(X)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        """Return the names of the output features.
        
        Parameters
        ----------
        input_features : object
            Input features used by this operation.
        
        Returns
        -------
        object
            Result computed as ``self.poly_.get_feature_names_out(input_features)[self.keep_]``."""
        check_is_fitted(self, 'keep_')
        return self.poly_.get_feature_names_out(input_features)[self.keep_]
