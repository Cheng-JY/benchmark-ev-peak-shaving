"""Ridge regressor utilities for the EV/V2G analysis workflow."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from .base_regressor import BaseRegressor

class RidgeRegressor(BaseRegressor):
    """Represent ridge regressor and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        super().__init__(scenario_data, scenario_config)
        self.model_type = "ridge"

    def fit(self, X, y):
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
            Result computed as ``{'train_r2': self.model.score(X_scaled, y)}``."""
        X['plug_in_duration_sq'] = X['plug_in_duration'] ** 2
        X['n_controlled_sq'] = X['n_controlled'] ** 2

        X = X[['plug_in_duration', 'plug_in_duration_sq', 'minimum_range', 'n_controlled', 'n_controlled_sq']]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        ridge = Ridge()
        param_grid = {
            'alpha': np.logspace(-2, 2, 20),
        }
        grid = GridSearchCV(ridge, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_scaled, y)

        self.model = grid.best_estimator_

        return {
            "train_r2": self.model.score(X_scaled, y)
        }

    def predict(self, X):
        """Generate predictions using the fitted model.
        
        Parameters
        ----------
        X : object
            Input feature matrix.
        
        Returns
        -------
        object
            Result computed as ``self.model.predict(X_scaled)``."""
        X['plug_in_duration_sq'] = X['plug_in_duration'] ** 2
        X['n_controlled_sq'] = X['n_controlled'] ** 2

        X = X[['plug_in_duration', 'plug_in_duration_sq', 'minimum_range', 'n_controlled', 'n_controlled_sq']]

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)