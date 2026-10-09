"""Gbm regressor utilities for the EV/V2G analysis workflow."""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from .base_regressor import BaseRegressor

class GBMRegressor(BaseRegressor):
    """Represent gbmregressor and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        super().__init__(scenario_data, scenario_config)
        self.model_type = "GBM"

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
            Result computed as ``{'train_r2': self.model.score(X, y)}``."""
        param_grid = {
            'n_estimators': [20, 40, 60, 80, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
        }

        grid = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_grid=param_grid,
            scoring="r2",
            cv=5,
            n_jobs=-1
        )

        grid.fit(X, y)
        self.model = grid.best_estimator_

        return {
            "train_r2": self.model.score(X, y)
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
            Result computed as ``self.model.predict(X)``."""
        return self.model.predict(X)