"""Linear regressor utilities for the EV/V2G analysis workflow."""

# linear_regressor.py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from .base_regressor import BaseRegressor

class LinearRegressor(BaseRegressor):
    """Represent linear regressor and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        super().__init__(scenario_data, scenario_config)
        self.model_type = "linear"

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
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LinearRegression()
        self.model.fit(X_scaled, y)

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
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)