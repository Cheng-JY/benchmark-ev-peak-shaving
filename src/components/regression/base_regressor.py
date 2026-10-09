"""Base regressor utilities for the EV/V2G analysis workflow."""

import pandas as pd
import pickle

class BaseRegressor:

    """Represent base regressor and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.cfg = scenario_config

        self.summary_path = scenario_data.analysis_dir / "summary.csv"
        self.save_dir = scenario_data.regression_dir

        self.model = None
        self.scaler = None
        self.model_type = None

    def load_data(self):
        """Load data.
        
        Returns
        -------
        object
            Result computed as ``(X, y, df)``."""
        df = pd.read_csv(self.summary_path, sep=";")

        # drop NaN or inconsistent rows
        df = df.dropna(subset=["reduced_relevant"])

        # Feature selection
        FEATURE_COLS = [
            "plug_in_duration",
            "minimum_range",
            "n_controlled",
        ]

        X = df[FEATURE_COLS]
        y = df["reduced_relevant"]

        return X, y, df
    
    def fit(self, X, y):
        """Fit the estimator or feature transformer using the supplied training data.
        
        Parameters
        ----------
        X : object
            Input feature matrix.
        y : object
            Target values corresponding to the input samples.
        
        Raises
        ------
        NotImplementedError
            If an explicit validation or processing check in this operation fails."""
        raise NotImplementedError
    
    def predict(self, X):
        """Generate predictions using the fitted model.
        
        Parameters
        ----------
        X : object
            Input feature matrix.
        
        Raises
        ------
        NotImplementedError
            If an explicit validation or processing check in this operation fails."""
        raise NotImplementedError
    
    def save(self, name: str):
        """Save model + scaler"""
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
        }
        fp = self.save_dir / f"{name}.pkl"
        with open(fp, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved: {fp}")

    def load_pre_train_model(self, name: str):
        """Load pre train model.
        
        Parameters
        ----------
        name : str
            Name used by this operation."""
        model_path = self.save_dir / f"{name}.pkl"
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.scaler = payload.get("scaler", None)
        self.model_type = payload.get("model_type", None)

    


    
