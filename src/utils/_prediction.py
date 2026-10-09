"""Prediction utilities for the EV/V2G analysis workflow."""

import pandas as pd

def prepare_data(model_type: str, X_input: pd.DataFrame, scaler:None) -> pd.DataFrame:
    """
    Prepares the input data for prediction based on the model type.

    Parameters:
    - model_type (str): The type of model ('rigde_regression' or 'XGB' or 'GBM').
    - X_input (pd.DataFrame): The input features for prediction.
    - net_pop_data (pd.DataFrame): The network population data.

    Returns:
    - pd.DataFrame: The prepared data for prediction.
    """
    required_columns = ["plug_in_duration", "minimum_range", "v2g_ratio"]
    missing = [col for col in required_columns if col not in X_input.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Expected columns are: {required_columns}")
    
    X_working = X_input.copy()

    X_working["n_controlled"] = X_working["v2g_ratio"] * 85

    if model_type in ['ridge_regression']:
        X_working['plug_in_duration_sq'] = X_working['plug_in_duration'] ** 2
        X_working["n_controlled_sq"] = X_working["n_controlled"] ** 2
        X = X_working[['plug_in_duration', 'plug_in_duration_sq', 'minimum_range', 'n_controlled', 'n_controlled_sq']]
        if scaler is None:
            raise ValueError("Scaler must be provided for ridge regression model type.")
        X_new_scaled = scaler.transform(X)
        return X_new_scaled
    elif model_type in ['XGB', 'GBM']:
        return X_working[['plug_in_duration', 'minimum_range', 'n_controlled']]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    