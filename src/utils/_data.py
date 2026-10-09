"""Data utilities for the EV/V2G analysis workflow."""

import numpy as np
import pandas as pd
import os
import datetime
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_state(profile: pd.DataFrame, plug_state='plug1', index=None):
    """Return state.
    
    Parameters
    ----------
    profile : pd.DataFrame
        Profile used by this operation.
    plug_state : object
        Plug state used by this operation.
    index : object
        Index used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch."""
    cols = cols_mapping[plug_state]
    if index is None:
        return (profile[cols[0]], profile[cols[1]], 
                profile[cols[2]], profile[cols[3]], 
                profile[cols[4]])
    else:
        return (profile[cols[0]][:index], profile[cols[1]][:index], 
                profile[cols[2]][:index], profile[cols[3]][:index], 
                profile[cols[4]][:index])