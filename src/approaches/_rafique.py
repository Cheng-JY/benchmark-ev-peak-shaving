import logging
import os

import numpy as np
import pandas as pd
from linopy import Model, merge
from linopy.solvers import available_solvers

logger = logging.getLogger(__name__)

# read variables
lookup = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "data", "lookup.csv"),
    index_col=["component", "variable"],
    )

def define_objectives(m:Model, sns:pd.Index) -> None:
    pass

def define_variables():
    pass

def define_constraints():
    pass

def create_model(
    sns: pd.Index, 
    ) -> Model:
    
    # Create a model instance
    model = Model(name="rafique")
    model.parametes = model.parameters.assign(snapshot=sns)

    # Define variables
    # Define variables and constraints
    define_variables(m, sns, lookup)
    define_constraints(m, sns, lookup)
    
    # Define objectives
    define_objectives(m, sns)
    
    # Merge the model with the solver
    m = merge(m, solver=solver, solver_options=solver_options)
    
    return m

    

