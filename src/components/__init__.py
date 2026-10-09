"""
The :mod:`jcheng.components` module
"""

from importlib import import_module

__all__ = [
    "aggregation",
    "analysis",
    "input_data",
    "optimizer",
    "regression",
    "scenario",
]

def __getattr__(name):
    """Load component packages only when they are actually requested.

    In particular, importing the standalone CSV optimizer must not import
    ``input_data.household_builder`` because that module initializes the
    project database connection.
    """

    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")