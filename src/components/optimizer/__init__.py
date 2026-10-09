"""
The :mod:`jcheng.components.optimizer` module
"""


from importlib import import_module

__all__ = [
    "BaseEVOptimizer",
    "GreedyEVOptimizer",
    "OptimizationConfig",
    "PyomoBasicEVOptimizer",
    "PyomoBidirectionalEVOptimizer"
]

_MODULE_BY_NAME = {
    "BaseEVOptimizer": "base_optimizer",
    "GreedyEVOptimizer": "greedy_optimizer",
    "OptimizationConfig": "optimization_config",
    "PyomoBasicEVOptimizer": "milp_basic_optimizer",
    "PyomoBidirectionalEVOptimizer": "milp_bidi_optimizer",
}


def __getattr__(name):
    """Load only the optimizer implementation requested by the caller."""

    try:
        module_name = _MODULE_BY_NAME[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value

