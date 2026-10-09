"""Input-data builders with lazy imports for optional EV dependencies."""

from importlib import import_module

__all__ = [
    "vehicle_data",
    "HeatPumpProfileBuilder",
    "HouseholdProfileBuilder",
]


def __getattr__(name):
    if name == "vehicle_data":
        value = import_module(f"{__name__}.vehicle_data")
    elif name == "HeatPumpProfileBuilder":
        value = import_module(f"{__name__}.heat_pump_builder").HeatPumpProfileBuilder
    elif name == "HouseholdProfileBuilder":
        value = import_module(f"{__name__}.household_builder").HouseholdProfileBuilder
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
