from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_results(path: Path) -> pd.DataFrame:
    """Read results.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``data``.
    
    Raises
    ------
    KeyError
        If an explicit validation or processing check in this operation fails."""
    data = pd.read_csv(path, sep=";")
    required = {
        "ch_refuel",
        "run_status",
        "reduce_mode",
        "reduce_factor",
        "soc_min_column",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    for column in (
        "ch_refuel",
        "reduce_factor",
        "optimized_peak_MW",
        "uncontrolled_peak_MW",
        "optimized_to_uncontrolled_peak_ratio",
        "peak_reduction_percent",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "optimized_to_uncontrolled_peak_ratio" in data.columns:
        data["optimized_peak_percent"] = (
            data["optimized_to_uncontrolled_peak_ratio"] * 100.0
        )
    elif {"optimized_peak_MW", "uncontrolled_peak_MW"}.issubset(data.columns):
        data["optimized_peak_percent"] = (
            data["optimized_peak_MW"]
            / data["uncontrolled_peak_MW"]
            * 100.0
        )
    elif "peak_reduction_percent" in data.columns:
        data["optimized_peak_percent"] = 100.0 - data["peak_reduction_percent"]
    else:
        raise KeyError(
            "Cannot calculate the optimized peak relative to the uncontrolled "
            "peak. No suitable result columns were found."
        )

    return data