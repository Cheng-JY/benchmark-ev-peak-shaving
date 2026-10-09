#!/usr/bin/env python3
r"""Plot flat and peak availability-reduction results for one ch_refuel value.

The y-axis reports the optimized feeder peak as a percentage of the
uncontrolled feeder peak.  A conceptual no-flexibility reference point is
added at retention factor 0 with a value of 100%.

Example (Windows cmd.exe):
    python ecv2g\user_dirs\jcheng\scripts\plot_ch_refuel_reduce_factor.py ^
      --input-csv "data\data_20260923_2\greedy_collected_results\greedy_results_tidy.csv" ^
      --output-dir "data\data_20260923_2\plots" ^
      --ch-refuel 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CH_REFUEL = 0.9
# TARGET_SOC_MIN_COLUMNS = {
#     "soc_min_lim_refuel_cp": "Behavioral plugging with fixed charging window",
#     "soc_min_lim_refuel_targetsoc_cp": (
#         "Behavioral plugging with fixed plugging window"
#     ),
# }

TARGET_SOC_MIN_COLUMNS = {
    "soc_min_refuel_cp": "Arrival-SoC-triggered plugging with flexible plugging window",

}

COLORS = {"flat": "#0072B2", "peak": "#D55E00"}
MARKERS = {"flat": "o", "peak": "s"}
LABELS = {"flat": "Uniform reduction", "peak": "Peak-weighted reduction"}
# UNREDUCED_AVAILABILITY_COLUMN = "ch_avail_refuel_targetsoc_cp"
UNREDUCED_AVAILABILITY_COLUMN = "ch_avail_refuel_cp"

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


def select_series(
    data: pd.DataFrame,
    soc_min_column: str,
    *,
    ch_refuel: float,
) -> pd.DataFrame:
    """Select series.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data used by this operation.
    soc_min_column : str
        Soc min column used by this operation.
    ch_refuel : float
        Plug-in motivation parameter associated with the scenario.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``selected.sort_values(['reduce_mode', 'reduce_factor'])``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    selected = data[
        data["ch_refuel"].sub(ch_refuel).abs().le(1e-9)
        & data["soc_min_column"].eq(soc_min_column)
        & data["reduce_mode"].isin(["flat", "peak"])
        & data["reduce_factor"].notna()
    ].copy()

    successful = selected["run_status"].astype(str).str.lower().eq("success")
    if "all_steps_feasible" in selected.columns:
        feasible = (
            selected["all_steps_feasible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        successful &= feasible

    # Treat failed or infeasible runs as no peak reduction, as requested.
    # Hence their optimized peak equals 100% of the uncontrolled peak.
    selected["plot_status"] = np.where(successful, "success", "failed")
    selected.loc[~successful, "optimized_peak_percent"] = 100.0
    selected = selected[selected["optimized_peak_percent"].notna()].copy()

    # At factor 1.0, use the corresponding unreduced
    # ch_avail_refuel_targetsoc_cp result for both curves.
    unreduced = data[
        data["ch_refuel"].sub(ch_refuel).abs().le(1e-9)
        & data["soc_min_column"].eq(soc_min_column)
        & data["availability_column"].eq(UNREDUCED_AVAILABILITY_COLUMN)
        & data["run_status"].astype(str).str.lower().eq("success")
        & data["optimized_peak_percent"].notna()
    ].copy()

    if "all_steps_feasible" in unreduced.columns:
        unreduced_feasible = (
            unreduced["all_steps_feasible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        unreduced = unreduced[unreduced_feasible].copy()

    if unreduced.empty:
        raise ValueError(
            f"No successful unreduced {UNREDUCED_AVAILABILITY_COLUMN} "
            f"result found for {soc_min_column}."
        )
    if len(unreduced) > 1:
        raise ValueError(
            f"Multiple unreduced reference rows found for {soc_min_column}."
        )

    reference_rows = []
    for mode in ("flat", "peak"):
        reference = unreduced.copy()
        reference["reduce_mode"] = mode
        reference["reduce_factor"] = 1.0
        reference["plot_status"] = "unreduced reference"
        reference_rows.append(reference)

    selected = pd.concat([selected, *reference_rows], ignore_index=True)

    if selected.empty:
        raise ValueError(
            f"No ch_refuel={ch_refuel} reduction results found for "
            f"{soc_min_column}."
        )

    return selected.sort_values(["reduce_mode", "reduce_factor"])


def plot_one(
    selected: pd.DataFrame,
    *,
    title: str,
    ch_refuel: float,
    output_stem: Path,
) -> None:
    """Plot one.
    
    Parameters
    ----------
    selected : pd.DataFrame
        Selected used by this operation.
    title : str
        Title used by this operation.
    ch_refuel : float
        Plug-in motivation parameter associated with the scenario.
    output_stem : Path
        Output stem used by this operation.
    
    Returns
    -------
    None"""
    fig, axis = plt.subplots(figsize=(8.2, 5.0))

    for mode in ("flat", "peak"):
        mode_data = selected[selected["reduce_mode"].eq(mode)]
        x_values = np.r_[100.0,(1.0 - mode_data["reduce_factor"].to_numpy(float)) * 100.0,]
        y_values = np.r_[100.0, mode_data["optimized_peak_percent"].to_numpy(float)]

        axis.plot(
            x_values,
            y_values,
            color=COLORS[mode],
            marker=MARKERS[mode],
            linewidth=2.2,
            markersize=6.5,
            label=LABELS[mode],
        )

    # axis.set_title(f"{title}", pad=12)
    axis.set_xlabel("Reduction factor in %")
    axis.set_ylabel("Optimized peak relative to uncontrolled peak in %")
    axis.set_xlim(102, -2)
    axis.set_ylim(0, 102)

    axis.set_xticks(np.arange(100, -1, -10))
    axis.set_yticks(np.arange(0, 101, 10))
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=True)

    # fig.text(
    #     0.5,
    #     0.012,
    #     "Factor 0 and failed/infeasible runs are 100%; factor 1 uses the unreduced target-SoC availability.",
    #     ha="center",
    #     va="bottom",
    #     fontsize=8.5,
    #     color="#555555",
    # )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot flat and peak availability-retention results for "
            "one ch_refuel value and two limited minimum-SoC definitions."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input greedy_results_tidy.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the PNG, PDF and plotted-values CSV outputs.",
    )
    parser.add_argument(
        "--ch-refuel",
        type=float,
        default=DEFAULT_CH_REFUEL,
        help=f"ch_refuel value to plot (default: {DEFAULT_CH_REFUEL}).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.ch_refuel <= 1.0:
        parser.error("--ch-refuel must be between 0 and 1.")

    data = read_results(args.input_csv)
    output_rows = []

    for soc_min_column, title in TARGET_SOC_MIN_COLUMNS.items():
        selected = select_series(
            data,
            soc_min_column,
            ch_refuel=args.ch_refuel,
        )
        output_rows.append(selected)
        stem = args.output_dir / soc_min_column.removesuffix("_cp")
        plot_one(
            selected,
            title=title,
            ch_refuel=args.ch_refuel,
            output_stem=stem,
        )
        print(f"Saved: {stem.with_suffix('.png')}")
        print(f"Saved: {stem.with_suffix('.pdf')}")

    plotted = pd.concat(output_rows, ignore_index=True)[
        [
            "ch_refuel",
            "soc_min_column",
            "reduce_mode",
            "reduce_factor",
            "plot_status",
            "optimized_peak_MW",
            "uncontrolled_peak_MW",
            "optimized_peak_percent",
        ]
    ]
    ch_refuel_tag = str(float(args.ch_refuel))
    table_path = (
        args.output_dir
        / f"ch_refuel_{ch_refuel_tag}_plotted_values.csv"
    )
    plotted.to_csv(table_path, sep=";", index=False, float_format="%.6f")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
