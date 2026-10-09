#!/usr/bin/env python3
r"""Plot an optimized-peak EV-growth × V2G-participation heatmap.

Example (Windows cmd.exe):
    python src\scripts\plot\plot_ev_v2g_heatmaps.py ^
      --input-dir "data\data_20260928_1\scenarios\hh_1__hp_1__ev_1__Kassel_2050__seed_13" ^
      --output-dir "data\data_20260928_1\scenarios\hh_1__hp_1__ev_1__Kassel_2050__seed_13\ev_v2g_heatmaps" ^
      --option 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

OPTION_LABEL = {
    1: "Always plugged in",
    2: "Behavioral plugging with flexible plugging window",
    3: "Behavioral plugging with fixed plugging window",
}

SUMMARY_PATTERN = re.compile(
    r"(?:^|/)data_\d+_(?P<ev_count>\d+)/"
    r"greedy_results_nctrl(?P<n_controlled>\d+(?:\.\d+)?)/"
    r"greedy_all_options_summary\.csv$",
    re.IGNORECASE,
)

METRICS = {
    "optimized_peak_MW": {
        "title": "Optimized Peak under EV and V2G Growth",
        "colorbar": "Optimized peak in MW",
        "format": ".3f",
        "cmap": "RdYlGn_r",
        "filename": "optimized_feeder_peak_heatmap",
    },
}

HOUSEHOLD_FILE_PATTERN = "household_load*.csv"
HEAT_PUMP_FILE_PATTERN = "heat_pump_profiles*.csv"


def find_one_csv(folder: Path, pattern: str, description: str) -> Path:
    """Find exactly one matching input CSV below *folder*."""
    matches = sorted(folder.expanduser().resolve().rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {description} CSV matching {pattern!r} was found below "
            f"{folder}."
        )
    if len(matches) > 1:
        listed = "\n  - ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple {description} CSV files were found. Specify the file "
            f"explicitly:\n  - {listed}"
        )
    return matches[0]


def calculate_base_peak(
    household_csv: Path,
    heat_pump_csv: Path,
) -> float:
    """Calculate the hourly household + heat-pump feeder peak in MW."""
    household = pd.read_csv(household_csv, sep=";")
    if "time" not in household.columns or "hh_load (MW)" not in household.columns:
        raise KeyError(
            f"{household_csv} must contain 'time' and 'hh_load (MW)'."
        )
    household_time = pd.to_datetime(household["time"], errors="coerce")
    household_power = pd.to_numeric(
        household["hh_load (MW)"], errors="coerce"
    )
    household_series = pd.Series(
        household_power.to_numpy(),
        index=household_time,
        name="household_MW",
    ).dropna()

    heat_pump = pd.read_csv(heat_pump_csv, sep=";")
    if "total_all" not in heat_pump.columns:
        raise KeyError(f"{heat_pump_csv} must contain 'total_all'.")
    time_column = heat_pump.columns[0]
    heat_pump_time = pd.to_datetime(heat_pump[time_column], errors="coerce")
    # total_all is stored in kW and is converted to MW.
    heat_pump_power = (
        pd.to_numeric(heat_pump["total_all"], errors="coerce") / 1000.0
    )
    heat_pump_series = pd.Series(
        heat_pump_power.to_numpy(),
        index=heat_pump_time,
        name="heat_pump_MW",
    ).dropna()

    for series, description in (
        (household_series, "household"),
        (heat_pump_series, "heat-pump"),
    ):
        if series.empty:
            raise ValueError(f"The {description} load series is empty.")
        if series.index.has_duplicates:
            raise ValueError(
                f"The {description} load series contains duplicate timestamps."
            )

    # Power is averaged when converting 15-minute profiles to the hourly
    # resolution used by the feeder optimization.
    household_hourly = household_series.sort_index().resample("1h").mean()
    heat_pump_hourly = heat_pump_series.sort_index().resample("1h").mean()
    combined = pd.concat(
        [household_hourly, heat_pump_hourly], axis=1, join="inner"
    ).dropna()

    if combined.empty:
        raise ValueError(
            "The household and heat-pump files have no matching hourly "
            "timestamps."
        )

    return float(combined.sum(axis=1).max())


def read_result_folder(root: Path) -> pd.DataFrame:
    """Read result folder.
    
    Parameters
    ----------
    root : Path
        Root used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``data.drop_duplicates(keys, keep='last')``.
    
    Raises
    ------
    NotADirectoryError
        If an explicit validation or processing check in this operation fails.
    ValueError
        If an explicit validation or processing check in this operation fails."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")

    frames: list[pd.DataFrame] = []

    for summary_path in root.rglob("greedy_all_options_summary.csv"):
        match = SUMMARY_PATTERN.search(summary_path.as_posix())
        if match is None:
            continue

        frame = pd.read_csv(summary_path, sep=";")
        frame["connected_ev_count"] = int(match.group("ev_count"))
        frame["v2g_share_requested"] = float(
            match.group("n_controlled")
        )
        frame["source_root"] = str(root)
        frame["source_file"] = str(summary_path)
        frames.append(frame)

    if not frames:
        raise ValueError(
            "No greedy_all_options_summary.csv files matching the expected "
            f"folder structure were found below: {root}"
        )

    data = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "option",
        "connected_ev_count",
        "v2g_share_requested",
        "controlled_vehicle_count",
        "n_controlled_realized",
        "uncontrolled_peak_MW",
        "optimized_peak_MW",
        "peak_reduction_percent",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Duplicate scenario trees must contain identical numerical results.
    keys = ["connected_ev_count", "v2g_share_requested", "option"]
    for metric in ("optimized_peak_MW",):
        spread = data.groupby(keys, dropna=False)[metric].agg(
            lambda values: values.max() - values.min()
        )
        if spread.fillna(0).gt(1e-9).any():
            raise ValueError(
                f"Duplicate scenarios contain inconsistent {metric} values."
            )

    return data.drop_duplicates(keys, keep="last")


def build_matrices(
    data: pd.DataFrame,
    *,
    option: int,
    metric: str,
    base_peak_mw: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return optimized values and displayed values on the complete grid.

    Unsuccessful or infeasible optimized results remain NaN in the first
    matrix. In the second matrix, those gaps are filled with the corresponding
    uncontrolled feeder peak whenever that value is available. The second
    matrix is used for both heatmap colors and annotations.
    """
    row_index = sorted(
        set(data["connected_ev_count"].dropna().unique()).union({0})
    )
    # Always include a 0% V2G baseline, even when the input folders start at
    # 10%. Its value is filled with the uncontrolled peak for each EV count.
    column_index = sorted(
        set(data["v2g_share_requested"].dropna().unique()).union({0.0})
    )

    selected = data[data["option"].eq(option)].copy()
    successful = selected.copy()

    if "run_status" in selected.columns:
        successful = successful[
            successful["run_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("success")
        ]
    if "all_steps_feasible" in successful.columns:
        successful = successful[
            successful["all_steps_feasible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        ]

    if selected.empty:
        raise ValueError(f"No results found for Option {option}.")

    if base_peak_mw is None or not np.isfinite(base_peak_mw):
        raise ValueError("A finite household + heat-pump peak is required.")
    if base_peak_mw < 0:
        raise ValueError("The household + heat-pump peak must be non-negative.")

    optimized_matrix = successful.pivot(
        index="connected_ev_count",
        columns="v2g_share_requested",
        values=metric,
    ).reindex(index=row_index, columns=column_index)

    # The uncontrolled peak is independent of the selected optimization
    # option. Use all available option rows so that it can also fill a cell
    # when the requested option failed but another option was saved.
    uncontrolled = data.dropna(subset=["uncontrolled_peak_MW"]).copy()
    uncontrolled_spread = uncontrolled.groupby(
        ["connected_ev_count", "v2g_share_requested"], dropna=False
    )["uncontrolled_peak_MW"].agg(lambda values: values.max() - values.min())
    if uncontrolled_spread.fillna(0).gt(1e-9).any():
        raise ValueError(
            "The same EV-count/participation cell contains inconsistent "
            "uncontrolled_peak_MW values."
        )

    uncontrolled_matrix = uncontrolled.pivot_table(
        index="connected_ev_count",
        columns="v2g_share_requested",
        values="uncontrolled_peak_MW",
        aggfunc="first",
    ).reindex(index=row_index, columns=column_index)

    # For every EV count, use its own uncontrolled peak at 0% participation.
    # Prefer the selected option and its lowest available participation rate;
    # fall back to any other option only when that option has no baseline.
    selected_reference = (
        selected.dropna(subset=["uncontrolled_peak_MW"])
        .sort_values(["connected_ev_count", "v2g_share_requested"])
        .drop_duplicates("connected_ev_count", keep="first")
        .set_index("connected_ev_count")["uncontrolled_peak_MW"]
    )
    fallback_reference = (
        uncontrolled.sort_values(
            ["connected_ev_count", "v2g_share_requested", "option"]
        )
        .drop_duplicates("connected_ev_count", keep="first")
        .set_index("connected_ev_count")["uncontrolled_peak_MW"]
    )
    uncontrolled_reference = selected_reference.combine_first(
        fallback_reference
    )

    for ev_count in row_index:
        if ev_count in uncontrolled_reference.index:
            reference_peak = uncontrolled_reference.loc[ev_count]

            if np.isfinite(reference_peak):
                uncontrolled_matrix.loc[ev_count, :] = reference_peak

    # With no EVs, every participation rate represents the same feeder load:
    # household load plus heat-pump load.
    optimized_matrix.loc[0, :] = base_peak_mw
    uncontrolled_matrix.loc[0, :] = base_peak_mw

    annotation_matrix = optimized_matrix.fillna(uncontrolled_matrix)
    return optimized_matrix, annotation_matrix, float(base_peak_mw)


def contrast_text_color(value: float, minimum: float, maximum: float) -> str:
    """Select a readable text color for a heatmap cell.
    
    Parameters
    ----------
    value : float
        Value used by this operation.
    minimum : float
        Minimum used by this operation.
    maximum : float
        Maximum used by this operation.
    
    Returns
    -------
    str
        Computed result for the selected input and execution branch."""
    if not np.isfinite(value) or maximum <= minimum:
        return "black"
    normalized = (value - minimum) / (maximum - minimum)
    return "white" if normalized < 0.18 or normalized > 0.82 else "black"


def plot_heatmap(
    matrix: pd.DataFrame,
    annotation_matrix: pd.DataFrame,
    *,
    base_peak_mw: float,
    option: int,
    metric: str,
    output_dir: Path,
) -> None:
    """Plot heatmap.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix used by this operation.
    annotation_matrix : pd.DataFrame
        Annotation matrix used by this operation.
    base_peak_mw : float
        Base peak mw used by this operation.
    option : int
        Option used by this operation.
    metric : str
        Metric used by this operation.
    output_dir : Path
        Directory for generated outputs.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    settings = METRICS[metric]
    values = matrix.to_numpy(dtype=float)
    displayed_values = annotation_matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(displayed_values)
    cmap = plt.get_cmap(settings["cmap"]).copy()
    cmap.set_bad("#D9D9D9")

    figure, axis = plt.subplots(figsize=(10.2, 7.0))
    image = axis.imshow(masked, aspect="auto", cmap=cmap, origin="lower")

    x_labels = [f"{value * 100:.0f}" for value in matrix.columns]
    y_labels = [f"{int(value)}" for value in matrix.index]
    axis.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    axis.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    axis.set_xlabel("V2G participation rate in %")
    axis.set_ylabel("Number of connected EVs")
    axis.set_title(f"{settings['title']}\n {OPTION_LABEL[option]}", pad=12)

    finite = displayed_values[np.isfinite(displayed_values)]
    if finite.size == 0:
        raise ValueError("No feeder-peak values are available for the heatmap.")
    minimum = float(finite.min())
    maximum = float(finite.max())
    value_format = settings["format"]
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            displayed_value = displayed_values[row, column]
            if np.isfinite(displayed_value):
                label = format(displayed_value, value_format)
                color = contrast_text_color(
                    displayed_value, minimum, maximum
                )
            else:
                label = "×"
                color = "#666666"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
            )

            # Outline cells whose successful optimized peak is below the
            # feeder peak without EVs. Failed/infeasible cells filled with the
            # uncontrolled peak are deliberately not outlined.
            optimized_value = values[row, column]
            if (
                np.isfinite(optimized_value)
                and optimized_value < base_peak_mw - 1e-9
            ):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=2.4,
                        zorder=4,
                    )
                )

    outline_legend = Rectangle(
        (0, 0),
        1,
        1,
        fill=False,
        edgecolor="#111111",
        linewidth=2.4,
        label="Optimized peak below EV=0 reference",
    )
    figure.legend(
        handles=[outline_legend],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.012),
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        fontsize=8.5,
    )

    colorbar = figure.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label(settings["colorbar"])
    axis.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)

    figure.tight_layout(rect=(0, 0.075, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"option_{option}_{settings['filename']}"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot the optimized feeder peak for combinations of connected "
            "EV count and V2G participation rate."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root folder containing the EV-count and n_controlled folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for PNG, PDF and extracted CSV files.",
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing household_load*.csv and "
            "heat_pump_profiles*.csv. Defaults to --input-dir."
        ),
    )
    parser.add_argument(
        "--household-load-csv",
        type=Path,
        default=None,
        help="Explicit household-load CSV; use together with --heat-pump-csv.",
    )
    parser.add_argument(
        "--heat-pump-csv",
        type=Path,
        default=None,
        help="Explicit heat-pump CSV; use together with --household-load-csv.",
    )
    parser.add_argument(
        "--option",
        type=int,
        default=1,
        help="Option to plot (default: 1).",
    )
    parser.add_argument(
        "--base-peak-mw",
        type=float,
        default=None,
        help=(
            "Optional direct household + heat-pump peak in MW. If omitted, "
            "the peak is calculated from the two load-profile CSV files."
        ),
    )
    args = parser.parse_args()

    if (args.household_load_csv is None) != (args.heat_pump_csv is None):
        parser.error(
            "--household-load-csv and --heat-pump-csv must be used together."
        )

    if args.base_peak_mw is not None:
        base_peak_mw = args.base_peak_mw
        household_csv = None
        heat_pump_csv = None
    else:
        if args.household_load_csv is not None:
            household_csv = args.household_load_csv.expanduser().resolve()
            heat_pump_csv = args.heat_pump_csv.expanduser().resolve()
        else:
            load_dir = (
                args.load_dir if args.load_dir is not None else args.input_dir
            )
            household_csv = find_one_csv(
                load_dir, HOUSEHOLD_FILE_PATTERN, "household-load"
            )
            heat_pump_csv = find_one_csv(
                load_dir, HEAT_PUMP_FILE_PATTERN, "heat-pump"
            )

        base_peak_mw = calculate_base_peak(household_csv, heat_pump_csv)
        print(f"Household load: {household_csv}")
        print(f"Heat-pump load: {heat_pump_csv}")

    print(f"EV=0 reference peak: {base_peak_mw:.6f} MW")

    data = read_result_folder(args.input_dir)
    selected_rows = data[data["option"].eq(args.option)].copy()
    if selected_rows.empty:
        available = sorted(data["option"].dropna().astype(int).unique())
        parser.error(
            f"Option {args.option} not found. Available options: {available}"
        )

    for metric in METRICS:
        matrix, annotation_matrix, base_peak_mw = build_matrices(
            data,
            option=args.option,
            metric=metric,
            base_peak_mw=base_peak_mw,
        )
        plot_heatmap(
            matrix,
            annotation_matrix,
            base_peak_mw=base_peak_mw,
            option=args.option,
            metric=metric,
            output_dir=args.output_dir,
        )

    csv_path = args.output_dir / f"option_{args.option}_ev_v2g_results.csv"
    selected_rows.sort_values(
        ["connected_ev_count", "v2g_share_requested"]
    ).to_csv(csv_path, sep=";", index=False, float_format="%.6f")

    print(f"Saved optimized-peak heatmap and data to: {args.output_dir}")


if __name__ == "__main__":
    main()