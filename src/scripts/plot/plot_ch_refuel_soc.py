#!/usr/bin/env python3
r"""Plot the influence of ch_refuel and minimum SoC.

Default comparison
------------------
* Option 1: technical availability, flexible SoC (reference);
* Option 2: behavioral/refuel availability, flexible SoC;
* Option 3: behavioral/refuel availability, limited SoC.

Options 2 and 3 use the same availability profile and therefore isolate the
effect of the minimum-SoC constraint.

Example (Windows cmd.exe)
-------------------------
python src\scripts\plot\plot_ch_refuel_soc.py ^
    --input-csv "data\data_20260923_2\greedy_collected_results\greedy_results_tidy.csv" ^
    --output-dir "data\data_20260923_2\figures"

Required packages: numpy, pandas, matplotlib
Optional package: plotly (for an additional interactive HTML figure)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from .plot_utils import read_results


DEFAULT_REFERENCE_OPTION = 1
DEFAULT_FLEXIBLE_OPTION = 2
DEFAULT_LIMITED_OPTION = 3
RATIO_COLUMN = "optimized_to_uncontrolled_peak_ratio"
PLOT_COLUMN = "optimized_to_uncontrolled_peak_percent"


def read_results(
    path: Path,
    *,
    selected_options: set[int],
) -> pd.DataFrame:
    """Read comma- or semicolon-separated results and validate the data."""
    data = pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig", index_col=False)
    required = {"ch_refuel", "option", RATIO_COLUMN}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    if "run_status" in data.columns:
        status = data["run_status"].astype(str).str.strip().str.lower()
        data = data[status == "success"].copy()

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=list(required)).copy()
    data[PLOT_COLUMN] = data[RATIO_COLUMN] * 100.0
    data["option"] = data["option"].astype(int)
    selected = data[data["option"].isin(selected_options)].copy()

    missing_options = sorted(selected_options - set(selected["option"]))
    if missing_options:
        raise ValueError(
            "Missing successful result rows for Options: "
            f"{missing_options}"
        )

    duplicates = selected.duplicated(
        ["option", "ch_refuel"],
        keep=False,
    )
    if duplicates.any():
        rows = selected.loc[
            duplicates,
            ["option", "ch_refuel"],
        ].drop_duplicates()
        raise ValueError(
            "Multiple successful rows were found for the same option and "
            "ch_refuel. Filter the input to one scenario first:\n"
            + rows.head(10).to_string(index=False)
        )

    return selected.sort_values(["option", "ch_refuel"])


def option_data(data: pd.DataFrame, option: int) -> pd.DataFrame:
    """Return one option sorted by ch_refuel."""
    result = data[data["option"] == option].copy()
    if result.empty:
        raise ValueError(f"No successful values found for Option {option}")
    return result.sort_values("ch_refuel")


def check_reference_is_constant(data: pd.DataFrame, option: int) -> float:
    """Return the reference mean and warn if it is not constant."""
    values = option_data(data, option)[
        PLOT_COLUMN
    ].to_numpy(dtype=float)
    spread = float(np.nanmax(values) - np.nanmin(values))
    if spread > 1e-6:
        print(
            f"Warning: Option {option} is not constant "
            f"(spread={spread:.4f} percentage points). "
            "The mean is used as the reference line."
        )
    return float(np.nanmean(values))


def calculate_x_range(
    flexible: pd.DataFrame,
    limited: pd.DataFrame,
) -> tuple[float, float]:
    """Return a padded x-axis range based on observed ch_refuel values."""
    values = np.concatenate(
        [
            flexible["ch_refuel"].to_numpy(dtype=float),
            limited["ch_refuel"].to_numpy(dtype=float),
        ]
    )
    values = values[np.isfinite(values)]
    lower = max(0.0, float(values.min()) - 0.02)
    upper = min(1.0, float(values.max()) + 0.02)
    return lower, upper


def create_static_figure(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    reference_option: int,
    flexible_option: int,
    limited_option: int,
) -> None:
    """Create publication-ready PNG, SVG and PDF figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9.5,
        }
    )

    flexible = option_data(data, flexible_option)
    limited = option_data(data, limited_option)
    reference = check_reference_is_constant(data, reference_option)
    x_lower, x_upper = calculate_x_range(flexible, limited)
    x_lower *= 100.0
    x_upper *= 100.0

    figure, axis = plt.subplots(
        figsize=(9.4, 6.3),
        constrained_layout=True,
    )

    axis.plot(
        flexible["ch_refuel"]*100.0,
        flexible[PLOT_COLUMN],
        color="#D55E00",
        linewidth=2.7,
        marker="o",
        markersize=6.5,
        label=(
            f"Arrival-SoC-triggered plugging with flexible plugging window"
        ),
        zorder=4,
    )
    axis.plot(
        limited["ch_refuel"]*100,
        limited[PLOT_COLUMN],
        color="#D55E00",
        linewidth=2.7,
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.4,
        markersize=6.5,
        label=(
            f"Arrival-SoC-triggered plugging with fixed plugging window"
        ),
        zorder=4,
    )
    axis.axhline(
        reference,
        color="#0072B2",
        linewidth=2.2,
        linestyle="-.",
        label=(
            f"Always plugged in"
        ),
        zorder=2,
    )

    axis.set_xlabel(r"Plug-in motivation in %")
    axis.set_ylabel(
        "Optimized peak relative to uncontrolled peak in %"
    )
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(0, 100)
    axis.set_xticks(np.arange(10, 101, 10))
    axis.set_yticks(np.arange(0, 101, 10))
    axis.grid(axis="y", color="#BDBDBD", alpha=0.45, linewidth=0.8)
    axis.grid(axis="x", color="#D9D9D9", alpha=0.22, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
        framealpha=0.94,
        columnspacing=1.4,
        handlelength=3.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "ch_refuel_minimum_soc_effect"
    figure.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {stem.with_suffix('.svg')}")
    print(f"Saved: {stem.with_suffix('.pdf')}")


def create_interactive_figure(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    reference_option: int,
    flexible_option: int,
    limited_option: int,
) -> None:
    """Create an optional interactive Plotly HTML figure."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed; interactive HTML export skipped.")
        return

    flexible = option_data(data, flexible_option)
    limited = option_data(data, limited_option)
    reference = check_reference_is_constant(data, reference_option)
    x_lower, x_upper = calculate_x_range(flexible, limited)
    x_lower *= 100.0
    x_upper *= 100.0

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=flexible["ch_refuel"] * 100.0,
            y=flexible[PLOT_COLUMN],
            mode="lines+markers",
            name=(
                f"Arrival-SoC-triggered plugging with flexible plugging window"
            ),
            line={"color": "#D55E00", "width": 3},
            marker={"symbol": "circle", "size": 9},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=limited["ch_refuel"] * 100.0,
            y=limited[PLOT_COLUMN],
            mode="lines+markers",
            name=(
                f"Arrival-SoC-triggered plugging with fixed plugging window"
            ),
            line={"color": "#D55E00", "width": 3, "dash": "dash"},
            marker={"symbol": "square-open", "size": 9},
        )
    )
    figure.add_hline(
        y=reference,
        line={"color": "#0072B2", "width": 2.2, "dash": "dashdot"},
        annotation_text=(
            "Always plugged in"
        ),
        annotation_position="top left",
    )

    figure.update_layout(
        template="plotly_white",
        xaxis={
            "title": "Plug-in motivation",
            "range": [x_lower, x_upper],
            "tick0": 10,
            "dtick": 10,
        },
        yaxis={
            "title": (
                "Optimized peak relative to uncontrolled peak in %"
            ),
            "range": [0, 100],
            "tick0": 0,
            "dtick": 10,
        },
        legend={
            "x": 0.99,
            "xanchor": "right",
            "y": 0.02,
            "yanchor": "bottom",
        },
        width=1100,
        height=680,
        font={"family": "Arial", "size": 14},
        hovermode="x unified",
    )

    output = output_dir / "ch_refuel_minimum_soc_effect.html"
    figure.write_html(output, include_plotlyjs=True)
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the selected settings.
    
    Returns
    -------
    argparse.Namespace
        Result computed as ``parser.parse_args()``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help=(
            "Collected greedy result CSV, for example "
            "greedy_results_tidy.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ch_refuel_soc_figure"),
        help="Output directory (default: ch_refuel_soc_figure).",
    )
    parser.add_argument(
        "--reference-option",
        type=int,
        default=DEFAULT_REFERENCE_OPTION,
        help=f"Technical reference option (default: {DEFAULT_REFERENCE_OPTION}).",
    )
    parser.add_argument(
        "--flexible-option",
        type=int,
        default=DEFAULT_FLEXIBLE_OPTION,
        help=f"Behavioral flexible-SoC option (default: {DEFAULT_FLEXIBLE_OPTION}).",
    )
    parser.add_argument(
        "--limited-option",
        type=int,
        default=DEFAULT_LIMITED_OPTION,
        help=f"Behavioral limited-SoC option (default: {DEFAULT_LIMITED_OPTION}).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip optional interactive Plotly HTML export.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    selected_options = {
        args.reference_option,
        args.flexible_option,
        args.limited_option,
    }
    if len(selected_options) != 3:
        raise ValueError(
            "Reference, flexible and limited option numbers must be different."
        )

    data = read_results(
        input_csv,
        selected_options=selected_options,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "ch_refuel_minimum_soc_selected_data.csv"
    data.to_csv(selected_path, index=False, float_format="%.9f")
    print(f"Saved: {selected_path}")

    create_static_figure(
        data,
        output_dir,
        reference_option=args.reference_option,
        flexible_option=args.flexible_option,
        limited_option=args.limited_option,
    )
    if not args.no_html:
        create_interactive_figure(
            data,
            output_dir,
            reference_option=args.reference_option,
            flexible_option=args.flexible_option,
            limited_option=args.limited_option,
        )


if __name__ == "__main__":
    main()
