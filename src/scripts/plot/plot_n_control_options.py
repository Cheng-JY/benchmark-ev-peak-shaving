#!/usr/bin/env python3
r"""Plot peak-load reduction for Options 1, 2 and 6 versus n_control.

The script searches recursively for files named
``greedy_all_options_summary.csv`` below ``--data-root``.  The x-axis uses the
realized controlled-EV share from ``n_controlled_realized`` by default.

Example (Windows cmd.exe)
-------------------------
python src\scripts\plot\plot_n_control_options.py ^
    --data-root "data\data_20260921_1\scenarios\hh_1__hp_1__ev_1__Kassel_2050__seed_13\data_20260921_357" ^
    --output-dir "data\data_20260921_1\scenarios\hh_1__hp_1__ev_1__Kassel_2050__seed_13\data_20260921_357\figures"

Required packages: pandas, matplotlib
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


OPTIONS = (1, 2)
OPTION_STYLES = {
    1: {
        "label": "Always plugged in",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    2: {
        "label": "Arrival-SoC-triggered plugging with flexible plugging window",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-",
    },
    6: {
        "label": "Arrival-SoC-triggered plugging with flexible plugging window",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "--",
    },
}


def read_summary(path: Path) -> pd.DataFrame:
    """Read one comma- or semicolon-separated summary file."""
    data = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    required = {
        "option",
        "n_controlled_realized",
        "peak_reduction_percent",
        "run_status",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")

    match = re.search(r"greedy_results_nctrl([0-9]+(?:\.[0-9]+)?)", str(path))
    data["n_controlled_nominal"] = (
        float(match.group(1)) if match else float("nan")
    )
    data["source_file"] = str(path)
    return data


def add_zero_participation_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic (0%, 100%) baseline when an option has no 0% result."""
    baseline_rows: list[pd.Series] = []

    for option in OPTIONS:
        option_data = data[data["option"].eq(option)]
        if option_data.empty or option_data["n_controlled_realized"].eq(0.0).any():
            continue

        baseline = option_data.iloc[0].copy()
        baseline["n_controlled_realized"] = 0.0
        baseline["n_controlled_nominal"] = 0.0
        baseline["peak_reduction_percent"] = 0.0
        baseline["optimized_peak_relative_percent"] = 100.0
        baseline["run_status"] = "synthetic_baseline"
        baseline["source_file"] = "synthetic 0% V2G baseline"
        baseline_rows.append(baseline)

    if baseline_rows:
        data = pd.concat([data, pd.DataFrame(baseline_rows)], ignore_index=True)

    return data.sort_values(["option", "n_controlled_realized"])


def collect_results(data_root: Path) -> tuple[pd.DataFrame, int]:
    """Collect successful Option 1/2/6 results from all n_control folders."""
    summary_paths = sorted(data_root.rglob("greedy_all_options_summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(
            f"No greedy_all_options_summary.csv files found below {data_root}"
        )

    combined = pd.concat([read_summary(path) for path in summary_paths], ignore_index=True)
    combined["option"] = pd.to_numeric(combined["option"], errors="coerce")
    combined["n_controlled_realized"] = pd.to_numeric(
        combined["n_controlled_realized"], errors="coerce"
    )
    combined["peak_reduction_percent"] = pd.to_numeric(
        combined["peak_reduction_percent"], errors="coerce"
    )
    status = combined["run_status"].astype(str).str.strip().str.lower()

    selected = combined[combined["option"].isin(OPTIONS) & status.eq("success")].copy()
    selected = selected.dropna(
        subset=["option", "n_controlled_realized", "peak_reduction_percent"]
    )
    selected["option"] = selected["option"].astype(int)

    duplicates = selected.duplicated(
        subset=["option", "n_controlled_realized"], keep=False
    )
    if duplicates.any():
        duplicate_rows = selected.loc[
            duplicates, ["option", "n_controlled_realized", "source_file"]
        ]
        raise ValueError(
            "Multiple results found for the same option and controlled share. "
            "Use a data root containing only one scenario:\n"
            + duplicate_rows.to_string(index=False)
        )

    # peak reduction = 100 * (1 - optimized / uncontrolled)
    selected["optimized_peak_relative_percent"] = (
        100.0 - selected["peak_reduction_percent"]
    )
    selected = add_zero_participation_baselines(selected)
    return selected, len(summary_paths)


def create_figure(data: pd.DataFrame, output_dir: Path) -> None:
    """Create PNG, PDF and SVG versions of the combined figure."""
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

    figure, axis = plt.subplots(figsize=(9.4, 6.3), constrained_layout=True)
    missing_options: list[int] = []

    for option in OPTIONS:
        option_data = data[data["option"].eq(option)].sort_values(
            "n_controlled_realized"
        )
        if option_data.empty:
            missing_options.append(option)
            print(f"Warning: Option {option} has no successful results and is not plotted.")
            continue

        style = OPTION_STYLES[option]
        axis.plot(
            option_data["n_controlled_realized"] * 100.0,
            option_data["optimized_peak_relative_percent"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            markersize=6.5,
        )

    if data.empty:
        raise ValueError("No successful results are available for Options 1, 2 or 6.")

    if missing_options:
        text = ", ".join(f"Option {option}" for option in missing_options)
        axis.text(
            0.02,
            0.97,
            f"No successful results: {text}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#555555",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#BDBDBD",
            },
        )

    axis.set_xlim(-2, 102)
    axis.set_xticks(range(0, 101, 10))
    axis.set_ylim(0, 105)
    axis.set_yticks(range(0, 101, 10))

    axis.set_title("Influence of V2G Participation on Optimized Peak")
    axis.set_xlabel("Realized V2G participation rate in %")
    axis.set_ylabel("Optimized peak relative to uncontrolled peak in %")
    axis.grid(axis="y", color="#BDBDBD", alpha=0.45, linewidth=0.8)
    axis.grid(axis="x", color="#D9D9D9", alpha=0.22, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="best", frameon=True, framealpha=0.94)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "optimized_peak_relative_to_uncontrolled_vs_v2g_participation"
    figure.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

    for suffix in (".png", ".pdf", ".svg"):
        print(f"Saved: {stem.with_suffix(suffix)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the selected settings.
    
    Returns
    -------
    argparse.Namespace
        Result computed as ``parser.parse_args()``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root folder containing greedy_results_nctrl* directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("n_control_figure"),
        help="Output directory (default: n_control_figure).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    data, summary_count = collect_results(data_root)
    print(f"Read {summary_count} summary files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "options_1_2_6_vs_v2g_participation.csv"
    data.to_csv(selected_path, index=False, float_format="%.9f")
    print(f"Saved: {selected_path}")

    create_figure(data, output_dir)


if __name__ == "__main__":
    main()
