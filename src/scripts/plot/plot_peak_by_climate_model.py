#!/usr/bin/env python3
r"""Plot uncontrolled and optimized peak load for one option and climate models.

The script reads the authoritative ``greedy_all_options_summary.csv`` in each
``greedy_results_YEAR`` directory.  The uncontrolled peak therefore follows
the baseline stored by the optimizer; by default the script verifies that this
baseline uses ``ch_refuel_cp``.

Examples (Windows cmd.exe)
--------------------------
python plot_peak_by_climate_model.py ^
  "data\data_20260827.zip" ^
  --option 1

python ecv2g\user_dirs\jcheng\scripts\plot_peak_by_climate_model.py ^
  "data\data_20260917" ^
  --option 2 ^
  --output "data\data_20260917\option2_peak_by_climate_model.pdf"
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 2009 is intentionally omitted because its EV input is byte-identical to 2010
# in the analyzed archive and some option definitions are inconsistent.
DEFAULT_MODEL_YEARS: list[tuple[str, int]] = [
    ("reference", 2012),
    ("aladin_26", 2050),
    ("aladin_85", 2035),
    ("cosmo_85", 2044),
    ("pecd_cmcc", 2058),
    ("pecd_ecec", 2059),
    ("pecd_mpim", 2039),
]

OPTION_COLORS = {
    1: "#2F6DAA",
    2: "#4B9B76",
}


def locate_data_root(base: Path) -> Path:
    """Find the directory containing ``greedy_results_YEAR`` folders."""
    candidates = [base, *[path for path in base.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if list(candidate.glob("greedy_results_*")):
            return candidate
    raise FileNotFoundError(
        f"No directory containing greedy_results_YEAR folders found under {base}."
    )


def parse_model_year(values: list[str]) -> list[tuple[str, int]]:
    """Parse repeated ``MODEL=YEAR`` arguments."""
    if not values:
        return DEFAULT_MODEL_YEARS.copy()

    parsed: list[tuple[str, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --model-year {value!r}; expected MODEL=YEAR."
            )
        model, year_text = value.rsplit("=", 1)
        model = model.strip()
        if not model:
            raise ValueError(f"Empty model name in --model-year {value!r}.")
        try:
            year = int(year_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid year in --model-year {value!r}."
            ) from exc
        parsed.append((model, year))
    return parsed


def read_option_rows(
    data_root: Path,
    option: int,
    model_years: list[tuple[str, int]],
    require_ch_refuel_baseline: bool,
) -> pd.DataFrame:
    """Read one successful option row for every requested model/year."""
    rows: list[dict[str, object]] = []
    required = {
        "option",
        "uncontrolled_peak_MW",
        "optimized_peak_MW",
        "peak_reduction_percent",
    }

    for model, year in model_years:
        summary_path = (
            data_root
            / f"greedy_results_{year}"
            / "greedy_all_options_summary.csv"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Summary not found: {summary_path}")

        summary = pd.read_csv(summary_path, sep=";", low_memory=False)
        missing = sorted(required - set(summary.columns))
        if missing:
            raise KeyError(f"{summary_path}: missing columns {missing}")

        summary["option"] = pd.to_numeric(summary["option"], errors="coerce")
        selected = summary.loc[summary["option"].eq(option)].copy()
        if "run_status" in selected.columns:
            selected = selected.loc[selected["run_status"].eq("success")]
        if len(selected) != 1:
            raise ValueError(
                f"{summary_path}: expected one successful row for option {option}, "
                f"found {len(selected)}."
            )

        row = selected.iloc[0]
        baseline_column = str(row.get("uncontrolled_charging_column", ""))
        if require_ch_refuel_baseline and baseline_column != "ch_refuel_cp":
            raise ValueError(
                f"{summary_path}: uncontrolled baseline is {baseline_column!r}, "
                "not 'ch_refuel_cp'. Use --allow-other-baseline only if this is "
                "intentional."
            )

        rows.append(
            {
                "model": model,
                "year": year,
                "uncontrolled_peak_MW": float(row["uncontrolled_peak_MW"]),
                "optimized_peak_MW": float(row["optimized_peak_MW"]),
                "peak_reduction_percent": float(row["peak_reduction_percent"]),
                "uncontrolled_charging_column": baseline_column,
            }
        )

    return pd.DataFrame(rows)


def plot_peak_comparison(data: pd.DataFrame, option: int, output: Path) -> None:
    """Create the paired-bar figure."""
    models = data["model"].tolist()
    years = data["year"].astype(int).tolist()
    uncontrolled = data["uncontrolled_peak_MW"].to_numpy(dtype=float)
    optimized = data["optimized_peak_MW"].to_numpy(dtype=float)
    reduction = data["peak_reduction_percent"].to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(11.6, 6.5))
    x = np.arange(len(models))
    width = 0.34
    option_color = OPTION_COLORS.get(option, "#E38B2C")

    bars_uncontrolled = ax.bar(
        x - width / 2,
        uncontrolled,
        width,
        label="Uncontrolled peak",
        color="#A7ADB4",
        edgecolor="white",
        linewidth=0.8,
    )
    bars_optimized = ax.bar(
        x + width / 2,
        optimized,
        width,
        label=f"Optimized peak",
        color=option_color,
        edgecolor="white",
        linewidth=0.8,
    )

    for bars, values in (
        (bars_uncontrolled, uncontrolled),
        (bars_optimized, optimized),
    ):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#303030",
            )

    reduction_color = "#1F4F7B" if option == 1 else "#347255"
    for index, (original, controlled, percent) in enumerate(
        zip(uncontrolled, optimized, reduction)
    ):
        ax.text(
            index,
            max(original, controlled) + 0.028,
            f"−{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=reduction_color,
        )

    ymax = max(0.31, float(uncontrolled.max()) * 1.20)
    tick_step = 0.05
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + tick_step / 2, tick_step))
    ax.set_title(
        f"Uncontrolled and optimized peak load by climate model",
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Peak load (MW)")
    ax.set_xlabel("Climate model and selected cold year")
    ax.set_xticks(
        x,
        [f"{model}\n({year})" for model, year in zip(models, years)],
    )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    """Run .
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed configuration or command-line arguments.
    
    Returns
    -------
    Path
        Result computed as ``output``.
    
    Raises
    ------
    FileNotFoundError
        If an explicit validation or processing check in this operation fails."""
    input_path = args.input.expanduser().resolve()
    model_years = parse_model_year(args.model_year)

    with tempfile.TemporaryDirectory(prefix="climate_peak_plot_") as temp_dir:
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            extract_dir = Path(temp_dir) / "extracted"
            with zipfile.ZipFile(input_path) as archive:
                archive.extractall(extract_dir)
            data_root = locate_data_root(extract_dir)
        elif input_path.is_dir():
            data_root = locate_data_root(input_path)
        else:
            raise FileNotFoundError(f"Input does not exist: {input_path}")

        data = read_option_rows(
            data_root=data_root,
            option=args.option,
            model_years=model_years,
            require_ch_refuel_baseline=not args.allow_other_baseline,
        )

        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else Path.cwd() / f"option{args.option}_peak_by_climate_model.png"
        )
        plot_peak_comparison(data, args.option, output)

    return output


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot uncontrolled and optimized peak load for one option across "
            "selected climate-model years."
        )
    )
    parser.add_argument("input", type=Path, help="Result ZIP or extracted directory")
    parser.add_argument("--option", type=int, required=True, help="Option number")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--model-year",
        action="append",
        default=[],
        metavar="MODEL=YEAR",
        help=(
            "Override the default model/year list; repeat once per model, e.g. "
            "--model-year aladin_26=2023"
        ),
    )
    parser.add_argument(
        "--allow-other-baseline",
        action="store_true",
        help="Do not require uncontrolled_charging_column to be ch_refuel_cp",
    )
    args = parser.parse_args()

    output = run(args)
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
