#!/usr/bin/env python3
r"""Heatmap of V2G participation rate, heat-pump count, and optimized peak.

Default analysis: Behavioral plugging (option 2), 73 EVs. Cells with an
optimized peak at or below the feeder peak limit receive a black border. If
``--peak-limit`` is omitted, the limit is the uncontrolled peak at 0 HP.

Windows example:

    python src\scripts\plot\plot_participation_hp_heatmap.py ^
      --input-dir "data\data_20260928_1\scenarios" ^
      --output-dir "data\data_20260928_1\scenarios\participation_hp_heatmap"^
      --ev-count 97
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


SUMMARY_NAME = "greedy_all_options_summary.csv"

DEFAULT_HP_COUNT_MAP = {
    0.0: 1,
    0.1: 2,
    0.2: 3,
    0.3: 19,
    0.4: 25,
    0.5: 31,
    0.6: 37,
    0.7: 41,
    0.8: 46,
    0.9: 51,
    1.0: 56,
}

HOUSEHOLD_Number = 56

DEFAULT_OPTION_LABELS = {
    1: "Always plugged in",
    2: "Arrival-SoC-triggered plugging with flexible plugging window",
    3: "Arrival-SoC-triggered plugging with fixed plugging window",
}

SCENARIO_RE = re.compile(r"hh_(\d+(?:p\d+)?)__hp_(\d+(?:p\d+)?)__ev_[^/]+")
EV_RE = re.compile(r"/data_[^/]*_(\d+)/")
RATE_RE = re.compile(r"/greedy_results_nctrl(\d+(?:\.\d+)?)/")


def encoded_number(text: str) -> float:
    """Encoded number.
    
    Parameters
    ----------
    text : str
        Text used by this operation.
    
    Returns
    -------
    float
        Result computed as ``float(text.replace('p', '.'))``."""
    return float(text.replace("p", "."))


def metadata_from_path(name: str) -> dict | None:
    """Metadata from path.
    
    Parameters
    ----------
    name : str
        Name used by this operation.
    
    Returns
    -------
    dict | None
        Computed result for the selected input and execution branch."""
    normalized = "/" + name.replace("\\", "/").lstrip("/")
    scenario = SCENARIO_RE.search(normalized)

    ev_matches = EV_RE.findall(normalized)

    rate_match = RATE_RE.search(normalized)

    if not (scenario and ev_matches and rate_match):
        return None

    return {
        "household_factor": encoded_number(scenario.group(1)),
        "hp_ratio": encoded_number(scenario.group(2)),
        "ev_count": int(ev_matches[-1]),
        "participation_rate": float(rate_match.group(1)),
    }


def read_summary(source, source_name: str) -> pd.DataFrame:
    """Read summary.
    
    Parameters
    ----------
    source : object
        Source used by this operation.
    source_name : str
        Source name used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``frame``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    try:
        frame = pd.read_csv(source, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read {source_name}: {exc}") from exc
    required = {"option", "optimized_peak_MW", "uncontrolled_peak_MW"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_name}: missing columns {sorted(missing)}")
    return frame


def iter_zip(path: Path):
    """Iter zip.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    
    Yields
    -------
    object
        Result computed as ``(metadata, read_summary(content, f'{path}!{name}'))``."""
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if PurePosixPath(name).name != SUMMARY_NAME:
                continue
            metadata = metadata_from_path(name)
            if metadata is None:
                continue
            with archive.open(name) as handle:
                content = io.BytesIO(handle.read())
            yield metadata, read_summary(content, f"{path}!{name}")


def iter_folder(path: Path):
    """Iter folder.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    
    Yields
    -------
    object
        Result computed as ``(metadata, read_summary(csv_path, str(csv_path)))``."""
    for csv_path in path.rglob(SUMMARY_NAME):
        metadata = metadata_from_path(csv_path.as_posix())
        if metadata is not None:
            yield metadata, read_summary(csv_path, str(csv_path))


def collect_results(input_dir: Path) -> pd.DataFrame:
    """Collect results.
    
    Parameters
    ----------
    input_dir : Path
        Input dir used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``pd.DataFrame(rows).drop_duplicates()``.
    
    Raises
    ------
    FileNotFoundError
        If an explicit validation or processing check in this operation fails.
    ValueError
        If an explicit validation or processing check in this operation fails."""
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input does not exist: {input_dir}")
    if input_dir.is_file() and input_dir.suffix.lower() == ".zip":
        sources = [input_dir]
    elif input_dir.is_dir():
        sources = [input_dir, *sorted(input_dir.rglob("*.zip"))]
    else:
        raise ValueError("--input-dir must be a folder or ZIP file")

    rows: list[dict] = []
    for path in sources:
        iterator = iter_zip(path) if path.is_file() else iter_folder(path)
        for metadata, frame in iterator:
            for _, record in frame.iterrows():
                rows.append(
                    {
                        **metadata,
                        "option": pd.to_numeric(record["option"], errors="coerce"),
                        "optimized_peak_MW": pd.to_numeric(
                            record["optimized_peak_MW"], errors="coerce"
                        ),
                        "uncontrolled_peak_MW": pd.to_numeric(
                            record["uncontrolled_peak_MW"], errors="coerce"
                        ),
                    }
                )
    if not rows:
        raise ValueError(f"No usable {SUMMARY_NAME} files found")
    return pd.DataFrame(rows).drop_duplicates()


def parse_hp_map(specification: str | None) -> dict[float, int]:
    """Parse hp map.
    
    Parameters
    ----------
    specification : str | None
        Specification used by this operation.
    
    Returns
    -------
    dict[float, int]
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    argparse.ArgumentTypeError
        If an explicit validation or processing check in this operation fails."""
    if specification is None:
        return DEFAULT_HP_COUNT_MAP.copy()
    mapping: dict[float, int] = {}
    try:
        for pair in specification.split(","):
            ratio, count = pair.split(":", maxsplit=1)
            mapping[float(ratio)] = int(count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--hp-count-map must look like '0:0,0.1:8,...,1:56'"
        ) from exc
    return mapping


def hp_count(ratio: float, mapping: dict[float, int]) -> int:
    """Hp count.
    
    Parameters
    ----------
    ratio : float
        Ratio used by this operation.
    mapping : dict[float, int]
        Mapping used by this operation.
    
    Returns
    -------
    int
        Result computed as ``value``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    for key, value in mapping.items():
        if np.isclose(ratio, key, atol=1e-9):
            return value
    raise ValueError(f"No heat-pump count defined for hp_ratio={ratio:g}")


def prepare_heatmaps(
    results: pd.DataFrame,
    options: list[int],
    ev_count: int,
    hp_map: dict[float, int],
    peak_limit: float | None,
    baseline_hp_count: int,
) -> tuple[dict[int, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, float, float]:
    """Prepare heatmaps.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results used by this operation.
    options : list[int]
        Options used by this operation.
    ev_count : int
        Ev count used by this operation.
    hp_map : dict[float, int]
        Hp map used by this operation.
    peak_limit : float | None
        Peak limit used by this operation.
    baseline_hp_count : int
        Baseline hp count used by this operation.
    
    Returns
    -------
    tuple[dict[int, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, float, float]
        Result computed as ``(outputs, pd.concat(all_tidy, ignore_index=True), peak_limit, household_factor)``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    ev_results = results[results["ev_count"] == ev_count].copy()
    if ev_results.empty:
        raise ValueError(f"No results for EV count={ev_count}")
    ev_results["hp_count"] = [
        hp_count(value, hp_map) for value in ev_results["hp_ratio"]
    ]

    # Uncontrolled values are shared by the options. They provide the 0% row
    # and replace missing/failed optimization results without creating grey cells.
    uncontrolled = (
        ev_results.dropna(subset=["uncontrolled_peak_MW"])
        .groupby(["participation_rate", "hp_count"], as_index=False)
        .agg(uncontrolled_peak_MW=("uncontrolled_peak_MW", "median"))
    )
    uncontrolled_matrix = uncontrolled.pivot(
        index="participation_rate", columns="hp_count", values="uncontrolled_peak_MW"
    ).sort_index().sort_index(axis=1)
    hp_columns = list(uncontrolled_matrix.columns)
    rates = list(uncontrolled_matrix.index)
    uncontrolled_by_hp = uncontrolled_matrix.median(axis=0)

    if baseline_hp_count not in uncontrolled_by_hp.index:
        raise ValueError(f"No uncontrolled baseline for HP count {baseline_hp_count}")
    if peak_limit is None:
        peak_limit = float(uncontrolled_by_hp.loc[baseline_hp_count])

    household_factor = float(ev_results["household_factor"].median())
    outputs: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_tidy: list[pd.DataFrame] = []

    for option in options:
        selected = ev_results[
            (ev_results["option"] == option)
            & ev_results["optimized_peak_MW"].notna()
        ]
        if selected.empty:
            raise ValueError(f"No successful optimized values for option={option}")
        optimized = (
            selected.groupby(["participation_rate", "hp_count"], as_index=False)
            .agg(optimized_peak_MW=("optimized_peak_MW", "mean"))
            .pivot(
                index="participation_rate",
                columns="hp_count",
                values="optimized_peak_MW",
            )
            .reindex(index=rates, columns=hp_columns)
        )
        fallback_mask = optimized.isna()
        filled = optimized.copy()
        for rate in rates:
            rate_uncontrolled = uncontrolled_matrix.loc[rate].fillna(uncontrolled_by_hp)
            filled.loc[rate] = filled.loc[rate].fillna(rate_uncontrolled)

        # Insert 0% participation as an explicit uncontrolled reference row.
        zero_row = pd.DataFrame(
            [uncontrolled_by_hp.to_numpy()], index=[0.0], columns=hp_columns
        )
        matrix = pd.concat([zero_row, filled]).sort_index()
        zero_mask = pd.DataFrame(False, index=[0.0], columns=hp_columns)
        mask = pd.concat([zero_mask, fallback_mask]).sort_index()
        matrix.index.name = "participation_rate"
        matrix.columns.name = "hp_count"
        mask.index.name = "participation_rate"
        mask.columns.name = "hp_count"

        tidy = matrix.stack().rename("displayed_peak_MW").reset_index()
        tidy["option"] = option
        tidy["participation_rate_percent"] = (
            tidy["participation_rate"] * 100
        ).round().astype(int)
        tidy["value_source"] = "optimized"
        tidy.loc[tidy["participation_rate"] == 0, "value_source"] = "0% uncontrolled baseline"
        mask_long = mask.stack().rename("is_uncontrolled_fallback").reset_index()
        tidy = tidy.merge(mask_long, on=["participation_rate", "hp_count"], how="left")
        tidy.loc[tidy["is_uncontrolled_fallback"], "value_source"] = (
            "uncontrolled fallback (optimization unavailable)"
        )
        tidy["within_reference_limit"] = tidy["displayed_peak_MW"] <= peak_limit
        tidy["reference_peak_MW"] = peak_limit
        outputs[option] = (matrix, mask)
        all_tidy.append(tidy)

    return outputs, pd.concat(all_tidy, ignore_index=True), peak_limit, household_factor


def shared_color_limits(matrices: dict[int, tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[float, float]:
    """Shared color limits.
    
    Parameters
    ----------
    matrices : dict[int, tuple[pd.DataFrame, pd.DataFrame]]
        Matrices used by this operation.
    
    Returns
    -------
    tuple[float, float]
        Result computed as ``(float(np.nanmin(values)), float(np.nanmax(values)))``."""
    values = np.concatenate([matrix.to_numpy().ravel() for matrix, _ in matrices.values()])
    return float(np.nanmin(values)), float(np.nanmax(values))


def draw_heatmap(
    matrix: pd.DataFrame,
    peak_limit: float,
    vmin: float,
    vmax: float,
    household_factor: float,
    output_dir: Path,
    output_stem: str,
    ev_count: int,
    option: int,
    option_label: str,
    dpi: int,
) -> tuple[Path, Path]:
    """Draw heatmap.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix used by this operation.
    peak_limit : float
        Peak limit used by this operation.
    vmin : float
        Vmin used by this operation.
    vmax : float
        Vmax used by this operation.
    household_factor : float
        Household factor used by this operation.
    output_dir : Path
        Directory for generated outputs.
    output_stem : str
        Output stem used by this operation.
    ev_count : int
        Ev count used by this operation.
    option : int
        Option used by this operation.
    option_label : str
        Option label used by this operation.
    dpi : int
        Dpi used by this operation.
    
    Returns
    -------
    tuple[Path, Path]
        Result computed as ``(png_path, pdf_path)``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    values = matrix.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=peak_limit, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn_r")

    width = max(8.4, 0.72 * len(matrix.columns) + 2.3)
    height = max(6.1, 0.50 * len(matrix.index) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if luminance > 0.57 else "white"
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=text_color,
                fontweight="semibold" if value <= peak_limit else "normal",
            )
            within_limit = (
            value <= peak_limit
            or np.isclose(value, peak_limit, rtol=0, atol=1e-6)
            )
            if within_limit:
                ax.add_patch(
                    Rectangle(
                        (col - 0.49, row - 0.49), 0.98, 0.98,
                        fill=False, edgecolor="black", linewidth=1.9,
                    )
                )
    # Emphasize the common standard: 0% participation and 0 HP.
    baseline_col = list(matrix.columns).index(0)
    baseline_row = list(matrix.index).index(0.0)
    ax.add_patch(
        Rectangle(
            (baseline_col - 0.47, baseline_row - 0.47), 0.94, 0.94,
            fill=False, edgecolor="#0057B8", linewidth=3.0,
        )
    )

    ax.set_xticks(range(len(matrix.columns)), [str(x) for x in matrix.columns])
    ax.set_yticks(
        range(len(matrix.index)),
        [f"{int(round(rate * 100))}%" for rate in matrix.index],
    )
    ax.set_xlabel("Number of heat-pump buildings", fontsize=11)
    ax.set_ylabel("V2G participation rate in %", fontsize=11)
    ax.set_title(
        f"{option_label}",
        fontsize=14,
        fontweight="bold",
        pad=32,
    )
    ax.text(
        0.5,
        1.045,
        f"{ev_count} EVs, Number of buildings: {HOUSEHOLD_Number}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.046)
    colorbar.set_label("Optimized peak in MW", fontsize=10)
    legend_items = [
        Patch(
            facecolor="none", edgecolor="#0057B8", linewidth=3,
            label=f"Reference: 0% participation, 0 HP ({peak_limit:.3f} MW)",
        ),
        Patch(
            facecolor="none", edgecolor="black", linewidth=2,
            label=f"At or below reference peak (≤ {peak_limit:.3f} MW)",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=1,
        frameon=False,
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.11, right=0.91, top=0.82, bottom=0.27)

    option_stem = f"{output_stem}_option_{option}"
    png_path = output_dir / f"{option_stem}.png"
    pdf_path = output_dir / f"{option_stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_option_labels(specifications: list[str]) -> dict[int, str]:
    """Parse option labels.
    
    Parameters
    ----------
    specifications : list[str]
        Specifications used by this operation.
    
    Returns
    -------
    dict[int, str]
        Result computed as ``labels``.
    
    Raises
    ------
    argparse.ArgumentTypeError
        If an explicit validation or processing check in this operation fails."""
    labels = DEFAULT_OPTION_LABELS.copy()
    for specification in specifications:
        try:
            option, label = specification.split(":", maxsplit=1)
            labels[int(option)] = label.strip()
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--option-label must look like '2:Behavioral plugging'"
            ) from exc
    return labels


def make_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Result computed as ``parser``."""
    parser = argparse.ArgumentParser(
        description="Heatmaps of participation rate, HP count, and feeder peak."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("participation_hp_heatmap"))
    parser.add_argument("--output-stem", default="participation_hp_peak_heatmap")
    parser.add_argument("--options", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--option-label", action="append", default=[],
        help="Optional override such as '2:Behavioral plugging'. May be repeated.",
    )
    parser.add_argument("--ev-count", type=int, default=73)
    parser.add_argument("--peak-limit", type=float, default=None)
    parser.add_argument("--baseline-hp-count", type=int, default=0)
    parser.add_argument("--hp-count-map", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    args = make_parser().parse_args()
    results = collect_results(args.input_dir)
    print("Detected EV counts:", sorted(results["ev_count"].unique()))
    heatmaps, tidy, peak_limit, household_factor = prepare_heatmaps(
        results=results,
        options=args.options,
        ev_count=args.ev_count,
        hp_map=parse_hp_map(args.hp_count_map),
        peak_limit=args.peak_limit,
        baseline_hp_count=args.baseline_hp_count,
    )
    labels = parse_option_labels(args.option_label)
    vmin, vmax = shared_color_limits(heatmaps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for option in args.options:
        matrix, _fallback_mask = heatmaps[option]
        paths.extend(
            draw_heatmap(
                matrix=matrix,
                peak_limit=peak_limit,
                vmin=vmin,
                vmax=vmax,
                household_factor=household_factor,
                output_dir=args.output_dir,
                output_stem=args.output_stem,
                ev_count=args.ev_count,
                option=option,
                option_label=labels.get(option, f"Option {option}"),
                dpi=args.dpi,
            )
        )
    csv_path = args.output_dir / f"{args.output_stem}_all_options_data.csv"
    tidy.to_csv(csv_path, index=False)
    paths.append(csv_path)
    print(f"Reference peak at 0% participation and 0 HP: {peak_limit:.9f} MW")
    print(f"Created {len(args.options)} heatmaps with options {args.options}")
    for path in paths:
        print(f"Created: {path.resolve()}")


if __name__ == "__main__":
    main()