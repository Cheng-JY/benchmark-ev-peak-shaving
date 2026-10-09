#!/usr/bin/env python3
r"""Analyse how the heat-pump count affects the optimized feeder peak.

The script accepts ZIP archives, extracted result folders, or a mixture of
both.  It reads ``greedy_all_options_summary.csv`` files directly and creates:

* one colour for each selected EV fleet size,
* solid lines for optimized peaks,
* dashed lines for uncontrolled peaks, and
* a tidy CSV containing the values used in the plot.

Example (Windows cmd.exe):

    python ecv2g\user_dirs\jcheng\scripts\plot_hp_count_optimized_peak.py ^
      --input-dir "data\data_20260907_1\scenario_participation_results" ^
      --output-dir data\data_20260907_1\scenario_participation_results\hp_count_analysis ^
      --option 2 ^
      --participation-rate 1.0 ^
      --ev-counts 40,50,60,70,73

        python ecv2g\user_dirs\jcheng\scripts\plot_participation_hp_heatmap.py ^
      --input-dir "data\data_20260910_1\scenarios" ^
      --output-dir "data\data_20260910_1\scenarios\participation_hp_heatmap"^
      --ev-count 607

    python ecv2g\user_dirs\jcheng\scripts\plot_hp_count_optimized_peak.py ^
        --input-dir "data\data_20260910_1\scenarios" ^
        --output-dir data\data_20260910_1\scenarios\hp_count_analysis ^
        --option 2 ^
        --participation-rate 1.0 ^
        --ev-counts 300, 400, 500, 600
    
    python ecv2g\user_dirs\jcheng\scripts\plot_hp_count_optimized_peak.py ^
            --input-dir "data\data_20260928_1\scenarios" ^
            --output-dir data\data_20260928_1\scenarios\hp_count_analysis ^
            --option 2 ^
            --participation-rate 1.0 ^
            --ev-counts 50, 60, 70, 80, 90
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SUMMARY_NAME = "greedy_all_options_summary.csv"

# Actual heat-pump-household counts for the Kassel hh_factor=1 scenarios.
# Override this on the command line if another grid/scenario is analysed.
"""
DEFAULT_HP_COUNT_MAP = {
    0.0: 0,
    0.1: 8,
    0.2: 14,
    0.3: 19,
    0.4: 25,
    0.5: 31,
    0.6: 37,
    0.7: 41,
    0.8: 46,
    0.9: 51,
    1.0: 56,
}
"""
DEFAULT_HP_COUNT_MAP = {
    0.0: 0,
    0.1: 1,
    0.2: 2,
    0.3: 3,
    0.4: 4,
    0.5: 5,
    0.6: 6,
    0.7: 7,
    0.8: 8,
    0.9: 10,
    1.0: 12,
}

SCENARIO_RE = re.compile(
    r"hh_(?P<hh>\d+(?:p\d+)?)__hp_(?P<hp>\d+(?:p\d+)?)__ev_[^/]+"
)
# Accept both extracted data directories (``..._30/``) and ZIP archive names
# (``..._30.zip/``).  Some result batches store the EV count only in the
# archive name, not in the paths of the archive members.
EV_COUNT_RE = re.compile(
    r"(?<=/)data_[^/]*_(?P<ev>\d+)(?:\.zip)?(?=/)"
)
PARTICIPATION_RE = re.compile(r"/greedy_results_nctrl(?P<rate>\d+(?:\.\d+)?)/")
FLEET_FOLDER_RE = re.compile(r"(?:^|_)(?P<ev>\d+)(?:_?evs?)?$", re.IGNORECASE)


def encoded_number(text: str) -> float:
    """Convert scenario notation such as ``0p4`` to 0.4."""
    return float(text.replace("p", "."))


def parse_metadata(
    member_name: str, ev_count_hint: int | None = None
) -> dict[str, float | int] | None:
    """Read household factor, HP ratio, EV count, and participation rate."""
    normalized = "/" + member_name.replace("\\", "/").lstrip("/")
    scenario = SCENARIO_RE.search(normalized)
    # A path can contain more than one ``data_*_<number>`` directory, e.g.
    # ``data_20260907_1/.../data_20260903_30/...``.  The outer directory's
    # trailing ``1`` is a batch identifier, whereas the innermost match is the
    # actual EV fleet size.  Using search() incorrectly classified such runs as
    # one-EV scenarios.
    ev_matches = list(EV_COUNT_RE.finditer(normalized))
    ev_match = ev_matches[-1] if ev_matches else None
    rate_match = PARTICIPATION_RE.search(normalized)
    if not (scenario and rate_match and (ev_match or ev_count_hint is not None)):
        return None
    return {
        "household_factor": encoded_number(scenario.group("hh")),
        "hp_ratio": encoded_number(scenario.group("hp")),
        "ev_count": (
            ev_count_hint if ev_count_hint is not None else int(ev_match.group("ev"))
        ),
        "participation_rate": float(rate_match.group("rate")),
    }


def ev_count_from_fleet_subfolder(root: Path, csv_path: Path) -> int | None:
    """Read the fleet size from the second folder below ``--input-dir``.

    The expected hierarchy is ``root / HP-scenario / EV-fleet / ...``.  The
    first level contains names such as ``hh_1__hp_0p5__ev_1_...``; the second
    level contains the fleet size.  Supported fleet-folder examples include
    ``10``, ``ev_10``, ``fleet_10``, and ``data_20260903_10``.
    """
    try:
        relative_parts = csv_path.relative_to(root).parts
    except ValueError:
        return None
    # Two directory levels plus the CSV filename are required.
    if len(relative_parts) < 3:
        return None
    match = FLEET_FOLDER_RE.search(relative_parts[1])
    return int(match.group("ev")) if match else None


def read_summary(stream: io.BytesIO | io.StringIO, source: str) -> pd.DataFrame:
    """Read a summary CSV, accepting semicolon- or comma-separated files."""
    try:
        frame = pd.read_csv(stream, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read {source}: {exc}") from exc

    required = {"option", "uncontrolled_peak_MW", "optimized_peak_MW"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    return frame


def summaries_from_zip(path: Path):
    """Summaries from zip.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    
    Yields
    -------
    object
        Result computed as ``(metadata, read_summary(data, f'{path}!{name}'), f'{path}!{name}')``."""
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if PurePosixPath(name).name != SUMMARY_NAME:
                continue
            # Include the archive path because older batches sometimes encode
            # the EV fleet size only in the ZIP filename.  parse_metadata()
            # selects the innermost/last data_*_<EV> match, so a more specific
            # member directory still takes precedence when present.
            metadata = parse_metadata(f"{path.as_posix()}/{name}")
            if metadata is None:
                continue
            with archive.open(name) as raw:
                data = io.BytesIO(raw.read())
            yield metadata, read_summary(data, f"{path}!{name}"), f"{path}!{name}"


def summaries_from_directory(path: Path):
    """Summaries from directory.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    
    Yields
    -------
    object
        Result computed as ``(metadata, read_summary(csv_path, str(csv_path)), str(csv_path))``."""
    for csv_path in path.rglob(SUMMARY_NAME):
        folder_ev_count = ev_count_from_fleet_subfolder(path, csv_path)
        metadata = parse_metadata(csv_path.as_posix(), ev_count_hint=folder_ev_count)
        if metadata is None:
            continue
        yield metadata, read_summary(csv_path, str(csv_path)), str(csv_path)


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    """Expand input directories while avoiding duplicate ZIP discovery."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        item = item.resolve()
        if not item.exists():
            raise FileNotFoundError(f"Input does not exist: {item}")
        candidates = [item]
        if item.is_dir():
            # The directory itself is scanned for extracted summaries. ZIPs
            # below it are handled separately.
            candidates += sorted(item.rglob("*.zip"))
        for candidate in candidates:
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(candidate)
    return resolved


def collect_results(inputs: list[Path]) -> pd.DataFrame:
    """Collect results.
    
    Parameters
    ----------
    inputs : list[Path]
        Inputs used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``result``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    rows: list[dict] = []
    for source_path in resolve_inputs(inputs):
        if source_path.is_file() and source_path.suffix.lower() == ".zip":
            iterator = summaries_from_zip(source_path)
        elif source_path.is_dir():
            iterator = summaries_from_directory(source_path)
        elif source_path.name == SUMMARY_NAME:
            metadata = parse_metadata(source_path.as_posix())
            iterator = [] if metadata is None else [
                (metadata, read_summary(source_path, str(source_path)), str(source_path))
            ]
        else:
            continue

        for metadata, frame, source in iterator:
            for _, record in frame.iterrows():
                row = dict(metadata)
                row.update(
                    option=pd.to_numeric(record["option"], errors="coerce"),
                    optimized_peak_MW=pd.to_numeric(
                        record["optimized_peak_MW"], errors="coerce"
                    ),
                    uncontrolled_peak_MW=pd.to_numeric(
                        record["uncontrolled_peak_MW"], errors="coerce"
                    ),
                    run_status=str(record.get("run_status", "")),
                    source=source,
                )
                rows.append(row)

    if not rows:
        raise ValueError(
            f"No readable {SUMMARY_NAME} files with recognizable scenario paths found."
        )
    result = pd.DataFrame(rows)
    result = result.drop_duplicates(
        subset=[
            "household_factor",
            "hp_ratio",
            "ev_count",
            "participation_rate",
            "option",
            "optimized_peak_MW",
            "uncontrolled_peak_MW",
        ]
    )
    return result


def parse_hp_count_map(specification: str | None) -> dict[float, int]:
    """Parse hp count map.
    
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
            mapping[float(ratio.strip())] = int(count.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--hp-count-map must look like '0:0,0.1:8,...,1.0:56'"
        ) from exc
    return mapping


def lookup_hp_count(ratio: float, mapping: dict[float, int]) -> int:
    """Lookup hp count.
    
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
    raise ValueError(
        f"No heat-pump count is defined for hp_ratio={ratio:g}. "
        "Add it with --hp-count-map."
    )


def build_plot_table(
    results: pd.DataFrame,
    option: int,
    participation_rate: float,
    hp_count_map: dict[float, int],
    ev_counts: list[int] | None,
) -> pd.DataFrame:
    """Build plot table.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results used by this operation.
    option : int
        Option used by this operation.
    participation_rate : float
        Participation rate used by this operation.
    hp_count_map : dict[float, int]
        Hp count map used by this operation.
    ev_counts : list[int] | None
        Ev counts used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``grouped.sort_values(['ev_count', 'heat_pump_count']).reset_index(drop=True)``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    selected = results[
        (results["option"] == option)
        & np.isclose(results["participation_rate"], participation_rate, atol=1e-9)
    ].copy()
    selected = selected.dropna(subset=["optimized_peak_MW"])
    if selected.empty:
        available_options = sorted(results["option"].dropna().astype(int).unique())
        available_rates = sorted(results["participation_rate"].unique())
        raise ValueError(
            f"No successful values for option={option}, participation_rate="
            f"{participation_rate:g}. Available options: {available_options}; "
            f"available rates: {available_rates}."
        )

    optimized = (
        selected.groupby(["hp_ratio", "ev_count"], as_index=False)
        .agg(
            optimized_peak_MW=("optimized_peak_MW", "mean"),
            same_fleet_uncontrolled_peak_MW=("uncontrolled_peak_MW", "mean"),
            household_factor=("household_factor", "first"),
        )
    )

    # The uncontrolled curve must come from the 0% participation run of the
    # same HP scenario and the same EV fleet.  Do not reuse the
    # uncontrolled_peak_MW stored in the selected (e.g. 100%) run, because
    # older summaries may contain a common 73-EV reference there.
    baseline = results[
        (results["option"] == option)
        & np.isclose(results["participation_rate"], 0.0, atol=1e-9)
    ].copy()
    # At participation_rate = 0, use the explicitly stored uncontrolled peak
    # for this exact HP-factor/EV-fleet combination.  Do not use
    # optimized_peak_MW here: depending on the optimizer workflow, that value
    # may represent a different fallback or omit part of the uncontrolled EV
    # charging load.
    baseline = baseline.dropna(subset=["uncontrolled_peak_MW"])
    baseline = (
        baseline.groupby(["hp_ratio", "ev_count"], as_index=False)
        .agg(uncontrolled_peak_MW=("uncontrolled_peak_MW", "mean"))
    )

    grouped = optimized.merge(
        baseline,
        on=["hp_ratio", "ev_count"],
        how="left",
        validate="one_to_one",
    )
    has_zero_percent = grouped["uncontrolled_peak_MW"].notna()
    grouped["uncontrolled_source"] = np.where(
        has_zero_percent,
        "same HP factor + same EV fleet + 0% participation",
        "same HP factor + same EV fleet summary",
    )
    grouped["uncontrolled_peak_MW"] = grouped["uncontrolled_peak_MW"].fillna(
        grouped["same_fleet_uncontrolled_peak_MW"]
    )
    grouped = grouped.drop(columns=["same_fleet_uncontrolled_peak_MW"])

    missing_uncontrolled = grouped[grouped["uncontrolled_peak_MW"].isna()]
    if not missing_uncontrolled.empty:
        missing_pairs = ", ".join(
            f"HP ratio {row.hp_ratio:g} / {int(row.ev_count)} EVs"
            for row in missing_uncontrolled.itertuples()
        )
        raise ValueError(
            "No uncontrolled_peak_MW was found for: "
            + missing_pairs
        )

    available_ev_counts = sorted(grouped["ev_count"].astype(int).unique())
    if ev_counts is not None:
        missing = sorted(set(ev_counts).difference(available_ev_counts))
        if missing:
            raise ValueError(
                "Requested EV count(s) not found: "
                + ", ".join(map(str, missing))
                + ". Available EV counts: "
                + ", ".join(map(str, available_ev_counts))
            )
        grouped = grouped[grouped["ev_count"].isin(ev_counts)].copy()

    grouped["heat_pump_count"] = [
        lookup_hp_count(value, hp_count_map) for value in grouped["hp_ratio"]
    ]
    return grouped.sort_values(["ev_count", "heat_pump_count"]).reset_index(drop=True)


def draw_plot(
    table: pd.DataFrame,
    output_dir: Path,
    output_stem: str,
    title: str,
    dpi: int,
) -> tuple[Path, Path, Path]:
    """Draw plot.
    
    Parameters
    ----------
    table : pd.DataFrame
        Table used by this operation.
    output_dir : Path
        Directory for generated outputs.
    output_stem : str
        Output stem used by this operation.
    title : str
        Title used by this operation.
    dpi : int
        Dpi used by this operation.
    
    Returns
    -------
    tuple[Path, Path, Path]
        Result computed as ``(png_path, pdf_path, csv_path)``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ev_counts = sorted(table["ev_count"].astype(int).unique())
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.90, len(ev_counts)))

    for colour, ev_count in zip(colours, ev_counts):
        fleet = table[table["ev_count"] == ev_count].sort_values("heat_pump_count")
        x = fleet["heat_pump_count"].to_numpy(dtype=float)
        ax.plot(
            x,
            fleet["optimized_peak_MW"].to_numpy(dtype=float),
            color=colour,
            linestyle="-",
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            zorder=3,
        )
        ax.plot(
            x,
            fleet["uncontrolled_peak_MW"].to_numpy(dtype=float),
            color=colour,
            linestyle="--",
            marker=None,
            linewidth=1.8,
            zorder=2,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    ax.set_xlabel("Heat-pump buildings", fontsize=12)
    ax.set_ylabel("Feeder peak load in MW", fontsize=12)
    ax.set_xticks(sorted(table["heat_pump_count"].unique()))
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fleet_handles = [
        Line2D([0], [0], color=colour, linewidth=2.5, label=f"{ev_count} EVs")
        for colour, ev_count in zip(colours, ev_counts)
    ]
    style_handles = [
        Line2D([0], [0], color="#444444", linestyle="-", marker="o",
               linewidth=2.0, markersize=4.5, label="Optimized peak"),
        Line2D([0], [0], color="#444444", linestyle="--",
               linewidth=1.8, label="Uncontrolled peak"),
    ]
    fleet_legend = ax.legend(
        handles=fleet_handles,
        title="EV fleet size",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        title_fontsize=9.5,
    )
    ax.add_artist(fleet_legend)
    ax.legend(
        handles=style_handles,
        title="Operating case",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        frameon=False,
        fontsize=9,
        title_fontsize=9.5,
    )
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.11, right=0.78)

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    csv_path = output_dir / f"{output_stem}_data.csv"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(csv_path, index=False)
    return png_path, pdf_path, csv_path


def make_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Result computed as ``parser``."""
    parser = argparse.ArgumentParser(
        description="Plot the effect of heat-pump count on optimized peak."
    )
    parser.add_argument(
        "--input-dir",
        "--input",
        dest="input_dir",
        type=Path,
        required=True,
        help=(
            "One folder containing all result ZIP files. The folder is searched "
            "recursively; a single ZIP or extracted result folder is also accepted."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("hp_count_analysis"))
    parser.add_argument("--output-stem", default="hp_count_optimized_peak")
    parser.add_argument("--option", type=int, default=2)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument(
        "--ev-counts",
        nargs="+",
        default=None,
        help=(
            "EV fleet sizes separated by spaces or commas, e.g. "
            "'10 30 50 70 73' or '10,30,50,70,73'. "
            "The default plots all available fleets."
        ),
    )
    parser.add_argument(
        "--title", default="Effect of heat-pump count on optimized peak"
    )
    parser.add_argument(
        "--hp-count-map",
        default=None,
        help="Ratio-to-count mapping, e.g. '0:0,0.1:8,...,1:56'.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    args = make_parser().parse_args()
    hp_count_map = parse_hp_count_map(args.hp_count_map)
    ev_counts = (
        None
        if args.ev_counts is None
        else [
            int(value.strip())
            for token in args.ev_counts
            for value in token.split(",")
            if value.strip()
        ]
    )
    results = collect_results([args.input_dir])
    table = build_plot_table(
        results=results,
        option=args.option,
        participation_rate=args.participation_rate,
        hp_count_map=hp_count_map,
        ev_counts=ev_counts,
    )
    paths = draw_plot(
        table=table,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Read {len(results)} result rows.")
    print(f"HP/EV scenario rows: {len(table)}")
    print("EV fleet sizes: " + ", ".join(map(str, sorted(table["ev_count"].unique()))))
    for source, count in table["uncontrolled_source"].value_counts().items():
        print(f"Uncontrolled source: {source} ({count} rows)")
    for path in paths:
        print(f"Created: {path.resolve()}")


if __name__ == "__main__":
    main()