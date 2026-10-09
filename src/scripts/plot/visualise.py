"""Create an interactive Plotly HTML comparison of Greedy result files.

The script only reads existing result CSV files.  It does not run the Greedy
optimizer and does not import any ecv2g component, so it can also be used
offline without a database connection.

Example (Windows CMD, one line)::

    python ecv2g/user_dirs/jcheng/scripts/visualise.py --options-txt data/data_20260805_0.1/option1.txt --results-root data/data_20260805_0.1 --output-html data/20260805_0.1/greedy_comparison.html

    python ecv2g/user_dirs/jcheng/scripts/visualise.py --options-txt data/example/option1.txt --results-root data/example --output-html data/example/greedy_comparison.html

The initial x-axis window is the first week of February.  The complete time
series remains available through Plotly zooming, panning and the range slider.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence
import webbrowser

import numpy as np
import pandas as pd


OPTION_HEADER = re.compile(r"(?im)^\s*option\s*(\d+)\s*:\s*$")
SOURCE_DICT = re.compile(r"SOURCE_COLUMNS\s*=\s*(\{.*?\})", re.DOTALL)


@dataclass(frozen=True)
class OptionDefinition:
    """One option and its SOURCE_COLUMNS mapping."""

    name: str
    number: int
    source_columns: dict[str, str]

    @property
    def ch_avail_label(self) -> str:
        """Ch avail label.
        
        Returns
        -------
        str
            Computed result for the selected input and execution branch."""
        return _mapping_value(
            self.source_columns,
            preferred_keys=("ch_avail", "charging_power_available"),
            key_contains="ch_avail",
            fallback="ch_avail: unknown",
        )

    @property
    def soc_min_label(self) -> str:
        """Soc min label.
        
        Returns
        -------
        str
            Computed result for the selected input and execution branch."""
        return _mapping_value(
            self.source_columns,
            preferred_keys=("soc_min",),
            key_contains="soc_min",
            fallback="soc_min: unknown",
        )

    @property
    def legend_label(self) -> str:
        """Human-readable legend text without an option number."""

        return f"{self.ch_avail_label}<br>{self.soc_min_label}"


@dataclass(frozen=True)
class OptionResult:
    """Complete optimized and uncontrolled feeder-load series."""

    definition: OptionDefinition
    result_csv: Path
    time: pd.DatetimeIndex
    optimized_MW: np.ndarray
    uncontrolled_MW: np.ndarray

    @property
    def optimized_peak_MW(self) -> float:
        """Optimized peak mw.
        
        Returns
        -------
        float
            Result computed as ``float(np.max(self.optimized_MW))``."""
        return float(np.max(self.optimized_MW))

    @property
    def uncontrolled_peak_MW(self) -> float:
        """Uncontrolled peak mw.
        
        Returns
        -------
        float
            Result computed as ``float(np.max(self.uncontrolled_MW))``."""
        return float(np.max(self.uncontrolled_MW))

    @property
    def peak_reduction_percent(self) -> float:
        """Peak reduction percent.
        
        Returns
        -------
        float
            Computed result for the selected input and execution branch."""
        if self.uncontrolled_peak_MW == 0.0:
            return 0.0
        return (
            (self.uncontrolled_peak_MW - self.optimized_peak_MW)
            / self.uncontrolled_peak_MW
            * 100.0
        )


def _mapping_value(
    mapping: dict[str, str],
    *,
    preferred_keys: Sequence[str],
    key_contains: str,
    fallback: str,
) -> str:
    """Find a SOURCE_COLUMNS value while tolerating small key variations."""

    for key in preferred_keys:
        value = mapping.get(key)
        if value:
            return value
    for key, value in mapping.items():
        if key_contains in key.lower() and value:
            return value
    return fallback


def parse_options_txt(path: Path) -> list[OptionDefinition]:
    """Read all optionN blocks and their SOURCE_COLUMNS dictionaries."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Options TXT not found: {path}")

    text = path.read_text(encoding="utf-8-sig")
    headers = list(OPTION_HEADER.finditer(text))
    if not headers:
        raise ValueError(f"{path.name}: no 'optionN:' blocks were found.")

    definitions: list[OptionDefinition] = []
    seen: set[int] = set()
    for index, header in enumerate(headers):
        number = int(header.group(1))
        if number in seen:
            raise ValueError(f"{path.name}: option{number} is defined twice.")
        seen.add(number)

        block_end = (
            headers[index + 1].start() if index + 1 < len(headers) else len(text)
        )
        block = text[header.end():block_end]
        dictionary_match = SOURCE_DICT.search(block)
        if dictionary_match is None:
            raise ValueError(
                f"{path.name}: option{number} has no SOURCE_COLUMNS dictionary."
            )

        try:
            raw_mapping = ast.literal_eval(dictionary_match.group(1))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{path.name}: option{number} has an invalid SOURCE_COLUMNS "
                "dictionary."
            ) from exc
        if not isinstance(raw_mapping, dict):
            raise ValueError(
                f"{path.name}: option{number} SOURCE_COLUMNS must be a dictionary."
            )

        mapping: dict[str, str] = {}
        for key, value in raw_mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"{path.name}: option{number} SOURCE_COLUMNS must contain "
                    "only string keys and values."
                )
            mapping[key] = value

        definitions.append(
            OptionDefinition(
                name=f"option{number}",
                number=number,
                source_columns=mapping,
            )
        )

    return sorted(definitions, key=lambda item: item.number)


def normalize_option_name(value: str) -> str:
    """Normalize 1 and option1 to option1."""

    match = re.fullmatch(r"(?i)(?:option)?\s*(\d+)", value.strip())
    if match is None:
        raise ValueError(f"Invalid option selector: {value!r}")
    return f"option{int(match.group(1))}"


def select_definitions(
    definitions: Sequence[OptionDefinition],
    requested: Sequence[str] | None,
) -> list[OptionDefinition]:
    """Select result configurations made available in the HTML."""

    if not requested:
        return list(definitions)

    tokens = [
        token.strip()
        for argument in requested
        for token in argument.split(",")
        if token.strip()
    ]
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return list(definitions)

    by_name = {definition.name: definition for definition in definitions}
    selected: list[OptionDefinition] = []
    seen: set[str] = set()
    for token in tokens:
        name = normalize_option_name(token)
        if name not in by_name:
            raise ValueError(f"{name} is not defined in the options TXT.")
        if name not in seen:
            selected.append(by_name[name])
            seen.add(name)
    return selected


def resolve_result_csv(
    *,
    results_root: Path,
    directory_template: str,
    definition: OptionDefinition,
) -> Path | None:
    """Return the result CSV path if the configured result directory exists.

    The directory template may contain the placeholders:
    - {option}: for example, option1
    - {number}: for example, 1
    - {x}: alias for {option}

    If either the result directory or the expected CSV file does not exist,
    None is returned.
    """

    try:
        directory_name = directory_template.format(
            option=definition.name,
            number=definition.number,
            x=definition.name,
        )
    except KeyError as exc:
        raise ValueError(
            "--result-dir-template may only use "
            "{option}, {number}, or {x}."
        ) from exc

    result_directory = (
        results_root
        .expanduser()
        .resolve()
        / directory_name
    )

    # Ignore options without a matching result directory.
    if not result_directory.is_dir():
        return None

    result_csv = result_directory / "greedy_peak_shaving_results.csv"

    # Ignore directories that do not contain the expected result file.
    if not result_csv.is_file():
        return None

    return result_csv


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-separated result files."""

    frame = pd.read_csv(path, sep=None, engine="python")
    if len(frame.columns) == 1:
        separator = ";" if ";" in frame.columns[0] else ","
        frame = pd.read_csv(path, sep=separator)
    return frame


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Find column.
    
    Parameters
    ----------
    frame : pd.DataFrame
        Frame used by this operation.
    candidates : Sequence[str]
        Candidates used by this operation.
    
    Returns
    -------
    str
        Result computed as ``candidate``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Missing result column. Expected one of: " + ", ".join(candidates)
    )


def _power_from_result_column(
    frame: pd.DataFrame,
    *,
    mw_candidates: Sequence[str],
    mwh_candidates: Sequence[str],
    step_hours: np.ndarray,
) -> np.ndarray:
    """Return MW, converting MWh columns using the actual time-step length."""

    for column in mw_candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            if np.isnan(values).any():
                raise ValueError(f"Column {column!r} contains invalid values.")
            return values

    for column in mwh_candidates:
        if column in frame.columns:
            energy = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            if np.isnan(energy).any():
                raise ValueError(f"Column {column!r} contains invalid values.")
            return energy / step_hours

    raise ValueError(
        "Missing result column. Expected one of: "
        + ", ".join((*mw_candidates, *mwh_candidates))
    )


def _step_hours(time: pd.DatetimeIndex) -> np.ndarray:
    """Step hours.
    
    Parameters
    ----------
    time : pd.DatetimeIndex
        Time used by this operation.
    
    Returns
    -------
    np.ndarray
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    if len(time) < 2:
        return np.ones(len(time), dtype=float)
    differences = np.diff(time.asi8) / 3.6e12
    if np.any(differences <= 0):
        raise ValueError("Timestamps must be unique and sorted.")
    fallback = float(np.median(differences))
    return np.append(differences, fallback)


def read_result_csv(
    definition: OptionDefinition,
    result_csv: Path,
    start: str | None,
    end: str | None,
) -> OptionResult:
    """Read one saved result and calculate both feeder-load curves in MW."""

    if not result_csv.is_file():
        raise FileNotFoundError(f"{definition.name}: result not found: {result_csv}")

    frame = _read_csv_flexible(result_csv)
    if "time" not in frame.columns:
        raise ValueError(f"{result_csv}: missing column: time")

    time_text = frame["time"].astype("string").str.strip()

    time = pd.to_datetime(
        time_text,
        format="mixed",
        dayfirst=True,
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    if time.isna().any():
        bad_rows = list(frame.index[time.isna()][:5])
        raise ValueError(
            f"{result_csv}: 'time' contains invalid timestamps at rows {bad_rows}."
        )
    if time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError(f"{result_csv}: timestamps must be unique and sorted.")

    datetime_index = pd.DatetimeIndex(time)
    steps = _step_hours(datetime_index)
    base_column = _find_column(frame, ("base_load_MW", "household_load_MW"))
    base_load = pd.to_numeric(frame[base_column], errors="coerce").to_numpy(float)
    if np.isnan(base_load).any():
        raise ValueError(f"{result_csv}: {base_column!r} contains invalid values.")

    optimized_ev = _power_from_result_column(
        frame,
        mw_candidates=("ev_optimized_MW",),
        mwh_candidates=("ev_optimized_MWh",),
        step_hours=steps,
    )
    uncontrolled_ev = _power_from_result_column(
        frame,
        mw_candidates=("ev_uncontrolled_MW",),
        mwh_candidates=("ev_uncontrolled_MWh",),
        step_hours=steps,
    )

    mask = np.ones(len(frame), dtype=bool)
    if start is not None:
        mask &= datetime_index >= pd.to_datetime(start, errors="raise")
    if end is not None:
        mask &= datetime_index <= pd.to_datetime(end, errors="raise")
    if not bool(mask.any()):
        raise ValueError(
            f"{result_csv}: no rows remain inside the selected time range."
        )

    return OptionResult(
        definition=definition,
        result_csv=result_csv,
        time=datetime_index[mask],
        optimized_MW=(base_load + optimized_ev)[mask],
        uncontrolled_MW=(base_load + uncontrolled_ev)[mask],
    )


def extrema_preserving_indices(
    series: Sequence[np.ndarray],
    max_points: int,
) -> np.ndarray:
    """Reduce long plots while preserving endpoints and local extrema."""

    length = len(series[0])
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=int)
    if any(len(values) != length for values in series):
        raise ValueError("All series passed to downsampling must have equal length.")

    points_per_bin = 2 + 2 * len(series)
    number_of_bins = max(1, max_points // points_per_bin)
    edges = np.linspace(0, length, number_of_bins + 1, dtype=int)
    selected: set[int] = {0, length - 1}

    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        selected.add(left)
        selected.add(right - 1)
        for values in series:
            part = values[left:right]
            selected.add(left + int(np.argmin(part)))
            selected.add(left + int(np.argmax(part)))

    return np.fromiter(sorted(selected), dtype=int)


def _initial_time_window(
    results: Sequence[OptionResult],
    initial_start: str | None,
    initial_end: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return an explicit window or the first week of February in the data."""

    full_start = min(pd.Timestamp(result.time[0]) for result in results)
    full_end = max(pd.Timestamp(result.time[-1]) for result in results)

    if initial_start is not None:
        left = pd.Timestamp(initial_start)
    else:
        candidate_years = range(full_start.year, full_end.year + 1)
        february_starts = [
            pd.Timestamp(year=year, month=2, day=1)
            for year in candidate_years
            if full_start <= pd.Timestamp(year=year, month=2, day=1) <= full_end
        ]
        left = february_starts[0] if february_starts else full_start

    right = (
        pd.Timestamp(initial_end)
        if initial_end is not None
        else left + pd.Timedelta(days=7)
    )
    left = max(left, full_start)
    right = min(right, full_end)
    if right <= left:
        return full_start, full_end
    return left, right


def create_interactive_figure(
    results: Sequence[OptionResult],
    *,
    initially_visible: Sequence[str] | None,
    max_points: int,
    initial_start: str | None,
    initial_end: str | None,
):
    """Create a Plotly figure with grouped legends and a zoomable time axis."""

    if not results:
        raise ValueError("No option results are available to plot.")

    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except ModuleNotFoundError as exc:
        if exc.name == "plotly":
            raise ModuleNotFoundError(
                "Plotly is required to create the interactive HTML. Install it "
                "in the active environment with: conda install -c conda-forge "
                "plotly"
            ) from exc
        raise

    available = {result.definition.name for result in results}
    if initially_visible:
        active_names = {
            normalize_option_name(value)
            for argument in initially_visible
            for value in argument.split(",")
            if value.strip()
        }
        unknown = sorted(active_names - available)
        if unknown:
            raise ValueError(
                "Initial options have no loaded result: " + ", ".join(unknown)
            )
    else:
        active_names = set(available)

    figure = go.Figure()
    colours = qualitative.Plotly
    for index, result in enumerate(results):
        sample = extrema_preserving_indices(
            [result.optimized_MW, result.uncontrolled_MW],
            max_points=max_points,
        )
        colour = colours[index % len(colours)]
        visibility: bool | str = (
            True if result.definition.name in active_names else "legendonly"
        )
        group = f"configuration-{index}"
        hover_common = (
            "Time: %{x|%Y-%m-%d %H:%M}<br>"
            "Feeder load: %{y:.4f} MW"
            "<extra></extra>"
        )

        figure.add_trace(
            go.Scattergl(
                x=result.time[sample],
                y=result.optimized_MW[sample],
                mode="lines",
                name="Optimized",
                legendgroup=group,
                legendgrouptitle_text=result.definition.legend_label,
                line={"color": colour, "width": 2},
                visible=visibility,
                hovertemplate=hover_common,
            )
        )
        figure.add_trace(
            go.Scattergl(
                x=result.time[sample],
                y=result.uncontrolled_MW[sample],
                mode="lines",
                name="Uncontrolled",
                legendgroup=group,
                line={"color": colour, "width": 1.5, "dash": "dash"},
                visible=visibility,
                hovertemplate=hover_common,
            )
        )

    window_start, window_end = _initial_time_window(
        results,
        initial_start,
        initial_end,
    )
    figure.update_layout(
        title={
            "text": "Greedy Peak-Shaving Comparison",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=760,
        margin={"l": 75, "r": 340, "t": 85, "b": 105},
        hovermode="x unified",
        dragmode="zoom",
        legend={
            "title": {"text": "Charging availability and minimum SoC"},
            "groupclick": "togglegroup",
            "tracegroupgap": 12,
            "x": 1.01,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "font": {"size": 11},
        },
        xaxis={
            "title": "Time",
            "type": "date",
            "range": [window_start, window_end],
            "rangeslider": {"visible": True, "thickness": 0.09},
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1 day", "step": "day", "stepmode": "backward"},
                    {"count": 7, "label": "1 week", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1 month", "step": "month", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
                "x": 0.0,
                "y": 1.08,
            },
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
        },
        yaxis={
            "title": "Feeder load [MW]",
            "fixedrange": False,
            "zeroline": True,
            "zerolinecolor": "#777",
        },
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.15)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(120,120,120,0.15)")
    return figure


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Result computed as ``parser``."""
    parser = argparse.ArgumentParser(
        description=(
            "Read existing Greedy result CSV files and write an interactive "
            "Plotly HTML comparison."
        )
    )
    parser.add_argument(
        "--options-txt",
        type=Path,
        required=True,
        help="TXT file containing optionN SOURCE_COLUMNS blocks.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Parent directory containing the Greedy result directories.",
    )
    parser.add_argument(
        "--result-dir-template",
        default="greedy_results_{number}",
        help=(
            "Result-directory template relative to --results-root. Available "
            "placeholders: {option}, {number}, {x}. "
            "Default: greedy_results_{number}."
        ),
    )
    parser.add_argument(
        "--options",
        nargs="*",
        default=None,
        help=(
            "Configurations included in the HTML, for example "
            "--options option1 option3 option7. Default: all."
        ),
    )
    parser.add_argument(
        "--initial-options",
        nargs="*",
        default=None,
        help=(
            "Configurations initially visible. Others start as legend-only. "
            "Default: all loaded configurations."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional first timestamp/date to load.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional last timestamp/date to load.",
    )
    parser.add_argument(
        "--initial-start",
        default=None,
        help=(
            "Optional initial x-axis start. Default: February 1 in the data year."
        ),
    )
    parser.add_argument(
        "--initial-end",
        default=None,
        help=(
            "Optional initial x-axis end. Default: seven days after initial start."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help=(
            "Maximum displayed points per configuration after extrema-preserving "
            "downsampling. Use 0 for all points. Default: 12000."
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("greedy_peak_shaving_comparison.html"),
        help="Output HTML path. Default: greedy_peak_shaving_comparison.html.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated HTML in the default browser.",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help=(
            "Load Plotly from a CDN to create a smaller HTML. Without this flag, "
            "the HTML is self-contained and also works offline."
        ),
    )
    return parser


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails.
    SystemExit
        If an explicit validation or processing check in this operation fails."""
    args = build_parser().parse_args()
    if args.max_points < 0:
        raise ValueError("--max-points must be zero or positive.")

    definitions = parse_options_txt(args.options_txt)
    selected = select_definitions(definitions, args.options)
    results: list[OptionResult] = []

    for definition in selected:
        result_csv = resolve_result_csv(
            results_root=args.results_root,
            directory_template=args.result_dir_template,
            definition=definition,
        )

        if result_csv is None:
            print(
                f"Skipping {definition.name}: "
                "no matching result directory or result CSV was found."
            )
            continue

        result = read_result_csv(
            definition=definition,
            result_csv=result_csv,
            start=args.start,
            end=args.end,
        )

        results.append(result)

        print(
            f"[{definition.ch_avail_label} | "
            f"{definition.soc_min_label}] "
            f"{result.uncontrolled_peak_MW:.6f} -> "
            f"{result.optimized_peak_MW:.6f} MW "
            f"({result.peak_reduction_percent:.2f}%)"
        )

    if not results:
        raise SystemExit("No result files were loaded.")

    figure = create_interactive_figure(
        results,
        initially_visible=args.initial_options,
        max_points=args.max_points,
        initial_start=args.initial_start,
        initial_end=args.initial_end,
    )

    output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_html,
        include_plotlyjs="cdn" if args.cdn else True,
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "greedy_peak_shaving_comparison",
                "scale": 2,
            },
        },
    )
    print(f"Saved interactive HTML: {output_html}")

    if args.open_browser:
        webbrowser.open(output_html.as_uri())


if __name__ == "__main__":
    main()