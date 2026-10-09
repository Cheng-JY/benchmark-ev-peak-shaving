#!/usr/bin/env python3
#!/usr/bin/env python3
r"""
Collect and summarize greedy optimization results across ch_refuel scenarios.

Purpose
-------
This script combines successful and failed optimization runs into a common
table for subsequent analysis and plotting. Successful runs are recalculated
against a shared uncontrolled feeder peak selected from a ch_refuel=0.4
scenario.

The script reads existing results only; it does not run the optimizer.

Input directory structure
-------------------------
The data root must contain immediate subdirectories whose names end with a
numeric ch_refuel value, for example:

    data/data_20260923_2/
        greedy_results_0.1/
            greedy_all_options_summary.csv
            greedy_failed_options.csv
        greedy_results_0.2/
            greedy_all_options_summary.csv
        ...
        greedy_results_1.0/
            greedy_all_options_summary.csv

Other folder prefixes are accepted, provided that the final suffix is numeric.
The suffix is interpreted as ch_refuel, not as an EV count or participation
rate.

Each scenario directory must contain at least one of:

    greedy_all_options_summary.csv
        Summary records for successful optimization runs.

    greedy_failed_options.csv
        Records for failed optimization runs, including error information.

Input CSV files must use semicolons as delimiters. UTF-8 files with or without
a byte-order mark are supported. Only immediate subdirectories are searched;
nested result directories are not scanned recursively.

Processing
----------
1. Detect scenario directories and extract ch_refuel from their names.
2. Read successful and failed optimization records.
3. Classify charging availability and minimum-SoC variants.
4. Select a common uncontrolled peak from a successful ch_refuel=0.4 record.
5. Recalculate peak-reduction metrics for successful runs.
6. Write the combined table, comparison matrices, and summary statistics.

Availability classification
---------------------------
The following availability columns are recognized:

    ch_avail_cp
        Technical charging availability.

    ch_avail_refuel_cp
        Refuel-based charging availability.

    ch_avail_refuel_targetsoc_cp
        Refuel-based availability with a target-SoC condition.

    *_reduced_flat<factor>_cp
    *_reduced_peak<factor>_cp
        Reduced availability. The reduction mode and numeric factor are
        extracted from the column name.

Unrecognized availability columns are classified as "other". The script
records the factor encoded in the column name without converting between
a removal fraction and a retention fraction.

Minimum-SoC columns containing "_lim_" or ending in "_lim_cp" are classified
as "limited". Other non-empty minimum-SoC columns are classified as "standard".

Common uncontrolled baseline
----------------------------
The baseline is selected from successful ch_refuel=0.4 records:

1. Prefer the first record whose uncontrolled charging column is
   "ch_direct_cp". If uncontrolled_charging_column is empty, charging_column
   is checked instead.
2. If no matching record is found, use the first successful ch_refuel=0.4
   record with a parseable, non-zero uncontrolled_peak_MW.

If no suitable record exists, the script raises an error.

The same baseline is applied to every successful record. Therefore, each
collection should contain comparable scenarios with the same feeder,
background load, and EV fleet size.

For each successful run:

    peak_reduction_MW = baseline_peak_MW - optimized_peak_MW

    peak_reduction_percent =
        100 * peak_reduction_MW / baseline_peak_MW

    optimized_to_uncontrolled_peak_ratio =
        optimized_peak_MW / baseline_peak_MW

The ratio is a fraction, not a percentage. For example, 0.60 means that the
optimized peak is 60% of the common uncontrolled peak.

Negative peak reductions are retained.

Failed runs
-----------
Failed runs are retained with run_status="failed" and their available error
information. The script does not assign an optimized peak, zero reduction,
or a 100% relative peak to failed runs.

Failed runs have empty cells in the peak-reduction and peak-ratio matrices.
Their status is included in the status matrix and success-rate summary.

Command-line arguments
----------------------
--data-root PATH
    Directory containing the ch_refuel scenario subdirectories.
    Default: data

--output-dir PATH
    Directory for the generated CSV files.
    Default: <data-root>/greedy_collected_results

Relative paths are resolved from the current working directory.

Usage
-----
Windows Command Prompt, using the default output directory:

    python ecv2g\user_dirs\jcheng\scripts\collect_greedy_results.py ^
        --data-root "data\data_20260923_2"

Windows Command Prompt, specifying an output directory:

    python ecv2g\user_dirs\jcheng\scripts\collect_greedy_results.py ^
        --data-root "data\data_20260923_2" ^
        --output-dir "data\data_20260923_2\greedy_collected_results"

Linux or macOS:

    python ecv2g/user_dirs/jcheng/scripts/collect_greedy_results.py \
        --data-root data/data_20260923_2 \
        --output-dir data/data_20260923_2/greedy_collected_results

Output files
------------
greedy_results_tidy.csv
    Combined successful and failed records, including scenario parameters,
    availability classifications, recalculated metrics, and error information.

peak_reduction_percent_matrix.csv
    Peak reduction in percent, with options as rows and ch_refuel values
    as columns.

optimized_to_uncontrolled_peak_ratio_matrix.csv
    Optimized peak divided by the common uncontrolled peak, with options
    as rows and ch_refuel values as columns.

run_status_matrix.csv
    Success or failure status for each option and ch_refuel combination.

ch_refuel_summary.csv
    Number of successful and failed records, success rate, and mean,
    minimum, and maximum peak-reduction percentages for each ch_refuel value.
    Reduction statistics include successful records with numeric reductions.

All output CSV files use semicolon delimiters and UTF-8 encoding with a
byte-order mark. Existing output files with the same names are overwritten.

Limitations
-----------
- ch_refuel values are formatted to one decimal place.
- Comparison matrices assume one record per option and ch_refuel combination.
  If duplicates exist, the last collected record is used in the matrices;
  all records remain in the combined table.
- The common baseline is selected from one record. Consistency between
  candidate baseline records is not checked.
- In the current implementation, optimizer_reported_uncontrolled_peak_MW
  contains the selected common baseline for recalculated successful runs,
  rather than preserving each run's original uncontrolled peak.
- Summary means describe the collected option records and do not represent
  a population-weighted average.

Dependencies
------------
Python 3.10 or later. Only Python standard-library modules are required.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


SUCCESS_FILE = "greedy_all_options_summary.csv"
FAILED_FILE = "greedy_failed_options.csv"
UNCONTROLLED_BASELINE_CH_REFUEL = 0.4
UNCONTROLLED_CHARGING_COLUMN = "ch_refuel_cp"

BASE_OUTPUT_COLUMNS = [
    "ch_refuel",
    "scenario_folder",
    "option",
    "run_status",
    "availability_column",
    "availability_variant",
    "reduce_mode",
    "reduce_factor",
    "soc_max_column",
    "soc_min_column",
    "soc_min_variant",
    "charging_column",
    "uncontrolled_charging_column",
    "consumption_column",
    "ch_min_column",
    "dem_temp_column",
    "vehicle_count",
    "start_discharging_kWh_per_vehicle",
    "charging_efficiency",
    "discharging_efficiency",
    "time_steps",
    "uncontrolled_baseline_ch_refuel",
    "uncontrolled_peak_MW",
    "optimized_peak_MW",
    "optimized_to_uncontrolled_peak_ratio",
    "peak_threshold_MW",
    "peak_reduction_MW",
    "peak_reduction_percent",
    "all_steps_feasible",
    "infeasible_steps",
    "optimizer_reported_uncontrolled_peak_MW",
    "error_type",
    "error_message",
]

"""Extract ch_refuel from the final numeric suffix of a folder name.

Parameters
----------
folder_name : str
    Folder name ending in an underscore followed by an integer or decimal
    value, for example "greedy_results_0.4".

Returns
-------
float
    Parsed ch_refuel value.

Raises
------
ValueError
    If the folder name does not contain a valid numeric suffix.

Notes
-----
The parsed value is not checked against the expected range [0, 1].
"""

def parse_ch_refuel(folder_name: str) -> float:
    """Parse the final folder-name suffix as ch_refuel.

    Examples
    --------
    data_20260805_0.1 -> 0.1
    data_20260805_0.9 -> 0.9
    data_20260805_1   -> 1.0
    """
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", folder_name)
    if not match:
        raise ValueError(
            f"Cannot parse ch_refuel from folder name: {folder_name!r}"
        )
    return float(match.group(1))

"""Read a semicolon-delimited CSV file into a list of records.

Parameters
----------
path : Path
    Path to the input CSV file.

Returns
-------
list of dict
    Records mapping column names to string values. Returns an empty list
    if the path does not identify an existing file.

Notes
-----
UTF-8 encoding with or without a byte-order mark is supported.
"""


def read_semicolon_csv(path: Path) -> list[dict[str, str]]:
    """Read semicolon csv.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    
    Returns
    -------
    list[dict[str, str]]
        Computed result for the selected input and execution branch."""
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))

"""Classify an availability column and extract reduction settings.

Parameters
----------
column : str
    Charging-availability column name from an optimization summary.

Returns
-------
tuple of str
    Availability variant, reduction mode, and reduction factor.
    The factor is formatted to two decimal places. Reduction mode and
    factor are empty strings for columns without a recognized reduction
    suffix.

Notes
-----
Recognized variants are "technical", "refuel", "refuel_targetsoc",
and "refuel_targetsoc_reduced". Other columns are classified as "other".
Any matching "_reduced_flat..." or "_reduced_peak..." suffix is assigned
to "refuel_targetsoc_reduced", regardless of the preceding column name.
"""

def classify_availability(column: str) -> tuple[str, str, str]:
    """Return availability_variant, reduce_mode, reduce_factor."""
    column = column or ""

    reduce_match = re.search(
        r"_reduced_(flat|peak)([0-9]+(?:\.[0-9]+)?)_cp$",
        column,
    )
    if reduce_match:
        return (
            "refuel_targetsoc_reduced",
            reduce_match.group(1),
            f"{float(reduce_match.group(2)):.2f}",
        )

    if column == "ch_avail_cp":
        return "technical", "", ""
    if column == "ch_avail_refuel_cp":
        return "refuel", "", ""
    if column == "ch_avail_refuel_targetsoc_cp":
        return "refuel_targetsoc", "", ""

    return "other", "", ""

"""Classify the minimum-SoC constraint from its column name.

Parameters
----------
column : str
    Minimum-SoC column name.

Returns
-------
str
    "limited" if the name contains "_lim_" or ends in "_lim_cp";
    "standard" for any other non-empty name; otherwise an empty string.
"""

def classify_soc_min(column: str) -> str:
    """Classify soc min.
    
    Parameters
    ----------
    column : str
        Column used by this operation.
    
    Returns
    -------
    str
        Computed result for the selected input and execution branch."""
    column = column or ""
    if "_lim_" in column or column.endswith("_lim_cp"):
        return "limited"
    if column:
        return "standard"
    return ""

"""Normalize a successful optimization record for the combined table.

Parameters
----------
row : dict of str to str
    Record read from the successful-run summary.
ch_refuel : float
    Parameter value extracted from the scenario folder name.
scenario_folder : str
    Name of the source scenario folder.

Returns
-------
dict of str to str
    Record containing the standard output fields, original input fields,
    and derived availability and minimum-SoC classifications.

Notes
-----
Missing standard fields are initialized as empty strings. An existing
non-empty run_status is retained; otherwise it defaults to "success".
The input record is not modified.
"""

def normalize_success(
    row: dict[str, str],
    *,
    ch_refuel: float,
    scenario_folder: str,
) -> dict[str, str]:
    """Normalize success.
    
    Parameters
    ----------
    row : dict[str, str]
        Record containing the fields used by this operation.
    ch_refuel : float
        Plug-in motivation parameter associated with the scenario.
    scenario_folder : str
        Name of the source scenario folder.
    
    Returns
    -------
    dict[str, str]
        Result computed as ``normalized``."""
    availability = row.get("availability_column", "")
    variant, reduce_mode, reduce_factor = classify_availability(availability)

    normalized = {key: "" for key in BASE_OUTPUT_COLUMNS}
    normalized.update(row)
    normalized["ch_refuel"] = f"{ch_refuel:.1f}"
    normalized["scenario_folder"] = scenario_folder
    normalized["run_status"] = row.get("run_status") or "success"
    normalized["availability_variant"] = variant
    normalized["reduce_mode"] = reduce_mode
    normalized["reduce_factor"] = reduce_factor
    normalized["soc_min_variant"] = classify_soc_min(
        row.get("soc_min_column", "")
    )
    return normalized

"""Normalize a failed optimization record for the combined table.

Parameters
----------
row : dict of str to str
    Record read from the failed-run summary.
ch_refuel : float
    Parameter value extracted from the scenario folder name.
scenario_folder : str
    Name of the source scenario folder.

Returns
-------
dict of str to str
    Record with run_status="failed", available source-column information,
    derived classifications, and error details.

Notes
-----
If availability_column is empty, ch_avail_column is used instead.
Fields not explicitly copied from the input remain empty. No peak values
or reduction metrics are assigned. The input record is not modified.
"""

def normalize_failure(
    row: dict[str, str],
    *,
    ch_refuel: float,
    scenario_folder: str,
) -> dict[str, str]:
    """Normalize failure.
    
    Parameters
    ----------
    row : dict[str, str]
        Record containing the fields used by this operation.
    ch_refuel : float
        Plug-in motivation parameter associated with the scenario.
    scenario_folder : str
        Name of the source scenario folder.
    
    Returns
    -------
    dict[str, str]
        Result computed as ``normalized``."""
    availability = (
        row.get("availability_column")
        or row.get("ch_avail_column")
        or ""
    )
    variant, reduce_mode, reduce_factor = classify_availability(availability)

    normalized = {key: "" for key in BASE_OUTPUT_COLUMNS}
    normalized["ch_refuel"] = f"{ch_refuel:.1f}"
    normalized["scenario_folder"] = scenario_folder
    normalized["option"] = row.get("option", "")
    normalized["run_status"] = "failed"
    normalized["availability_column"] = availability
    normalized["availability_variant"] = variant
    normalized["reduce_mode"] = reduce_mode
    normalized["reduce_factor"] = reduce_factor
    normalized["soc_max_column"] = row.get("soc_max_column", "")
    normalized["soc_min_column"] = row.get("soc_min_column", "")
    normalized["soc_min_variant"] = classify_soc_min(
        row.get("soc_min_column", "")
    )
    normalized["charging_column"] = row.get("charging_column", "")
    normalized["consumption_column"] = row.get("consumption_column", "")
    normalized["ch_min_column"] = row.get("ch_min_column", "")
    normalized["dem_temp_column"] = row.get("dem_temp_column", "")
    normalized["error_type"] = row.get("error_type", "")
    normalized["error_message"] = row.get("error_message", "")
    return normalized

"""Find immediate subdirectories containing optimization summaries.

Parameters
----------
data_root : Path
    Directory containing the scenario folders.

Returns
-------
list of Path
    Matching folders sorted by their parsed ch_refuel values.

Notes
-----
A folder must end in a numeric suffix and contain SUCCESS_FILE or
FAILED_FILE. Nested directories are not searched recursively.
"""

def discover_scenario_folders(data_root: Path) -> list[Path]:
    """Discover scenario folders.
    
    Parameters
    ----------
    data_root : Path
        Root directory containing the input scenario data.
    
    Returns
    -------
    list[Path]
        Result computed as ``sorted(candidates, key=lambda p: parse_ch_refuel(p.name))``."""
    candidates: list[Path] = []

    for path in data_root.iterdir():
        if not path.is_dir():
            continue
        if not re.search(r"_[0-9]+(?:\.[0-9]+)?$", path.name):
            continue
        if (path / SUCCESS_FILE).exists() or (path / FAILED_FILE).exists():
            candidates.append(path)

    return sorted(candidates, key=lambda p: parse_ch_refuel(p.name))

"""Collect, normalize, and recalculate records from all scenario folders.

Parameters
----------
data_root : Path
    Directory containing the ch_refuel scenario folders.

Returns
-------
list of dict
    Combined successful and failed records, sorted by ch_refuel and
    option number. Records without an option are placed last within
    each ch_refuel group.

Raises
------
FileNotFoundError
    If no matching scenario folders are found.
ValueError
    If no suitable common uncontrolled baseline exists or a non-empty
    option value cannot be converted to an integer.

Notes
-----
Prints the number of successful and failed records in each folder.
Successful records are recalculated against a shared uncontrolled peak
by apply_fixed_uncontrolled_baseline.
"""

def collect_rows(data_root: Path) -> list[dict[str, str]]:
    """Collect rows.
    
    Parameters
    ----------
    data_root : Path
        Root directory containing the input scenario data.
    
    Returns
    -------
    list[dict[str, str]]
        Result computed as ``rows``.
    
    Raises
    ------
    FileNotFoundError
        If an explicit validation or processing check in this operation fails."""
    rows: list[dict[str, str]] = []

    scenario_folders = discover_scenario_folders(data_root)
    if not scenario_folders:
        raise FileNotFoundError(
            f"No scenario folders with {SUCCESS_FILE} / {FAILED_FILE} "
            f"found below {data_root}"
        )

    for folder in scenario_folders:
        ch_refuel = parse_ch_refuel(folder.name)

        successes = read_semicolon_csv(folder / SUCCESS_FILE)
        failures = read_semicolon_csv(folder / FAILED_FILE)

        rows.extend(
            normalize_success(
                row,
                ch_refuel=ch_refuel,
                scenario_folder=folder.name,
            )
            for row in successes
        )
        rows.extend(
            normalize_failure(
                row,
                ch_refuel=ch_refuel,
                scenario_folder=folder.name,
            )
            for row in failures
        )

        print(
            f"{folder.name}: ch_refuel={ch_refuel:.1f}, "
            f"success={len(successes)}, failed={len(failures)}, "
            f"total={len(successes) + len(failures)}"
        )

    apply_fixed_uncontrolled_baseline(rows)
    rows.sort(
        key=lambda row: (
            float(row["ch_refuel"]),
            int(row["option"]) if row["option"] else 10**9,
        )
    )
    return rows


def apply_fixed_uncontrolled_baseline(
    rows: list[dict[str, str]],
) -> float:
    """Use one available ch_refuel=0.4 uncontrolled peak."""

    baseline = None

    # 第一优先级：寻找明确标记为 ch_direct_cp 的 0.4 结果
    for row in rows:
        if row.get("run_status") != "success":
            continue

        try:
            ch_refuel = float(row["ch_refuel"])
            uncontrolled_peak = float(row["uncontrolled_peak_MW"])
        except (TypeError, ValueError, KeyError):
            continue

        if abs(ch_refuel - 0.4) > 1e-9:
            continue

        baseline_column = (
            row.get("uncontrolled_charging_column")
            or row.get("charging_column")
        )

        if (
            baseline_column == "ch_direct_cp"
            and uncontrolled_peak != 0
        ):
            baseline = uncontrolled_peak
            break

    # 如果没有列名称标记，使用第一条有效的 ch_refuel=0.4 结果
    if baseline is None:
        for row in rows:
            if row.get("run_status") != "success":
                continue

            try:
                ch_refuel = float(row["ch_refuel"])
                uncontrolled_peak = float(row["uncontrolled_peak_MW"])
            except (TypeError, ValueError, KeyError):
                continue

            if (
                abs(ch_refuel - 0.4) <= 1e-9
                and uncontrolled_peak != 0
            ):
                baseline = uncontrolled_peak
                break

    if baseline is None:
        raise ValueError(
            "No successful ch_refuel=0.4 result with a valid "
            "uncontrolled peak was found."
        )

    print(
        f"Fixed uncontrolled baseline: "
        f"ch_refuel=0.4, peak={baseline:.9f} MW"
    )

    for row in rows:
        if row.get("run_status") != "success":
            continue

        try:
            optimized_peak = float(row["optimized_peak_MW"])
        except (TypeError, ValueError, KeyError):
            continue

        peak_reduction = baseline - optimized_peak
        peak_reduction_percent = (
            peak_reduction / baseline * 100.0
        )
        optimized_to_uncontrolled_peak_ratio = (
            optimized_peak / baseline
        )

        row["uncontrolled_baseline_ch_refuel"] = "0.4"
        row["uncontrolled_peak_MW"] = f"{baseline:.9f}"
        row["optimized_to_uncontrolled_peak_ratio"] = (
            f"{optimized_to_uncontrolled_peak_ratio:.9f}"
        )
        row["optimizer_reported_uncontrolled_peak_MW"] = (
            f"{baseline:.9f}"
        )
        row["peak_reduction_MW"] = f"{peak_reduction:.9f}"
        row["peak_reduction_percent"] = (
            f"{peak_reduction_percent:.9f}"
        )

    return baseline


def write_csv(
    path: Path,
    rows: Iterable[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write csv.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    rows : Iterable[dict[str, str]]
        Records processed by this operation.
    fieldnames : list[str]
        Output CSV column names in the required order.
    
    Returns
    -------
    None"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(
    rows: list[dict[str, str]],
    *,
    value_column: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Build matrix.
    
    Parameters
    ----------
    rows : list[dict[str, str]]
        Records processed by this operation.
    value_column : str
        Name of the metric to place in the output matrix.
    
    Returns
    -------
    tuple[list[str], list[dict[str, str]]]
        Result computed as ``(headers, matrix_rows)``."""
    ch_values = sorted({float(row["ch_refuel"]) for row in rows})
    option_values = sorted(
        {
            int(row["option"])
            for row in rows
            if row.get("option", "").strip()
        }
    )

    lookup = {
        (int(row["option"]), float(row["ch_refuel"])): row
        for row in rows
        if row.get("option", "").strip()
    }

    headers = ["option"] + [f"ch_refuel_{value:.1f}" for value in ch_values]
    matrix_rows: list[dict[str, str]] = []

    for option in option_values:
        out = {"option": str(option)}
        for ch in ch_values:
            row = lookup.get((option, ch))
            key = f"ch_refuel_{ch:.1f}"
            if row is None or row.get("run_status") != "success":
                out[key] = ""
            else:
                out[key] = row.get(value_column, "")
        matrix_rows.append(out)

    return headers, matrix_rows


def build_status_matrix(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Build status matrix.
    
    Parameters
    ----------
    rows : list[dict[str, str]]
        Records processed by this operation.
    
    Returns
    -------
    tuple[list[str], list[dict[str, str]]]
        Result computed as ``(headers, matrix_rows)``."""
    ch_values = sorted({float(row["ch_refuel"]) for row in rows})
    option_values = sorted(
        {
            int(row["option"])
            for row in rows
            if row.get("option", "").strip()
        }
    )

    lookup = {
        (int(row["option"]), float(row["ch_refuel"])): row
        for row in rows
        if row.get("option", "").strip()
    }

    headers = ["option"] + [f"ch_refuel_{value:.1f}" for value in ch_values]
    matrix_rows: list[dict[str, str]] = []

    for option in option_values:
        out = {"option": str(option)}
        for ch in ch_values:
            row = lookup.get((option, ch))
            out[f"ch_refuel_{ch:.1f}"] = (
                "" if row is None else row.get("run_status", "")
            )
        matrix_rows.append(out)

    return headers, matrix_rows


def summarize_by_ch_refuel(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Summarize by ch refuel.
    
    Parameters
    ----------
    rows : list[dict[str, str]]
        Records processed by this operation.
    
    Returns
    -------
    list[dict[str, str]]
        Result computed as ``result``."""
    ch_values = sorted({float(row["ch_refuel"]) for row in rows})
    result: list[dict[str, str]] = []

    for ch in ch_values:
        group = [
            row for row in rows
            if float(row["ch_refuel"]) == ch
        ]
        successes = [
            row for row in group
            if row.get("run_status") == "success"
        ]
        failures = [
            row for row in group
            if row.get("run_status") == "failed"
        ]

        reductions = []
        for row in successes:
            try:
                reductions.append(float(row["peak_reduction_percent"]))
            except (TypeError, ValueError):
                pass

        result.append(
            {
                "ch_refuel": f"{ch:.1f}",
                "total_options": str(len(group)),
                "successful_options": str(len(successes)),
                "failed_options": str(len(failures)),
                "success_rate_percent": (
                    f"{100.0 * len(successes) / len(group):.2f}"
                    if group else ""
                ),
                "mean_peak_reduction_percent": (
                    f"{sum(reductions) / len(reductions):.6f}"
                    if reductions else ""
                ),
                "max_peak_reduction_percent": (
                    f"{max(reductions):.6f}"
                    if reductions else ""
                ),
                "min_peak_reduction_percent": (
                    f"{min(reductions):.6f}"
                    if reductions else ""
                ),
            }
        )

    return result


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If an explicit validation or processing check in this operation fails."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect greedy optimizer results from data_*_<ch_refuel> folders "
            "into analysis-ready CSV files."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Folder containing data_*_<ch_refuel> scenario folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <data-root>/greedy_collected_results"
        ),
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "greedy_collected_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(data_root)

    tidy_path = output_dir / "greedy_results_tidy.csv"
    write_csv(tidy_path, rows, BASE_OUTPUT_COLUMNS)

    matrix_headers, matrix_rows = build_matrix(
        rows,
        value_column="peak_reduction_percent",
    )
    matrix_path = output_dir / "peak_reduction_percent_matrix.csv"
    write_csv(matrix_path, matrix_rows, matrix_headers)

    ratio_headers, ratio_rows = build_matrix(
        rows,
        value_column="optimized_to_uncontrolled_peak_ratio",
    )
    ratio_path = (
        output_dir / "optimized_to_uncontrolled_peak_ratio_matrix.csv"
    )
    write_csv(ratio_path, ratio_rows, ratio_headers)

    status_headers, status_rows = build_status_matrix(rows)
    status_path = output_dir / "run_status_matrix.csv"
    write_csv(status_path, status_rows, status_headers)

    ch_summary = summarize_by_ch_refuel(rows)
    ch_summary_path = output_dir / "ch_refuel_summary.csv"
    write_csv(
        ch_summary_path,
        ch_summary,
        [
            "ch_refuel",
            "total_options",
            "successful_options",
            "failed_options",
            "success_rate_percent",
            "mean_peak_reduction_percent",
            "max_peak_reduction_percent",
            "min_peak_reduction_percent",
        ],
    )

    success_count = sum(
        row["run_status"] == "success" for row in rows
    )
    failure_count = sum(
        row["run_status"] == "failed" for row in rows
    )

    print()
    print("=" * 72)
    print(f"Rows collected: {len(rows)}")
    print(f"Successful:     {success_count}")
    print(f"Failed:         {failure_count}")
    print(f"Tidy data:      {tidy_path}")
    print(f"Peak matrix:    {matrix_path}")
    print(f"Peak ratio:     {ratio_path}")
    print(f"Status matrix:  {status_path}")
    print(f"ch_refuel info: {ch_summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
