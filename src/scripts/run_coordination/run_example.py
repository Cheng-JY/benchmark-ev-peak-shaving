r"""Run the greedy EV optimizer for every SOURCE_COLUMNS option in option1.txt.

The script reads an aggregated EV profile, heat-pump load, household load,
and ``option1.txt`` containing ``optionN: SOURCE_COLUMNS = {...}`` blocks.

Each option is run independently and written to ``greedy_results_N`` with:

* ``greedy_peak_shaving_results.csv``
* ``greedy_peak_shaving_summary.csv``
* ``source_columns.csv``

python src\scripts\run_example.py ^
    --ev-csv "data\data_20260922_2\ch_profiles_agg_home_DEU_20260922_2026_cal2026_R0_0.8.csv" ^
    --heat-pump-csv "data\data_20260922_2\heat_pump_profiles.csv" ^
    --household-csv "data\data_20260922_2\household_load.csv" ^
    --options-file "data\data_20260922_2\option1.txt" ^
    --output-root "data\data_20260922_2\greedy_results_0.8"

A combined ``greedy_all_options_summary.csv`` is written to the output root.
All optimizer calculations use MWh per one-hour step; power outputs are MW.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# Allow the file to be run directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.components.optimizer.greedy_optimizer import (
    GreedyEVOptimizer,
)
from src.components.optimizer.optimization_config import (  # noqa: E402
    OptimizationConfig,
)


STEP_HOURS = 1.0
OPTIMIZATION_INTERVAL = pd.Timedelta(hours=1)
LEGACY_INTERVAL = pd.Timedelta(minutes=15)
CONTROLLED_NAME = "controlled_v2g"
UNCONTROLLED_REFERENCE_AVAILABILITY = "ch_avail_refuel_cp"
UNCONTROLLED_BASELINE_CH_REFUEL = 0.4
DEFAULT_CHARGING_EFFICIENCY = 0.95
CHARGING_PARAMETER_SHEET = "Charging_parameter"
UNCONTROLLED_CHARGING_COLUMN = "ch_refuel_cp"
REQUIRED_SOURCE_KEYS = {
    "ch_avail",
    "soc_max",
    "soc_min",
    "charging",
    "consumption",
    "dem_temp",
}
OPTION_PATTERN = re.compile(r"^\s*option(\d+)\s*:\s*$", re.IGNORECASE)
DEFAULT_OPTIONS_FILE = Path(__file__).with_name("option1.txt")


@dataclass(frozen=True)
class BaseLoad:
    """Represent base load and its associated operations."""
    index: pd.DatetimeIndex
    household_MW: pd.Series
    heat_pump_MW: pd.Series
    total_MWh: pd.Series


@dataclass(frozen=True)
class EVProfileState:
    """Hourly fleet profile used by the direct-CSV example."""

    ch_avail: pd.Series
    soc_max: pd.Series
    soc_min: pd.Series
    charging: pd.Series
    consumption: pd.Series
    dem_temp: pd.Series
    ch_min: pd.Series | None
    charging_efficiency: float
    discharging_efficiency: float
    source_columns: dict[str, str]


@dataclass(frozen=True)
class RunOutputs:
    """Represent run outputs and its associated operations."""
    result_csv: Path
    summary_csv: Path


@dataclass(frozen=True)
class OptionDefinition:
    """Represent option definition and its associated operations."""
    number: int
    source_columns: dict[str, str]


def resolve_baseline_ev_csv(
    ev_csv: Path,
    explicit_baseline_csv: Path | None,
) -> Path:
    """Resolve the *_0.4.csv file used as uncontrolled baseline."""

    # If explicitly specified on the command line, use that file.
    if explicit_baseline_csv is not None:
        baseline = explicit_baseline_csv.expanduser().resolve()
        if not baseline.is_file():
            raise FileNotFoundError(
                f"Uncontrolled ch_refuel=0.4 baseline CSV not found: {baseline}"
            )
        return baseline

    current = ev_csv.expanduser().resolve()
    if not current.is_file():
        raise FileNotFoundError(f"Controlled EV CSV not found: {current}")

    # Match the ch_refuel value at the end of the filename:
    # ..._0.1.csv, ..._0.4.csv, ..._1.csv, or ..._1.0.csv
    match = re.match(
        r"^(?P<prefix>.+)_(?:0(?:\.\d+)?|1(?:\.0+)?)$",
        current.stem,
    )

    if match is None:
        raise ValueError(
            "Cannot infer the ch_refuel=0.4 baseline from the EV filename. "
            "The filename must end with _0.1.csv, ..., _0.9.csv, or _1.csv. "
            "Alternatively, pass --uncontrolled-baseline-ev-csv explicitly."
        )

    baseline_name = (
        f"{match.group('prefix')}_"
        f"{UNCONTROLLED_BASELINE_CH_REFUEL:.1f}"
        f"{current.suffix}"
    )
    baseline = current.with_name(baseline_name)

    if not baseline.is_file():
        raise FileNotFoundError(
            "The automatically inferred ch_refuel=0.4 baseline CSV "
            f"does not exist: {baseline}. "
            "Pass --uncontrolled-baseline-ev-csv explicitly."
        )

    return baseline

class _UnusedScenarioData:
    """Placeholder required by ``GreedyEVOptimizer.__init__``.

    The direct-CSV workflow supplies all arrays explicitly and therefore does
    not call the scenario-data reader in ``BaseEVOptimizer``.
    """


class _UnusedScenarioConfig:
    """Placeholder required by ``GreedyEVOptimizer.__init__``."""


class _CopyOnGetDict(dict):
    """Return a fresh SoC array for every feasibility calculation branch."""

    def get(self, key, default=None):
        """Return .
        
        Parameters
        ----------
        key : object
            Key used by this operation.
        default : object
            Default used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        value = super().get(key, default)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value


class DirectCSVGreedyOptimizer(GreedyEVOptimizer):
    """Greedy optimizer variant with independent SoC trials.

    ``update_charging_smart`` and ``update_charging_v2g`` update their SoC
    arrays in-place.  The binary search must start every threshold trial from
    the same initial SoC profile, so this wrapper supplies a fresh copy.
    """
    def compute_feasibility(
        self,
        threshold,
        load,
        ch_avail_dict,
        soc_max_dict,
        soc_min_dict,
        charging_dict,
        consumption_dict,
        soc_dict,
        controlled_name,
    ):
        """Check whether the proposed trajectory satisfies the modeled feasibility conditions.
        
        Parameters
        ----------
        threshold : object
            Threshold used by this operation.
        load : object
            Load used by this operation.
        ch_avail_dict : object
            Ch avail dict used by this operation.
        soc_max_dict : object
            Soc max dict used by this operation.
        soc_min_dict : object
            Soc min dict used by this operation.
        charging_dict : object
            Charging dict used by this operation.
        consumption_dict : object
            Consumption dict used by this operation.
        soc_dict : object
            Soc dict used by this operation.
        controlled_name : object
            Controlled name used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        return super().compute_feasibility(
            threshold,
            load,
            ch_avail_dict,
            soc_max_dict,
            soc_min_dict,
            charging_dict,
            consumption_dict,
            _CopyOnGetDict(soc_dict),
            controlled_name,
        )


def load_options_file(path: Path) -> list[OptionDefinition]:
    """Parse ``optionN: SOURCE_COLUMNS = {...}`` blocks without executing them."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Options file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    options: list[OptionDefinition] = []
    seen_numbers: set[int] = set()
    i = 0

    while i < len(lines):
        match = OPTION_PATTERN.match(lines[i])
        if match is None:
            i += 1
            continue

        number = int(match.group(1))
        if number in seen_numbers:
            raise ValueError(f"{path.name}: duplicate option{number}.")

        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines) or not lines[i].lstrip().startswith("SOURCE_COLUMNS"):
            raise ValueError(
                f"{path.name}: option{number} must be followed by SOURCE_COLUMNS = {{...}}."
            )

        assignment = lines[i]
        if "=" not in assignment:
            raise ValueError(f"{path.name}: invalid SOURCE_COLUMNS for option{number}.")

        rhs = assignment.split("=", 1)[1].strip()
        dict_lines = [rhs]
        brace_depth = rhs.count("{") - rhs.count("}")
        i += 1

        while brace_depth > 0 and i < len(lines):
            line = lines[i]
            dict_lines.append(line)
            brace_depth += line.count("{") - line.count("}")
            i += 1

        if brace_depth != 0:
            raise ValueError(f"{path.name}: unclosed SOURCE_COLUMNS for option{number}.")

        try:
            source_columns = ast.literal_eval("\n".join(dict_lines))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{path.name}: cannot parse SOURCE_COLUMNS for option{number}: {exc}"
            ) from exc

        if not isinstance(source_columns, dict):
            raise ValueError(
                f"{path.name}: SOURCE_COLUMNS in option{number} must be a dict."
            )

        source_columns = {
            str(key): str(value) for key, value in source_columns.items()
        }

        missing = sorted(REQUIRED_SOURCE_KEYS - source_columns.keys())
        if missing:
            raise ValueError(
                f"{path.name}: option{number} is missing keys {missing}."
            )

        seen_numbers.add(number)
        options.append(
            OptionDefinition(
                number=number,
                source_columns=source_columns,
            )
        )

    if not options:
        raise ValueError(f"{path.name}: no optionN blocks were found.")

    options.sort(key=lambda item: item.number)
    return options


def validate_option_columns(
    ev_csv: Path,
    options: list[OptionDefinition],
) -> None:
    """Fail before optimization if any option references a missing EV column."""

    ev_csv = ev_csv.expanduser().resolve()
    if not ev_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {ev_csv}")

    available = set(pd.read_csv(ev_csv, sep=";", nrows=0).columns)
    problems: list[str] = []

    for option in options:
        missing = sorted(
            {
                column
                for column in option.source_columns.values()
                if column not in available
            }
        )
        if missing:
            problems.append(
                f"option{option.number}: missing {missing}"
            )

    if problems:
        raise ValueError(
            f"{ev_csv.name}: option1.txt references column(s) not present in "
            "the EV CSV:\n  " + "\n  ".join(problems)
        )


def _read_profile(path: Path) -> pd.DataFrame:
    """Read profile.
    
    Parameters
    ----------
    path : Path
        File or directory path used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``frame``.
    
    Raises
    ------
    FileNotFoundError
        If an explicit validation or processing check in this operation fails.
    ValueError
        If an explicit validation or processing check in this operation fails."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    frame = pd.read_csv(path, sep=";", index_col=0)
    parsed_index = pd.to_datetime(frame.index, errors="coerce")
    if parsed_index.isna().any():
        raise ValueError(f"{path.name}: the first column is not a complete datetime index.")

    frame.index = pd.DatetimeIndex(parsed_index)
    if frame.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate timestamps were found.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path.name}: timestamps are not sorted.")
    return frame


def _require_numeric_column(
    frame: pd.DataFrame,
    column: str,
    source_path: Path,
) -> pd.Series:
    """Require numeric column.
    
    Parameters
    ----------
    frame : pd.DataFrame
        Frame used by this operation.
    column : str
        Column used by this operation.
    source_path : Path
        Source path used by this operation.
    
    Returns
    -------
    pd.Series
        Result computed as ``values.astype(float)``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    if column not in frame.columns:
        raise ValueError(f"{source_path.name}: required column '{column}' is missing.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"{source_path.name}: column '{column}' contains missing or non-numeric values."
        )
    return values.astype(float)


def _profile_interval(
    index: pd.DatetimeIndex,
    source_path: Path,
) -> pd.Timedelta:
    """Return the continuous source interval accepted by this workflow."""

    if len(index) <= 1:
        return OPTIMIZATION_INTERVAL
    intervals = index.to_series().diff().dropna().unique()
    if len(intervals) != 1:
        raise ValueError(f"{source_path.name}: timestamps are not continuous.")
    interval = pd.Timedelta(intervals[0])
    if interval not in {LEGACY_INTERVAL, OPTIMIZATION_INTERVAL}:
        raise ValueError(
            f"{source_path.name}: expected 15-minute or hourly values, "
            f"found {interval}."
        )
    return interval


def _to_hourly_power(
    values: pd.Series,
    source_path: Path,
) -> pd.Series:
    """Convert a power series to hourly mean power without changing energy."""

    interval = _profile_interval(values.index, source_path)
    if interval == OPTIMIZATION_INTERVAL:
        return values.astype(float)

    counts = values.resample("1h").count()
    if not counts.eq(4).all():
        incomplete = counts.index[~counts.eq(4)][:3].tolist()
        raise ValueError(
            f"{source_path.name}: incomplete 15-minute data in hourly bin(s), "
            f"for example {incomplete}."
        )
    return values.resample("1h").mean()


def _to_hourly_values(
    values: pd.Series,
    source_path: Path,
    *,
    value_kind: str,
) -> pd.Series:
    """Convert 15-minute or hourly EV values to one-hour optimizer steps."""

    interval = _profile_interval(values.index, source_path)
    if interval == OPTIMIZATION_INTERVAL:
        return values.astype(float)

    counts = values.resample("1h").count()
    if not counts.eq(4).all():
        incomplete = counts.index[~counts.eq(4)][:3].tolist()
        raise ValueError(
            f"{source_path.name}: incomplete 15-minute EV data in hourly bin(s), "
            f"for example {incomplete}."
        )
    if value_kind == "power":
        return values.resample("1h").mean()
    if value_kind == "energy":
        return values.resample("1h").sum()
    if value_kind == "state":
        return values.resample("1h").first()
    raise ValueError(f"Unknown EV value kind: {value_kind!r}.")


def _validate_efficiency(name: str, value: float) -> float:
    """Validate the supplied efficiency value before using it in calculations.
    
    Parameters
    ----------
    name : str
        Name used by this operation.
    value : float
        Value used by this operation.
    
    Returns
    -------
    float
        Result computed as ``value``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1], got {value!r}.")
    return value


def _read_normal_charging_efficiency(source_path: Path) -> float:
    """Read the normal charging efficiency used by a4_SOC_Generation."""

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Charging-parameter workbook not found: {source_path}"
        )
    parameters = pd.read_excel(
        source_path,
        sheet_name=CHARGING_PARAMETER_SHEET,
        header=None,
        usecols=[0, 1],
    )
    keys = parameters.iloc[:, 0].astype(str).str.strip()
    matches = parameters.loc[
        keys.eq("charging_efficiency_normal"),
        parameters.columns[1],
    ]
    if matches.empty:
        raise ValueError(
            f"{source_path.name}: sheet '{CHARGING_PARAMETER_SHEET}' contains "
            "no 'charging_efficiency_normal' entry."
        )
    value = pd.to_numeric(matches.iloc[0], errors="coerce")
    if pd.isna(value):
        raise ValueError(
            f"{source_path.name}: 'charging_efficiency_normal' is not numeric."
        )
    return _validate_efficiency("charging_efficiency", float(value))


def load_hourly_ev_profile(
    ev_csv: Path,
    *,
    source_columns: dict[str, str],
    vehicle_count: int,
    start_discharging_kwh: float,
    expected_index: pd.DatetimeIndex,
    charging_efficiency: float | None,
    discharging_efficiency: float | None,
    charging_parameters_xlsx: Path | None,
) -> EVProfileState:
    """Read and scale EV data locally, without adapter time-step conversion."""

    profile = _read_profile(ev_csv)
    names_to_load = list(REQUIRED_SOURCE_KEYS)
    if "ch_min" in source_columns:
        names_to_load.append("ch_min")

    raw = {
        name: _require_numeric_column(profile, source_columns[name], ev_csv)
        for name in names_to_load
    }
    for name, values in raw.items():
        if (values < -1e-9).any():
            raise ValueError(
                f"{ev_csv.name}: '{source_columns[name]}' contains negative values."
            )
    if (raw["soc_min"] > raw["soc_max"] + 1e-6).any():
        raise ValueError(f"{ev_csv.name}: soc_min exceeds soc_max.")

    connected = raw["ch_avail"] > 0
    raw["soc_min"] = raw["soc_min"].where(
        ~connected,
        np.maximum(raw["soc_min"], float(start_discharging_kwh)),
    )
    raw["soc_min"] = np.minimum(raw["soc_min"], raw["soc_max"])

    kinds = {
        "ch_avail": "power",
        "soc_max": "state",
        "soc_min": "state",
        "charging": "energy",
        "consumption": "energy",
        "dem_temp": "energy",
        "ch_min": "power",
    }
    hourly = {
        name: _to_hourly_values(values, ev_csv, value_kind=kinds[name])
        for name, values in raw.items()
    }
    lengths = {name: len(values) for name, values in hourly.items()}
    if any(length != len(expected_index) for length in lengths.values()):
        raise ValueError(
            f"{ev_csv.name}: converted EV profile has {lengths['ch_avail']} "
            f"hourly steps, but feeder load has {len(expected_index)}."
        )

    if charging_efficiency is None:
        charging_efficiency = (
            _read_normal_charging_efficiency(charging_parameters_xlsx)
            if charging_parameters_xlsx is not None
            else DEFAULT_CHARGING_EFFICIENCY
        )
    charging_efficiency = _validate_efficiency(
        "charging_efficiency", charging_efficiency
    )
    if discharging_efficiency is None:
        discharging_efficiency = charging_efficiency
    discharging_efficiency = _validate_efficiency(
        "discharging_efficiency", discharging_efficiency
    )

    scale = float(vehicle_count) / 1000.0

    def scaled(name: str, output_name: str) -> pd.Series:
        """Scaled .
        
        Parameters
        ----------
        name : str
            Name used by this operation.
        output_name : str
            Output name used by this operation.
        
        Returns
        -------
        pd.Series
            Result computed as ``pd.Series(hourly[name].to_numpy(dtype=float) * scale, index=expected_index, name=output_name)``."""
        return pd.Series(
            hourly[name].to_numpy(dtype=float) * scale,
            index=expected_index,
            name=output_name,
        )

    return EVProfileState(
        ch_avail=scaled("ch_avail", "ch_avail_MW"),
        soc_max=scaled("soc_max", "soc_max_MWh"),
        soc_min=scaled("soc_min", "soc_min_MWh"),
        charging=scaled("charging", "ch_uncontrolled_MWh"),
        consumption=scaled("consumption", "consumption_MWh"),
        dem_temp=scaled("dem_temp", "dem_temp_MWh"),
        ch_min=(
            scaled("ch_min", "charging_power_min_MW")
            if "ch_min" in hourly
            else None
        ),
        charging_efficiency=charging_efficiency,
        discharging_efficiency=discharging_efficiency,
        source_columns=source_columns.copy(),
    )


def load_base_load(household_csv: Path, heat_pump_csv: Path) -> BaseLoad:
    """Read base-load profiles and return hourly MW and MWh.

    New hourly CSV files are used directly.  Existing 15-minute files remain
    supported and are aggregated with the hourly mean because both source
    columns represent power.
    """

    household = _read_profile(household_csv)
    heat_pump = _read_profile(heat_pump_csv)

    household_MW = _to_hourly_power(
        _require_numeric_column(household, "hh_load (MW)", household_csv),
        household_csv,
    )
    heat_pump_MW = _to_hourly_power(
        _require_numeric_column(heat_pump, "total_all", heat_pump_csv),
        heat_pump_csv,
    ) / 1000.0
    if not household_MW.index.equals(heat_pump_MW.index):
        raise ValueError(
            "Household and heat-pump timestamps do not match after conversion "
            "to hourly resolution."
        )
    household_MW.name = "household_load_MW"
    heat_pump_MW.name = "heat_pump_load_MW"

    total_MWh = (household_MW + heat_pump_MW) * STEP_HOURS
    total_MWh.name = "base_load_MWh"
    return BaseLoad(
        index=household_MW.index,
        household_MW=household_MW,
        heat_pump_MW=heat_pump_MW,
        total_MWh=total_MWh,
    )


def infer_vehicle_count(ev_csv: Path) -> int:
    """Read the exported ``nprofiles`` value."""

    profile = _read_profile(ev_csv)
    if "nprofiles" not in profile.columns:
        raise ValueError(
            f"{ev_csv.name}: 'nprofiles' is missing; pass --vehicle-count explicitly."
        )

    values = pd.to_numeric(profile["nprofiles"], errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        raise ValueError(
            f"{ev_csv.name}: no positive nprofiles value; pass --vehicle-count explicitly."
        )

    rounded = np.rint(values.to_numpy(dtype=float)).astype(int)
    if not np.allclose(values.to_numpy(dtype=float), rounded):
        raise ValueError(f"{ev_csv.name}: nprofiles must contain integer counts.")
    unique = np.unique(rounded)
    if len(unique) != 1:
        raise ValueError(
            f"{ev_csv.name}: multiple nprofiles values found: {unique.tolist()}."
        )
    return int(unique[0])


def load_selected_availability(
    ev_csv: Path,
    column: str,
    vehicle_count: int,
    expected_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load one EV charging-power column as fleet MW at hourly resolution."""

    profile = _read_profile(ev_csv)
    values = _require_numeric_column(profile, column, ev_csv)
    if (values < -1e-9).any():
        raise ValueError(f"{ev_csv.name}: '{column}' contains negative power.")

    if len(profile.index) > 1:
        intervals = profile.index.to_series().diff().dropna().unique()
        if len(intervals) != 1:
            raise ValueError(f"{ev_csv.name}: EV timestamps are not continuous.")
        interval = pd.Timedelta(intervals[0])
    else:
        interval = pd.Timedelta(minutes=15)

    if interval == OPTIMIZATION_INTERVAL:
        values_hourly = values.to_numpy(dtype=float)
    elif interval == LEGACY_INTERVAL:
        if len(values) % 4:
            raise ValueError(
                f"{ev_csv.name}: 15-minute availability length must be divisible by 4."
            )
        values_hourly = values.to_numpy(dtype=float).reshape(-1, 4).mean(axis=1)
    else:
        raise ValueError(
            f"{ev_csv.name}: EV interval must be hourly or 15-minute, found {interval}."
        )

    if len(values_hourly) != len(expected_index):
        raise ValueError(
            f"{ev_csv.name}: availability has {len(values_hourly)} converted steps, "
            f"but feeder load has {len(expected_index)}."
        )

    # kW per representative vehicle -> fleet MW
    return pd.Series(
        values_hourly * vehicle_count / 1000.0,
        index=expected_index,
        name="charging_power_available_MW",
    )


def load_uncontrolled_charging(
    ev_csv: Path,
    *,
    charging_column: str,
    vehicle_count: int,
    expected_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load the fixed uncontrolled charging reference in fleet MWh."""

    profile = _read_profile(ev_csv)
    charging = _require_numeric_column(profile, charging_column, ev_csv)
    if (charging < -1e-9).any():
        raise ValueError(
            f"{ev_csv.name}: '{charging_column}' contains negative charging values."
        )
    hourly = _to_hourly_values(charging, ev_csv, value_kind="energy")
    if len(hourly) != len(expected_index):
        raise ValueError(
            f"{ev_csv.name}: uncontrolled charging has {len(hourly)} hourly "
            f"steps, but feeder load has {len(expected_index)}."
        )
    return pd.Series(
        hourly.to_numpy(dtype=float) * float(vehicle_count) / 1000.0,
        index=expected_index,
        name="ev_uncontrolled_reference_MWh",
    )


def run_peak_shaving(
    *,
    ev_csv: Path,
    uncontrolled_baseline_ev_csv: Path,
    heat_pump_csv: Path,
    household_csv: Path,
    output_dir: Path,
    source_columns: dict[str, str],
    uncontrolled_charging_column: str,
    option_number: int | None,
    vehicle_count: int | None,
    start_discharging_kwh: float,
    tolerance_MWh: float,
    charging_efficiency: float | None = None,
    discharging_efficiency: float | None = None,
    charging_parameters_xlsx: Path | None = None,
) -> RunOutputs:
    """Execute greedy peak shaving and save detailed and summary CSV files."""
    print("excute in run_example")
    ev_csv = ev_csv.resolve()
    uncontrolled_baseline_ev_csv = uncontrolled_baseline_ev_csv.resolve()
    heat_pump_csv = heat_pump_csv.resolve()
    household_csv = household_csv.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_columns_path = output_dir / "source_columns.csv"
    pd.DataFrame(
        [
            {"parameter": key, "column": value}
            for key, value in source_columns.items()
        ]
    ).to_csv(source_columns_path, sep=";", index=False)

    if vehicle_count is None:
        vehicle_count = infer_vehicle_count(ev_csv)
    if vehicle_count <= 0:
        raise ValueError("vehicle_count must be positive.")
    if start_discharging_kwh < 0:
        raise ValueError("start_discharging_kwh must be non-negative.")
    if tolerance_MWh <= 0:
        raise ValueError("tolerance_MWh must be positive.")

    base = load_base_load(household_csv, heat_pump_csv)
    state = load_hourly_ev_profile(
        ev_csv,
        source_columns=source_columns,
        vehicle_count=vehicle_count,
        start_discharging_kwh=start_discharging_kwh,
        expected_index=base.index,
        charging_efficiency=charging_efficiency,
        discharging_efficiency=discharging_efficiency,
        charging_parameters_xlsx=charging_parameters_xlsx,
    )
    uncontrolled_reference_MWh = load_uncontrolled_charging(
        uncontrolled_baseline_ev_csv,
        charging_column=uncontrolled_charging_column,
        vehicle_count=vehicle_count,
        expected_index=base.index,
    )

    ch_avail_MW = state.ch_avail.rename("charging_power_available_MW")
    availability_used = state.source_columns["ch_avail"]

    opt_cfg = OptimizationConfig()
    opt_cfg.controlled_name = CONTROLLED_NAME
    optimizer = DirectCSVGreedyOptimizer(
        opt_config=opt_cfg,
        scenario_config=_UnusedScenarioConfig(),
        scenario_data=_UnusedScenarioData(),
        charging_efficiency=state.charging_efficiency,
        discharging_efficiency=state.discharging_efficiency,
    )
    optimizer.tolerance = tolerance_MWh
    SOC_TOLERANCE_KWH_PER_VEHICLE = 0.01

    optimizer.soc_feasibility_tolerance = max(
        1e-5,
        vehicle_count * SOC_TOLERANCE_KWH_PER_VEHICLE / 1000.0
    )
    

    # Greedy state transitions use energy per one-hour time step.
    ch_avail_MWh = ch_avail_MW.to_numpy(dtype=float) * STEP_HOURS
    soc_max_MWh = state.soc_max.to_numpy(dtype=float)
    soc_min_MWh = state.soc_min.to_numpy(dtype=float)
    charging_uncontrolled_MWh = state.charging.to_numpy(dtype=float)
    consumption_MWh = state.consumption.to_numpy(dtype=float)
    soc_initial_MWh = optimizer.get_soc(
        ch_avail_MWh,
        soc_max_MWh[0],
        charging_uncontrolled_MWh,
        consumption_MWh,
    )

    result, threshold_MWh, uncontrolled_peak_MWh = optimizer.get_peak_reduction(
        base.total_MWh,
        {CONTROLLED_NAME: ch_avail_MWh},
        {CONTROLLED_NAME: soc_max_MWh},
        {CONTROLLED_NAME: soc_min_MWh},
        {CONTROLLED_NAME: charging_uncontrolled_MWh},
        {CONTROLLED_NAME: consumption_MWh},
        {CONTROLLED_NAME: soc_initial_MWh},
        uncontrolled_charging_dict={
            CONTROLLED_NAME: uncontrolled_reference_MWh.to_numpy(dtype=float)
        },
    )

    optimized_MWh = result[f"charging_updated_{CONTROLLED_NAME}"].astype(float)
    uncontrolled_MWh = uncontrolled_reference_MWh
    soc_optimized_MWh = result[f"SOC_update_{CONTROLLED_NAME}"].astype(float)
    feasible = result[f"feasible_{CONTROLLED_NAME}"].astype(bool)

    detailed = pd.DataFrame(index=base.index)
    detailed.index.name = "time"
    detailed["household_load_MW"] = base.household_MW
    detailed["heat_pump_load_MW"] = base.heat_pump_MW
    detailed["base_load_MW"] = base.total_MWh / STEP_HOURS
    detailed["ev_uncontrolled_MWh"] = uncontrolled_MWh
    detailed["ev_uncontrolled_MW"] = uncontrolled_MWh / STEP_HOURS
    detailed["ev_optimized_MWh"] = optimized_MWh
    detailed["ev_optimized_MW"] = optimized_MWh / STEP_HOURS
    detailed["feeder_uncontrolled_MW"] = (
        detailed["base_load_MW"] + detailed["ev_uncontrolled_MW"]
    )
    detailed["feeder_optimized_MW"] = (
        detailed["base_load_MW"] + detailed["ev_optimized_MW"]
    )
    detailed["charging_power_available_MW"] = ch_avail_MW
    if state.ch_min is not None:
        detailed["charging_power_min_MW"] = state.ch_min
    detailed["consumption_MWh"] = state.consumption
    detailed["soc_min_MWh"] = state.soc_min
    detailed["soc_max_MWh"] = state.soc_max
    detailed["soc_optimized_MWh"] = soc_optimized_MWh
    detailed["soc_optimized_per_vehicle_kWh"] = (
        soc_optimized_MWh * 1000.0 / vehicle_count
    )
    detailed["shifted_MWh"] = result["shifted"].astype(float)
    detailed["feasible"] = feasible
    detailed["peak_threshold_MW"] = threshold_MWh / STEP_HOURS
    detailed["charging_efficiency"] = state.charging_efficiency
    detailed["discharging_efficiency"] = state.discharging_efficiency

    result_path = output_dir / "greedy_peak_shaving_results.csv"
    detailed.to_csv(result_path, sep=";", float_format="%.9f")

    uncontrolled_peak_MW = float(detailed["feeder_uncontrolled_MW"].max())
    optimized_peak_MW = float(detailed["feeder_optimized_MW"].max())
    peak_reduction_MW = uncontrolled_peak_MW - optimized_peak_MW
    peak_reduction_pct = (
        0.0
        if uncontrolled_peak_MW == 0
        else peak_reduction_MW / uncontrolled_peak_MW * 100.0
    )
    summary = pd.DataFrame(
        [
            {
                "option": option_number,
                "vehicle_count": vehicle_count,
                "start_discharging_kWh_per_vehicle": start_discharging_kwh,
                "availability_column": availability_used,
                "soc_max_column": state.source_columns["soc_max"],
                "soc_min_column": state.source_columns["soc_min"],
                "charging_column": state.source_columns["charging"],
                "uncontrolled_baseline_ch_refuel": (
                    UNCONTROLLED_BASELINE_CH_REFUEL
                ),
                "uncontrolled_baseline_ev_csv": (
                    uncontrolled_baseline_ev_csv.name
                ),
                "uncontrolled_baseline_availability_column": (
                    UNCONTROLLED_REFERENCE_AVAILABILITY
                ),
                "uncontrolled_charging_column": uncontrolled_charging_column,
                "consumption_column": state.source_columns["consumption"],
                "ch_min_column": state.source_columns.get("ch_min", ""),
                "dem_temp_column": state.source_columns["dem_temp"],
                "charging_efficiency": state.charging_efficiency,
                "discharging_efficiency": state.discharging_efficiency,
                "time_steps": len(detailed),
                "uncontrolled_peak_MW": uncontrolled_peak_MW,
                "optimized_peak_MW": optimized_peak_MW,
                "peak_threshold_MW": threshold_MWh / STEP_HOURS,
                "peak_reduction_MW": peak_reduction_MW,
                "peak_reduction_percent": peak_reduction_pct,
                "all_steps_feasible": bool(feasible.all()),
                "infeasible_steps": int((~feasible).sum()),
                "optimizer_reported_uncontrolled_peak_MW": (
                    uncontrolled_peak_MWh / STEP_HOURS
                ),
            }
        ]
    )
    summary_path = output_dir / "greedy_peak_shaving_summary.csv"
    summary.to_csv(summary_path, sep=";", index=False, float_format="%.9f")

    if option_number is not None:
        print(f"Option: {option_number}")
    print(f"Vehicle count: {vehicle_count}")
    print(f"Availability column: {availability_used}")
    print(
        "Uncontrolled baseline: "
        f"ch_refuel={UNCONTROLLED_BASELINE_CH_REFUEL:.1f}, "
        f"file={uncontrolled_baseline_ev_csv.name}, "
        f"charging={uncontrolled_charging_column}"
    )
    print(f"SoC minimum column: {state.source_columns['soc_min']}")
    print(f"Charging efficiency: {state.charging_efficiency:.4f}")
    print(f"Discharging efficiency: {state.discharging_efficiency:.4f}")
    print(f"Uncontrolled peak: {uncontrolled_peak_MW:.6f} MW")
    print(f"Optimized peak: {optimized_peak_MW:.6f} MW")
    print(
        f"Peak reduction: {peak_reduction_MW:.6f} MW "
        f"({peak_reduction_pct:.2f}%)"
    )
    print(f"All steps feasible: {bool(feasible.all())}")
    print(f"Saved detailed result: {result_path}")
    print(f"Saved summary: {summary_path}")
    return RunOutputs(result_csv=result_path, summary_csv=summary_path)


def run_all_options(
    *,
    options_file: Path,
    ev_csv: Path,
    uncontrolled_baseline_ev_csv: Path | None,
    heat_pump_csv: Path,
    household_csv: Path,
    output_root: Path | None,
    vehicle_count: int | None,
    start_discharging_kwh: float,
    tolerance_MWh: float,
    charging_efficiency: float | None,
    discharging_efficiency: float | None,
    charging_parameters_xlsx: Path | None,
    selected_options: set[int] | None = None,
) -> Path:
    """Run every configuration independently and continue after option errors."""

    options = load_options_file(options_file)

    # The uncontrolled reference is always ch_refuel_cp from the *_0.4.csv file.
    uncontrolled_charging_column = UNCONTROLLED_CHARGING_COLUMN

    baseline_ev_csv = resolve_baseline_ev_csv(
        ev_csv,
        uncontrolled_baseline_ev_csv,
    )


    if selected_options is not None:
        options = [item for item in options if item.number in selected_options]
        missing = sorted(
            selected_options - {item.number for item in options}
        )
        if missing:
            print(
                "WARNING: requested option(s) not found in "
                f"{options_file.name}: {missing}. Continuing with the options "
                "that were found."
            )

    output_root = (
        ev_csv.expanduser().resolve().parent
        if output_root is None
        else output_root.expanduser().resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Read the EV CSV header only once. Missing columns are handled per option
    # rather than aborting the entire batch.
    ev_csv_resolved = ev_csv.expanduser().resolve()
    if not ev_csv_resolved.is_file():
        raise FileNotFoundError(f"Input CSV not found: {ev_csv_resolved}")
    available_columns = set(
        pd.read_csv(ev_csv_resolved, sep=";", nrows=0).columns
    )

    all_summaries: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    total = len(options)

    print(f"Loaded {total} option(s) from: {options_file.resolve()}")
    print(
        "Uncontrolled peak baseline: "
        f"ch_refuel={UNCONTROLLED_BASELINE_CH_REFUEL:.1f}, "
        f"file={baseline_ev_csv}, "
        f"charging={uncontrolled_charging_column}, "
    )
    print(f"Result root: {output_root}")
    print("Errors in individual options will be recorded and skipped.")
    print()

    for run_index, option in enumerate(options, start=1):
        option_dir = output_root / f"greedy_results_{option.number}"
        option_dir.mkdir(parents=True, exist_ok=True)

        # Always save the requested source-column combination, including for
        # failed runs.
        pd.DataFrame(
            [
                {"parameter": key, "column": value}
                for key, value in option.source_columns.items()
            ]
        ).to_csv(
            option_dir / "source_columns.csv",
            sep=";",
            index=False,
        )

        print("=" * 80)
        print(f"[{run_index}/{total}] Running option{option.number}")
        print(f"  ch_avail = {option.source_columns['ch_avail']}")
        print(f"  soc_min  = {option.source_columns['soc_min']}")
        print(f"  output   = {option_dir}")
        print("=" * 80)

        try:
            missing_columns = sorted(
                {
                    column
                    for column in option.source_columns.values()
                    if column not in available_columns
                }
            )
            if missing_columns:
                raise ValueError(
                    "Required EV CSV column(s) are missing: "
                    + ", ".join(missing_columns)
                )

            outputs = run_peak_shaving(
                ev_csv=ev_csv,
                uncontrolled_baseline_ev_csv=baseline_ev_csv,
                heat_pump_csv=heat_pump_csv,
                household_csv=household_csv,
                output_dir=option_dir,
                source_columns=option.source_columns,
                uncontrolled_charging_column=uncontrolled_charging_column,
                option_number=option.number,
                vehicle_count=vehicle_count,
                start_discharging_kwh=start_discharging_kwh,
                tolerance_MWh=tolerance_MWh,
                charging_efficiency=charging_efficiency,
                discharging_efficiency=discharging_efficiency,
                charging_parameters_xlsx=charging_parameters_xlsx,
            )

            summary = pd.read_csv(outputs.summary_csv, sep=";")
            summary["run_status"] = "success"
            summary["error_type"] = ""
            summary["error_message"] = ""
            all_summaries.append(summary)

            print(f"option{option.number}: SUCCESS")

        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc).replace("\n", " | ")

            failure = {
                "option": option.number,
                "run_status": "failed",
                "error_type": error_type,
                "error_message": error_message,
                "ch_avail_column": option.source_columns.get("ch_avail", ""),
                "soc_max_column": option.source_columns.get("soc_max", ""),
                "soc_min_column": option.source_columns.get("soc_min", ""),
                "charging_column": option.source_columns.get("charging", ""),
                "consumption_column": option.source_columns.get(
                    "consumption", ""
                ),
                "ch_min_column": option.source_columns.get("ch_min", ""),
                "dem_temp_column": option.source_columns.get("dem_temp", ""),
                "output_dir": str(option_dir),
            }
            failures.append(failure)

            # Keep an error record directly inside greedy_results_N.
            pd.DataFrame([failure]).to_csv(
                option_dir / "greedy_error.csv",
                sep=";",
                index=False,
            )

            print(f"option{option.number}: FAILED")
            print(f"  {error_type}: {error_message}")
            print("  -> continuing with the next option")

        print()

    # Successful-run summary. It is still created when all options fail.
    combined_path = output_root / "greedy_all_options_summary.csv"
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined = combined.sort_values("option").reset_index(drop=True)
        combined.to_csv(
            combined_path,
            sep=";",
            index=False,
            float_format="%.9f",
        )
    else:
        pd.DataFrame(
            columns=[
                "option",
                "run_status",
                "error_type",
                "error_message",
            ]
        ).to_csv(
            combined_path,
            sep=";",
            index=False,
        )

    # One global file collecting all failed runs.
    failures_path = output_root / "greedy_failed_options.csv"
    if failures:
        pd.DataFrame(failures).sort_values("option").to_csv(
            failures_path,
            sep=";",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "option",
                "run_status",
                "error_type",
                "error_message",
            ]
        ).to_csv(
            failures_path,
            sep=";",
            index=False,
        )

    success_count = len(all_summaries)
    failure_count = len(failures)

    print("=" * 80)
    print("Batch finished.")
    print(f"Successful options: {success_count}")
    print(f"Failed options:     {failure_count}")
    print(f"Success summary:    {combined_path}")
    print(f"Failure summary:    {failures_path}")
    print("=" * 80)

    return combined_path

def _parse_option_selection(values: list[int] | None) -> set[int] | None:
    """Parse option selection.
    
    Parameters
    ----------
    values : list[int] | None
        Values used by this operation.
    
    Returns
    -------
    set[int] | None
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    if not values:
        return None
    if any(value <= 0 for value in values):
        raise ValueError("--options values must be positive integers.")
    return set(values)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Result computed as ``parser``."""
    parser = argparse.ArgumentParser(
        description=(
            "Read optionN/SOURCE_COLUMNS combinations from option1.txt and run "
            "the greedy EV optimizer once per option."
        )
    )
    parser.add_argument("--ev-csv", type=Path, required=True)
    parser.add_argument("--heat-pump-csv", type=Path, required=True)
    parser.add_argument("--household-csv", type=Path, required=True)
    parser.add_argument(
        "--uncontrolled-baseline-ev-csv",
        type=Path,
        default=None,
        help=(
            "EV aggregate for the fixed ch_refuel=0.4 uncontrolled baseline. "
            "Default: infer the sibling *_0.4 folder from --ev-csv."
        ),
    )
    parser.add_argument(
        "--options-file",
        type=Path,
        default=DEFAULT_OPTIONS_FILE,
        help=(
            "Text file containing optionN: SOURCE_COLUMNS = {...} blocks. "
            "Default: option1.txt next to this script."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for greedy_results_1, greedy_results_2, ... "
            "Default: directory containing --ev-csv."
        ),
    )
    parser.add_argument(
        "--options",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional subset to run, e.g. --options 1 7 8. "
            "Default: run every option in the file."
        ),
    )
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=None,
        help="Fleet size. Default: infer the unique positive nprofiles value.",
    )
    parser.add_argument(
        "--start-discharging-kwh",
        type=float,
        default=0.0,
        help="Additional minimum SoC per vehicle in kWh.",
    )
    parser.add_argument(
        "--tolerance-mwh",
        type=float,
        default=0.0015,
        help="Binary-search peak tolerance in MWh per one-hour step.",
    )
    parser.add_argument(
        "--charging-efficiency",
        type=float,
        default=None,
        help=(
            "Grid-to-battery charging efficiency. Default: read from "
            "--charging-parameters-xlsx, otherwise 0.95."
        ),
    )
    parser.add_argument(
        "--discharging-efficiency",
        type=float,
        default=None,
        help="Battery-to-grid efficiency. Default: equal to charging efficiency.",
    )
    parser.add_argument(
        "--charging-parameters-xlsx",
        type=Path,
        default=None,
        help=(
            "Optional ekfzladeprofile Input_Ladeprofile.xlsx; reads "
            "charging_efficiency_normal from the Charging_parameter sheet."
        ),
    )
    return parser


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    args = build_parser().parse_args()

    run_all_options(
        options_file=args.options_file,
        ev_csv=args.ev_csv,
        uncontrolled_baseline_ev_csv=args.uncontrolled_baseline_ev_csv,
        heat_pump_csv=args.heat_pump_csv,
        household_csv=args.household_csv,
        output_root=args.output_root,
        vehicle_count=args.vehicle_count,
        start_discharging_kwh=args.start_discharging_kwh,
        tolerance_MWh=args.tolerance_mwh,
        charging_efficiency=args.charging_efficiency,
        discharging_efficiency=args.discharging_efficiency,
        charging_parameters_xlsx=args.charging_parameters_xlsx,
        selected_options=_parse_option_selection(args.options),
    )


if __name__ == "__main__":
    main()
