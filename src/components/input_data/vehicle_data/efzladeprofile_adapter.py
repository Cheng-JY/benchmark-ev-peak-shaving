"""Adapter for ``ch_profiles_agg_*.csv`` files.

The adapter intentionally supports only the current aggregated
``ekfzladeprofile`` schema.  It converts the representative per-vehicle
profile in kW/kWh to the fleet profile in MW/MWh used by
``emob_cold_v2g``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


OPTIMIZATION_INTERVAL = pd.Timedelta(minutes=15)
HOURLY_INTERVAL = pd.Timedelta(hours=1)
FILE_PREFIX = "ch_profiles_agg_"
DEFAULT_CHARGING_EFFICIENCY = 0.95

# 这里可以换不同的组合
SOURCE_COLUMNS = {
    "ch_avail": "ch_avail_cp",
    "soc_max": "soc_max_cp",
    "soc_min": "soc_min_cp",
    "charging": "ch_direct_cp",
    "consumption": "consumption_cp",
    "ch_min": "charging_power_min_kw",
    "dem_temp": "dem_temp_cp",
}


class EFZLadeprofileInterfaceError(ValueError):
    """Raised when an efzladeprofile file does not satisfy the interface."""


@dataclass(frozen=True)
class EVProfileState:
    """Canonical fleet profile consumed by the V2G optimizers.

    ``ch_avail`` and ``ch_min`` are power in MW.  All other numeric time
    series are energy in MWh per 15-minute interval.
    """

    ch_avail: pd.Series
    soc_max: pd.Series
    soc_min: pd.Series
    charging: pd.Series
    consumption: pd.Series
    ch_min: pd.Series
    dem_temp: pd.Series
    source_path: Path
    source_columns: dict[str, str | None]

    def legacy_tuple(
        self,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Return the five values used by the existing optimizer API."""

        return (
            self.ch_avail,
            self.soc_max,
            self.soc_min,
            self.charging,
            self.consumption,
        )


def _read_numeric_column(
    df: pd.DataFrame,
    logical_name: str,
) -> pd.Series:
    """Read numeric column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame used by this operation.
    logical_name : str
        Logical name used by this operation.
    
    Returns
    -------
    pd.Series
        Result computed as ``values.astype(float)``.
    
    Raises
    ------
    EFZLadeprofileInterfaceError
        If an explicit validation or processing check in this operation fails."""
    column = SOURCE_COLUMNS[logical_name]
    if column not in df.columns:
        raise EFZLadeprofileInterfaceError(
            f"Missing required ch_profiles_agg column '{column}'."
        )

    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        locations = values.index[values.isna()][:3].tolist()
        raise EFZLadeprofileInterfaceError(
            f"Column '{column}' contains non-numeric or missing values, "
            f"for example at {locations}."
        )
    return values.astype(float)


def _parse_supported_index(
    df: pd.DataFrame,
    source_path: Path,
) -> tuple[pd.DataFrame, pd.Timedelta]:
    """Parse supported index.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame used by this operation.
    source_path : Path
        Source path used by this operation.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.Timedelta]
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    EFZLadeprofileInterfaceError
        If an explicit validation or processing check in this operation fails."""
    parsed_index = pd.to_datetime(df.index, errors="coerce")
    if parsed_index.isna().any():
        raise EFZLadeprofileInterfaceError(
            f"{source_path.name} has no complete datetime index in its first column."
        )

    result = df.copy()
    result.index = pd.DatetimeIndex(parsed_index)
    if result.index.has_duplicates:
        raise EFZLadeprofileInterfaceError(
            f"{source_path.name} contains duplicate timestamps."
        )
    if not result.index.is_monotonic_increasing:
        raise EFZLadeprofileInterfaceError(
            f"{source_path.name} timestamps are not sorted."
        )

    if len(result.index) <= 1:
        return result, OPTIMIZATION_INTERVAL

    deltas = result.index.to_series().diff().dropna().unique()
    if len(deltas) != 1:
        found = ", ".join(str(pd.Timedelta(delta)) for delta in deltas[:3])
        raise EFZLadeprofileInterfaceError(
            f"{source_path.name} must use a continuous time axis; "
            f"found interval(s): {found}."
        )

    interval = pd.Timedelta(deltas[0])
    if interval not in {OPTIMIZATION_INTERVAL, HOURLY_INTERVAL}:
        raise EFZLadeprofileInterfaceError(
            f"{source_path.name} must use 15-minute or hourly values; "
            f"found interval {interval}."
        )
    return result, interval


def _to_optimization_steps(
    values: pd.Series,
    source_interval: pd.Timedelta,
    *,
    value_kind: str,
) -> np.ndarray:
    """Convert one source series to 15-minute optimizer values.

    Hourly power and SoC values are held constant over the four quarter hours.
    Hourly energy values are distributed equally across the four steps so their
    hourly and annual sums are preserved.
    """

    array = values.to_numpy(dtype=float)
    if source_interval == OPTIMIZATION_INTERVAL:
        return array
    if source_interval != HOURLY_INTERVAL:
        raise EFZLadeprofileInterfaceError(
            f"Unsupported source interval: {source_interval}."
        )

    if value_kind == "energy":
        array = array / 4.0
    elif value_kind not in {"power", "state"}:
        raise ValueError(f"Unknown value kind: {value_kind!r}.")
    return np.repeat(array, 4)


def _validate_non_negative(name: str, values: pd.Series) -> None:
    """Validate non negative.
    
    Parameters
    ----------
    name : str
        Name used by this operation.
    values : pd.Series
        Values used by this operation.
    
    Returns
    -------
    None
    
    Raises
    ------
    EFZLadeprofileInterfaceError
        If an explicit validation or processing check in this operation fails."""
    if (values < -1e-9).any():
        minimum = float(values.min())
        raise EFZLadeprofileInterfaceError(
            f"'{name}' must be non-negative, but its minimum is {minimum:.6g}."
        )


def load_efzladeprofile_csv(
    source_path: str | Path,
    *,
    vehicle_count: int,
    start_discharging_kwh: float = 0.0,
    expected_index: pd.DatetimeIndex | None = None,
) -> EVProfileState:
    """Load and scale one ``ch_profiles_agg_*.csv`` file.

    The exported profile is a per-vehicle mean.  Power columns are converted
    from kW/vehicle to fleet MW and energy columns from kWh/vehicle to fleet
    MWh.  ``start_discharging_kwh`` is applied per vehicle while it is
    connected, mirroring the former ``SOCProfileBuilder`` behavior.
    """

    path = Path(source_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"efzladeprofile input not found: {path}")
    if not path.name.startswith(FILE_PREFIX) or path.suffix.lower() != ".csv":
        raise EFZLadeprofileInterfaceError(
            f"Expected a '{FILE_PREFIX}*.csv' file, got '{path.name}'."
        )
    if int(vehicle_count) != vehicle_count or vehicle_count < 0:
        raise EFZLadeprofileInterfaceError(
            f"vehicle_count must be a non-negative integer, got {vehicle_count!r}."
        )
    if start_discharging_kwh < 0:
        raise EFZLadeprofileInterfaceError(
            "start_discharging_kwh must be non-negative."
        )

    profile = pd.read_csv(path, sep=";", index_col=0)
    profile, source_interval = _parse_supported_index(profile, path)

    ch_avail = _read_numeric_column(profile, "ch_avail")
    soc_max = _read_numeric_column(profile, "soc_max")
    soc_min = _read_numeric_column(profile, "soc_min")
    charging = _read_numeric_column(profile, "charging")
    consumption = _read_numeric_column(profile, "consumption")
    ch_min = _read_numeric_column(profile, "ch_min")
    dem_temp = _read_numeric_column(profile, "dem_temp")

    for name, values in {
        "ch_avail": ch_avail,
        "ch_min": ch_min,
        "soc_max": soc_max,
        "soc_min": soc_min,
        "ch_uncontrolled": charging,
        "consumption": consumption,
        "dem_temp": dem_temp,
    }.items():
        _validate_non_negative(name, values)

    if (ch_min > ch_avail + 1e-9).any():
        raise EFZLadeprofileInterfaceError(
            "Minimum charging power exceeds maximum charging power."
        )
    if (soc_min > soc_max + 1e-6).any():
        raise EFZLadeprofileInterfaceError(
            "Minimum state of charge exceeds maximum state of charge."
        )

    connected = ch_avail > 0
    soc_min = soc_min.where(
        ~connected,
        np.maximum(soc_min, float(start_discharging_kwh)),
    )
    soc_min = np.minimum(soc_min, soc_max)

    values_by_name = {
        "ch_avail": _to_optimization_steps(
            ch_avail, source_interval, value_kind="power"
        ),
        "soc_max": _to_optimization_steps(
            soc_max, source_interval, value_kind="state"
        ),
        "soc_min": _to_optimization_steps(
            pd.Series(soc_min, index=profile.index),
            source_interval,
            value_kind="state",
        ),
        "charging": _to_optimization_steps(
            charging, source_interval, value_kind="energy"
        ),
        "consumption": _to_optimization_steps(
            consumption, source_interval, value_kind="energy"
        ),
        "ch_min": _to_optimization_steps(
            ch_min, source_interval, value_kind="power"
        ),
        "dem_temp": _to_optimization_steps(
            dem_temp, source_interval, value_kind="energy"
        ),
    }

    scale = float(vehicle_count) / 1000.0
    output_length = len(values_by_name["ch_avail"])
    state_index = pd.date_range(
        start=profile.index[0],
        periods=output_length,
        freq=OPTIMIZATION_INTERVAL,
    )
    if expected_index is not None:
        expected_index = pd.DatetimeIndex(expected_index)
        if len(expected_index) != output_length:
            raise EFZLadeprofileInterfaceError(
                f"Profile length after conversion ({output_length}) does not match the feeder-load "
                f"length {len(expected_index)}."
            )
        state_index = expected_index

    def scaled(values: np.ndarray, name: str) -> pd.Series:
        """Scaled .
        
        Parameters
        ----------
        values : np.ndarray
            Values used by this operation.
        name : str
            Name used by this operation.
        
        Returns
        -------
        pd.Series
            Result computed as ``pd.Series(values * scale, index=state_index, name=name)``."""
        return pd.Series(values * scale, index=state_index, name=name)

    return EVProfileState(
        ch_avail=scaled(values_by_name["ch_avail"], "ch_avail_MW"),
        soc_max=scaled(values_by_name["soc_max"], "soc_max_MWh"),
        soc_min=scaled(values_by_name["soc_min"], "soc_min_MWh"),
        charging=scaled(values_by_name["charging"], "ch_uncontrolled_MWh"),
        consumption=scaled(values_by_name["consumption"], "consumption_MWh"),
        ch_min=scaled(values_by_name["ch_min"], "ch_min_MW"),
        dem_temp=scaled(values_by_name["dem_temp"], "dem_temp_MWh"),
        source_path=path.resolve(),
        source_columns=SOURCE_COLUMNS.copy(),
    )


def resolve_efzladeprofile_csv(
    explicit_path: str | Path | None,
    *,
    search_directories: list[str | Path],
    filter_name: str = "all",
    region_name: str = "all",
) -> Path:
    """Resolve an explicit file or one unambiguous aggregated output."""

    directories = [Path(directory) for directory in search_directories]
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [
            directory / candidate for directory in directories
        ] + [candidate]
        for resolved in candidates:
            if resolved.is_file():
                return resolved.resolve()
        checked = ", ".join(str(item) for item in candidates)
        raise FileNotFoundError(
            f"Configured efzladeprofile CSV was not found. Checked: {checked}"
        )

    pattern = re.compile(
        rf"^ch_profiles_agg_{re.escape(filter_name)}_DEU_.*_{re.escape(region_name)}\.csv$"
    )
    matches = sorted(
        {
            path.resolve()
            for directory in directories
            if directory.is_dir()
            for path in directory.glob("ch_profiles_agg_*.csv")
            if pattern.match(path.name)
        }
    )
    if not matches:
        raise FileNotFoundError(
            "No efzladeprofile aggregation CSV found. Set "
            "ScenarioConfig.efzladeprofile_csv explicitly."
        )
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise EFZLadeprofileInterfaceError(
            "More than one efzladeprofile CSV matches the configured filter and "
            f"region; set efzladeprofile_csv explicitly. Matches: {names}"
        )
    return matches[0]