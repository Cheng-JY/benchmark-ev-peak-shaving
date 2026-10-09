"""Batch-generate household and heat-pump profiles with existing builders.

Run this module from the repository root. Every parameter combination is saved
in a separate folder and ``scenario_summary.csv`` is updated after each run.

Example::

    python -m jcheng.scripts.run_household_heat_pump_profiles ^
      --date-input 20260929 --net-name uw_langenselbold ^
      --station-name "ST. Feldstr. 630KVATR" --region-name R4 ^
      --household-ratios 1.0 --hp-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ^
      --ev-ratios 1.0 ^
      --years 2026 --climate-models Kassel^
      --output-root data/data_20260929^
      --seed 43

For climate-specific heat-pump inputs, add::

    --hp-timeseries-pattern \
      "data/hp_inputs/Zeitreihen_WP_{model}_{year}.xlsx"

Custom workbooks must contain ``Zeitreihen`` plus ``GLZ`` or ``GLZ (neu)``.
The bundled defaults are weather/city variants, not aladin/era5/pecd climate
model years. A year changes the profile only when the selected custom workbook
itself represents that year.
"""

from __future__ import annotations

import argparse
import itertools
import re
import traceback
from pathlib import Path
from typing import Iterable

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_generation_dependencies():
    """Load the optional external database workflow only for generation."""
    global HeatPumpProfileBuilder, HouseholdProfileBuilder
    global ScenarioConfig, ScenarioBuilder, ScenarioData, db, ph
    global list_station_names, get_lv_net
    try:
        from jcheng.components.input_data import (
            HeatPumpProfileBuilder,
            HouseholdProfileBuilder,
        )
        from jcheng.components.scenario.config import ScenarioConfig
        from jcheng.components.scenario.scenario_builder import (
            ScenarioBuilder,
            ScenarioData,
        )
        from ecv2g.db_api import db, ph
        from ecv2g.db_api.mongodb import list_station_names, get_lv_net
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Profile generation requires the original ecv2g package and its "
            "configured database, which are not included in this repository. "
            "Install/configure ecv2g, or run coordination with exported CSV inputs."
        ) from exc


DEFAULT_MODELS = (
    "Hamburg", "Kassel", "Muenchen",
)
JCHENG_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HP_WORKBOOKS = {
    "Hamburg": JCHENG_DIR / "Zeitreihen_WP_Update_Hamburg.xlsx",
    "Kassel": JCHENG_DIR / "Zeitreihen_WP_Update_Kassel.xlsx",
    "Muenchen": JCHENG_DIR / "Zeitreihen_WP_Update_Muenchen.xlsx",
}


def _safe_name(value: object) -> str:
    """Safe name.
    
    Parameters
    ----------
    value : object
        Value used by this operation.
    
    Returns
    -------
    str
        Result computed as ``re.sub('[^A-Za-z0-9_.-]+', '_', str(value).strip()).strip('_')``."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")


def _ratio_name(value: float) -> str:
    """Ratio name.
    
    Parameters
    ----------
    value : float
        Value used by this operation.
    
    Returns
    -------
    str
        Result computed as ``f'{value:g}'.replace('.', 'p')``."""
    return f"{value:g}".replace(".", "p")


def _hp_workbook(pattern: str | None, model: str, year: int) -> Path | None:
    """Hp workbook.
    
    Parameters
    ----------
    pattern : str | None
        Pattern used by this operation.
    model : str
        Model used by this operation.
    year : int
        Year used by this operation.
    
    Returns
    -------
    Path | None
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    if pattern is None:
        if model not in DEFAULT_HP_WORKBOOKS:
            raise ValueError(
                f"No default heat-pump workbook is configured for {model!r}. "
                "Use --hp-timeseries-pattern for custom model/year files."
            )
        return DEFAULT_HP_WORKBOOKS[model].resolve()
    try:
        value = pattern.format(model=model, year=year)
    except KeyError as exc:
        raise ValueError(
            "--hp-timeseries-pattern may only use {model} and {year}"
        ) from exc
    return Path(value).expanduser().resolve()

def _check_building_allocation(
    net_name: str,
    station_name: str,
) -> dict[str, object]:
    """Check whether the selected station has building-allocation data."""

    net = get_lv_net(station_name, net_name)
    net_id = ph._get_net_id_from_name(net_name, db)

    # MongoDB queries should receive native Python integers.
    bus_idx = [int(idx) for idx in net.bus.index.tolist()]

    collection = db.building_allocation

    net_count = collection.count_documents({
        "net_id": net_id,
    })

    matching_count = collection.count_documents({
        "net_id": net_id,
        "nearest_bus_idx": {"$in": bus_idx},
    })

    print("\nGrid/building-allocation precheck")
    print(f"  net_name: {net_name!r}")
    print(f"  station_name: {station_name!r}")
    print(f"  net_id: {net_id!r}")
    print(f"  station buses: {len(bus_idx)}")
    print(f"  allocations for complete net: {net_count}")
    print(f"  allocations for selected station: {matching_count}")

    if matching_count == 0:
        try:
            available_stations = pd.DataFrame(
                list_station_names(net_name=net_name)
            )
            station_text = available_stations.to_string(index=False)
        except Exception as exc:
            station_text = (
                "Could not list stations: "
                f"{type(exc).__name__}: {exc}"
            )

        if net_count == 0:
            reason = (
                f"No building allocations exist for net_id={net_id!r}. "
                "The building_allocation collection must first be populated "
                "for this network."
            )
        else:
            reason = (
                f"The network has {net_count} building allocations, but none "
                f"belong to the buses of station {station_name!r}. "
                "Check the network/station combination."
            )

        raise ValueError(
            f"{reason}\n\n"
            f"Available stations for {net_name!r}:\n{station_text}"
        )

    return {
        "net_id": net_id,
        "station_bus_count": len(bus_idx),
        "net_allocation_count": net_count,
        "station_allocation_count": matching_count,
    }

def _metrics(df: pd.DataFrame, column: str) -> dict[str, float | int]:
    """Metrics .
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame used by this operation.
    column : str
        Column used by this operation.
    
    Returns
    -------
    dict[str, float | int]
        Result computed as ``{'steps': len(values), 'peak': float(values.max()), 'energy': float(values.sum() * 0.25)}``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column!r} contains missing/non-numeric values")
    return {
        "steps": len(values),
        "peak": float(values.max()),
        "energy": float(values.sum() * 0.25),
    }


def _scenario_id(
    hh: float,
    hp: float,
    ev: float,
    model: str,
    year: int,
    seed: int,
) -> str:
    """Scenario id.
    
    Parameters
    ----------
    hh : float
        Hh used by this operation.
    hp : float
        Hp used by this operation.
    ev : float
        Ev used by this operation.
    model : str
        Model used by this operation.
    year : int
        Year used by this operation.
    seed : int
        Seed used by this operation.
    
    Returns
    -------
    str
        Computed result for the selected input and execution branch."""
    return (
        f"hh_{_ratio_name(hh)}__hp_{_ratio_name(hp)}"
        f"__ev_{_ratio_name(ev)}"
        f"__{_safe_name(model)}_{year}__seed_{seed}"
    )


def generate_profiles(
    *,
    date_input: str,
    net_name: str,
    station_name: str,
    region_name: str,
    household_ratios: Iterable[float],
    hp_ratios: Iterable[float],
    ev_ratios: Iterable[float],
    climate_models: Iterable[str],
    years: Iterable[int],
    output_root: Path,
    hp_timeseries_pattern: str | None = None,
    cp_double_ev_ratio: float = 0.2,
    seeds: Iterable[int] = (13,),
    time_interval: int = 96 * 7,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """Generate all requested parameter combinations."""
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _check_building_allocation(net_name, station_name)
    combinations = list(itertools.product(
        household_ratios, hp_ratios, ev_ratios, climate_models, years, seeds
    ))
    if not combinations:
        raise ValueError("The scenario grid is empty")

    rows: list[dict[str, object]] = []
    summary_path = output_root / "scenario_summary.csv"
    scenario_cache: dict[tuple[float, float, int], dict[str, object]] = {}

    for number, (hh_ratio, hp_ratio, ev_ratio, model, year, seed) in enumerate(
        combinations, start=1
    ):
        scenario_id = _scenario_id(
            hh_ratio, hp_ratio, ev_ratio, model, year, seed
        )
        scenario_folder = output_root / scenario_id
        workbook = _hp_workbook(hp_timeseries_pattern, model, year)
        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "status": "running",
            "date_input": date_input,
            "net_name": net_name,
            "station_name": station_name,
            "region_name": region_name,
            "household_ratio": hh_ratio,
            "scenario_hp_ratio": hp_ratio,
            "scenario_ev_ratio": ev_ratio,
            "hp_timeseries_variant": model,
            "climate_year": year,
            "seed": seed,
            "hp_timeseries_file": str(workbook) if workbook else "project default",
            "year_applied_by_workbook": (
                hp_timeseries_pattern is not None and "{year}" in hp_timeseries_pattern
            ),
            "scenario_folder": str(scenario_folder),
            "message": "",
        }
        print(f"[{number}/{len(combinations)}] {scenario_id}")

        try:
            if hh_ratio <= 0:
                raise ValueError("household ratios must be greater than zero")
            if not 0 <= hp_ratio <= 1:
                raise ValueError("heat-pump ratios must lie between 0 and 1")
            if not 0 <= ev_ratio <= 1:
                raise ValueError("EV ratios must lie between 0 and 1")
            if workbook is not None and not workbook.is_file():
                raise FileNotFoundError(
                    f"Model-specific heat-pump workbook not found: {workbook}"
                )

            cache_key = (hp_ratio, ev_ratio, seed)
            if cache_key not in scenario_cache:
                cfg = ScenarioConfig(
                    date_input=date_input,
                    net_name=net_name,
                    station_name=station_name,
                    region_name=region_name,
                    seed=seed,
                    scenario_year=year,
                    existing_hp_ratio=0.0,
                    scenario_hp_ratio=hp_ratio,
                    scenario_ev_ratio=ev_ratio,
                    cp_double_ev_ratio=cp_double_ev_ratio,
                )
                scenario_builder = ScenarioBuilder(
                    cfg,
                    output_folder=scenario_folder,
                    include_grid_precheck=False,
                    on_conflict="keep",
                )
                net_pop_data = scenario_builder.build()
                net_building_data = scenario_builder.net_building_data
                scenario_document = scenario_builder.scenario_document
                scenario_cache[cache_key] = {
                    "scenario_hash": cfg.scenario_hash,
                    "net_pop_data": net_pop_data.copy(),
                    "net_building_data": net_building_data.copy(),
                    "scenario_document": scenario_document,
                }
                scenario_data = scenario_builder.scenario_data
            else:
                cached = scenario_cache[cache_key]
                cfg = ScenarioConfig(
                    date_input=date_input,
                    net_name=net_name,
                    station_name=station_name,
                    region_name=region_name,
                    seed=seed,
                    scenario_year=year,
                    existing_hp_ratio=0.0,
                    scenario_hp_ratio=hp_ratio,
                    scenario_ev_ratio=ev_ratio,
                    cp_double_ev_ratio=cp_double_ev_ratio,
                    scenario_hash=cached["scenario_hash"],
                )
                net_pop_data = cached["net_pop_data"].copy()
                net_building_data = cached["net_building_data"].copy()
                scenario_document = cached["scenario_document"]
                scenario_data = ScenarioData(
                    date_input,
                    net_name,
                    station_name,
                    output_folder=scenario_folder,
                )
                net_pop_data.to_csv(
                    scenario_data.folder / "net_pop_data.csv",
                    sep=";",
                    index=False,
                )
                net_building_data.to_csv(
                    scenario_data.folder / "net_building_data.csv",
                    sep=";",
                    index=False,
                )
                cfg.save_yaml(scenario_data.folder / "config.yaml")

            hh_df = HouseholdProfileBuilder(scenario_data, cfg).build(
                net_pop_data,
                household_ratio=hh_ratio,
                profile=scenario_document["hh_profile"],
            )
            hp_df = HeatPumpProfileBuilder(scenario_data, cfg).build(
                net_building_data,
                time_interval=time_interval,
                timeseries_filepath=workbook,
            )

            hh_metrics = _metrics(hh_df, "hh_load (MW)")
            hp_metrics = _metrics(hp_df, "total_all")
            if "building_hh_count" in net_building_data.columns:
                base_hh_count = int(net_building_data["building_hh_count"].sum())
            else:
                base_hh_count = len(net_building_data)
            row.update({
                "status": "success",
                "population_rows": len(net_pop_data),
                "household_count_base": base_hh_count,
                "household_count_generated": max(
                    1, int(round(base_hh_count * hh_ratio))
                ),
                "assigned_heat_pumps": int(net_building_data["has_hp"].sum()),
                "household_steps": hh_metrics["steps"],
                "household_peak_mw": hh_metrics["peak"],
                "household_energy_mwh": hh_metrics["energy"],
                "heat_pump_steps": hp_metrics["steps"],
                "heat_pump_peak_kw": hp_metrics["peak"],
                "heat_pump_energy_kwh": hp_metrics["energy"],
                "household_csv": str(
                    scenario_data.household_dir / "household_load.csv"
                ),
                "heat_pump_csv": str(
                    scenario_data.heat_pump_dir / "heat_pump_profiles.csv"
                ),
            })
        except Exception as exc:
            row["status"] = "failed"
            row["message"] = f"{type(exc).__name__}: {exc}"
            scenario_folder.mkdir(parents=True, exist_ok=True)
            (scenario_folder / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            if fail_fast:
                rows.append(row)
                pd.DataFrame(rows).to_csv(summary_path, sep=";", index=False)
                raise

        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, sep=";", index=False)

    summary = pd.DataFrame(rows)
    succeeded = int((summary["status"] == "success").sum())
    print(f"Finished: {succeeded}/{len(summary)} scenarios succeeded")
    print(f"Summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the selected settings.
    
    Returns
    -------
    argparse.Namespace
        Result computed as ``parser.parse_args()``."""
    parser = argparse.ArgumentParser(
        description="Generate a grid of household and heat-pump profiles"
    )
    parser.add_argument("--date-input", required=True)
    parser.add_argument("--net-name", required=True)
    parser.add_argument("--station-name", required=True)
    parser.add_argument("--region-name", default="R3")
    parser.add_argument("--household-ratios", nargs="+", type=float, default=[1.0])
    parser.add_argument("--hp-ratios", nargs="+", type=float, default=[0.3])
    parser.add_argument(
        "--ev-ratios",
        "--scenario-ev-ratio",
        dest="ev_ratios",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "One or more EV penetration ratios. The old "
            "--scenario-ev-ratio name remains available as an alias."
        ),
    )
    parser.add_argument("--climate-models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--years", nargs="+", type=int, default=[2050])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--cp-double-ev-ratio", type=float, default=0.2)
    parser.add_argument("--time-interval", type=int, default=96 * 7)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--hp-timeseries-pattern",
        default=None,
        help=(
            "Optional workbook pattern containing {model} and/or {year}, e.g. "
            "data/hp/Zeitreihen_WP_{model}_{year}.xlsx. If omitted, every "
            "scenario uses the bundled Hamburg/Kassel/Muenchen workbook "
            "matching --climate-models."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module.
    
    Returns
    -------
    None"""
    args = parse_args()
    _load_generation_dependencies()
    generate_profiles(
        date_input=args.date_input,
        net_name=args.net_name,
        station_name=args.station_name,
        region_name=args.region_name,
        household_ratios=args.household_ratios,
        hp_ratios=args.hp_ratios,
        ev_ratios=args.ev_ratios,
        climate_models=args.climate_models,
        years=args.years,
        output_root=args.output_root,
        hp_timeseries_pattern=args.hp_timeseries_pattern,
        cp_double_ev_ratio=args.cp_double_ev_ratio,
        seeds=args.seeds,
        time_interval=args.time_interval,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
