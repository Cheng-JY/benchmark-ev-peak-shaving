"""Heat pump builder utilities for the EV/V2G analysis workflow."""

import numpy as np
import pandas as pd
from pathlib import Path
from ...utils._heat_pump import (
    read_heatpump_timeseries,
    get_simultaneity_factor,
    adjust_timeseries,
    get_heat_pump_statistics
)

class HeatPumpProfileBuilder:
    """Represent heat pump profile builder and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.heat_pump_dir = scenario_data.heat_pump_dir
        self.cfg = scenario_config
    
    def build(
        self,
        net_pop_data,
        time_interval=96*7,
        timeseries_filepath=None,
        sheet_sim=None,
        output_filename="heat_pump_profiles.csv",
    ):
        """Build and save the aggregated heat-pump profile.

        Parameters
        ----------
        timeseries_filepath:
            Optional model/year-specific replacement for ``Zeitreihen_WP.xlsx``.
            If omitted, the original project workbook is used.
        output_filename:
            CSV filename inside ``scenario_data.heat_pump_dir``.

        The defaults preserve the former API and output layout.
        """
        if not output_filename.lower().endswith(".csv"):
            raise ValueError("output_filename must end with '.csv'")

        heat_stats = get_heat_pump_statistics(net_pop_data)
        print(heat_stats)

        sum_hp = heat_stats["hp_count"].sum()

        # read heatpump timeseries and simultaneity factors
        jcheng_dir = Path(__file__).resolve().parents[2]
        config_filename = jcheng_dir / "utils" / "heat_pump_config.xlsx"
        df_config = pd.read_excel(config_filename, index_col=None)

        filepath = (
            Path(timeseries_filepath).expanduser().resolve()
            if timeseries_filepath is not None
            else jcheng_dir / "utils" / "Zeitreihen_WP.xlsx"
        )
        if not filepath.is_file():
            raise FileNotFoundError(f"Heat-pump timeseries workbook not found: {filepath}")
        sheet_ts = "Zeitreihen"  # sheet name with timeseries
        if sheet_sim is None:
            available_sheets = pd.ExcelFile(filepath).sheet_names
            sheet_sim = "GLZ (neu)" if "GLZ (neu)" in available_sheets else "GLZ"

        df_ts, df_sim = read_heatpump_timeseries(filepath, sheet_ts, sheet_sim)

        # Updated city workbooks use "*_mix" instead of "*_unsaniert".
        fallback_columns = {
            "EZFH_Luft_unsaniert": "EZFH_Luft_mix",
            "EZFH_Wasser_unsaniert": "EZFH_Wasser_mix",
        }
        for idx, row in df_config.iterrows():
            configured = row["col_excel"]
            if configured in df_ts.columns:
                continue
            fallback = fallback_columns.get(configured)
            if fallback in df_ts.columns:
                df_config.at[idx, "col_excel"] = fallback

        # add simultaneity factors to config dataframe
        df_config["sim_factor"] = df_config.apply(lambda r: get_simultaneity_factor(r, df_sim, sum_hp), axis=1)

        # update the hp_dict with simultaneity factors for each type (source * renovated/partly renovated)

        # Adjust timeseries for all types and building types make sum considering number of heatpumps

        adjusted_timeseries = {}
        for _, row in heat_stats.iterrows():
            building_type = row["building_type"]
            is_renovated = row["sanierung"]
            hp_count = row["hp_count"]

            ts_final = adjust_timeseries(
                df_config=df_config,
                df_ts=df_ts,
                building_type=building_type,
                hp_type="air",
                is_renovated=is_renovated,
                heatpump_stat=heat_stats,
                plot=False,
                time_interval=time_interval,
            )

            if ts_final is None:
                raise ValueError(
                    "No heat-pump timeseries could be generated for "
                    f"building_type={building_type!r}, "
                    f"sanierung={is_renovated!r}, "
                    f"hp_type='air', "
                    f"hp_count={hp_count}. "
                    "Check heat_pump_config.xlsx and adjust_timeseries()."
                )

            key = f"{building_type}_air_{is_renovated}"

            adjusted_timeseries[key] = {
                "timeseries": ts_final,
                "hp_count": hp_count,
                "floor_area_m2": row["floor_area_m2"],
                "building_count": row["building_count"],
            }
        
        time_index = pd.date_range(start='2012-01-01', periods=len(df_ts), freq='15min')
        df_out = pd.DataFrame(index=time_index)
        for key, info in adjusted_timeseries.items():
            ts = info["timeseries"]
            hp_count = info["hp_count"]
            df_out[f"{key}_total"] = ts * hp_count  # total load for this type
            df_out[f"{key}_per_unit"] = ts  # per unit load for this type

        df_out["total_all"] = df_out[[col for col in df_out.columns if col.endswith("_total")]].sum(axis=1)
        df_out["total_per_unit"] = df_out[[col for col in df_out.columns if col.endswith("_per_unit")]].sum(axis=1)

        heat_pump_path = self.heat_pump_dir / output_filename
        df_out.to_csv(heat_pump_path, sep=";", float_format="%.6f")

        return df_out
    
    """
    cfg = ScenarioConfig.from_yaml("configs/scenario_r7.yaml")
    sb = ScenarioBuilder(cfg)
    scenario_data = sb.build()

    # Run heat pump builder (a2)
    from src.components.heat_pump.heat_pump_builder import HeatPumpProfileBuilder
    hp_builder = HeatPumpProfileBuilder(scenario_data)
    hp_df = hp_builder.build()
    """
