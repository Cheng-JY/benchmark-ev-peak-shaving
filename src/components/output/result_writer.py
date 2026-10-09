"""Result writer utilities for the EV/V2G analysis workflow."""

import pandas as pd
import numpy as np
import os
from .....db_api.mongodb import (
    write_ev_uncontrolled_timeseries_to_db,
    write_ev_v2g_timeseries_to_db,
    write_hh_timeseries_to_db,
    write_hp_timeseries_to_db,
)
from ...utils import get_state

class ResultWriter:

    """Represent result writer and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.cfg = scenario_config
        self.scenario_hash = scenario_config.scenario_hash
        self.opt_dir = scenario_data.optimization_dir

    def write_hh_load(self):
        """Write hh load."""
        hh_fp = self.scenario_data.household_dir / "household_load.csv"

        df_hh = pd.read_csv(hh_fp, 
                            sep=";", 
                            index_col=0,
        )


        hh_mw_array = df_hh["hh_load (MW)"].to_numpy()

        scenario_hash = self.scenario_hash

        write_hh_timeseries_to_db(scenario_hash=scenario_hash,
                                  timeseries_mw=hh_mw_array,
                                  bus_idx=[])
    
    def write_hp_load(self):
        """Write hp load."""
        hp_fp = self.scenario_data.heat_pump_dir / "heat_pump_profiles.csv"

        df_hp = pd.read_csv(hp_fp, 
                        sep=";", 
                        index_col=0,
                        )
        
        df_hp["hp_MW"] = df_hp["total_all"] / 1000

        hp_mw_array = df_hp["hp_MW"].to_numpy()

        scenario_hash = self.scenario_hash

        write_hp_timeseries_to_db(scenario_hash=scenario_hash,
                                  timeseries_mw=hp_mw_array,
                                  bus_idx=[])
    
    def write_ev_uncontrolled(self,
                              scenario_name,
                              ev_num,
                              v2g_ratio):
        """Write ev uncontrolled.
        
        Parameters
        ----------
        scenario_name : object
            Scenario name used by this operation.
        ev_num : object
            Ev num used by this operation.
        v2g_ratio : object
            V2g ratio used by this operation."""
        scenario_use = scenario_name[:-5]
        cal_data_input = "20251125"
        filename = f"ch_agg_0.4_0_DEU_" + \
                   f"{cal_data_input}_2012_cal2012_" + \
                   f"{self.cfg.region_name}_{scenario_use}_uncontrolled_{ev_num}.csv"

        ev_fp = self.scenario_data.aggregation_dir / filename
        plug_state = "plug2"
        aggregated_profile = pd.read_csv(ev_fp, sep=";")
        index = 365 * 24 * 4
        ch_avail, soc_max, soc_min, charging_status_quo, consumption = get_state(
            aggregated_profile, plug_state=plug_state, index=index)
        
        charging_array = charging_status_quo.to_numpy()

        write_ev_uncontrolled_timeseries_to_db(
            scenario_hash=self.scenario_hash,
            timeseries_mw=charging_array,
            bus_idx=[],
            ev_scenario=scenario_name,
            v2g_ratio=v2g_ratio,
        )
        
    def write_ev_controlled(self,
                            scenario_name,
                            start_discharging,
                            ev_connected_hours,
                            ev_min_range_km,
                            ev_num,
                            v2g_ratio,
                            participate_hh_ids,
                            yearly_flex_cost_euro):
        """Write ev controlled.
        
        Parameters
        ----------
        scenario_name : object
            Scenario name used by this operation.
        start_discharging : object
            Start discharging used by this operation.
        ev_connected_hours : object
            Ev connected hours used by this operation.
        ev_min_range_km : object
            Ev min range km used by this operation.
        ev_num : object
            Ev num used by this operation.
        v2g_ratio : object
            V2g ratio used by this operation.
        participate_hh_ids : object
            Participate hh ids used by this operation.
        yearly_flex_cost_euro : object
            Yearly flex cost euro used by this operation."""
        scenario_use = scenario_name[:-5]
        for fn in os.listdir(self.opt_dir):
            pattern = f"result_{scenario_use}_{start_discharging}"
            check = fn.startswith(pattern)
            if check:
                fp = self.opt_dir / fn
                result_df = pd.read_csv(fp, sep=";")
                result_df["charging_updated_MW"] = result_df[f"charging_updated_{scenario_use}_controlled_v2g"] * 4
        
                charging_mw_array_after = result_df["charging_updated_MW"].to_numpy()

                write_ev_v2g_timeseries_to_db(
                    scenario_hash=self.scenario_hash,
                    timeseries_mw=charging_mw_array_after,
                    bus_idx=[],
                    num_evs=ev_num,
                    v2g_ratio=v2g_ratio,
                    v2g_scenario=scenario_name,
                    yearly_flex_cost_euro=yearly_flex_cost_euro,
                    ev_connected_hours=ev_connected_hours,
                    ev_min_range_km=ev_min_range_km,
                    v2g_hh_ids=participate_hh_ids,
                )
                
                result_df["charging_status_MW"] = result_df[f"charging_status_{scenario_use}_controlled_v2g"] * 4
        
                charging_mw_array_before = result_df["charging_status_MW"].to_numpy()
                
                write_ev_v2g_timeseries_to_db(
                    scenario_hash=self.scenario_hash,
                    timeseries_mw=charging_mw_array_before,
                    bus_idx=[],
                    num_evs=ev_num,
                    v2g_ratio=v2g_ratio,
                    v2g_scenario=scenario_name,
                    yearly_flex_cost_euro=yearly_flex_cost_euro,
                    ev_connected_hours=ev_connected_hours,
                    ev_min_range_km=ev_min_range_km,
                    v2g_hh_ids=participate_hh_ids,
                )
                
                break
        


