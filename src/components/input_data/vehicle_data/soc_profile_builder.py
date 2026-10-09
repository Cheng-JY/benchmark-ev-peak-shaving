"""Soc profile builder utilities for the EV/V2G analysis workflow."""

from ....utils import load_parquet

import numpy as np
import os
from joblib import Parallel, delayed
import pandas as pd

class SOCProfileBuilder:

    """Represent socprofile builder and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.ev_dir = scenario_data.ev_dir
        self.cfg = scenario_config

    def generate(
            self,
            ch_refuel_list,
            start_discharging_list,
            total_parts=1,
            njobs=6,
    ):

        """Generate .
        
        Parameters
        ----------
        ch_refuel_list : object
            Ch refuel list used by this operation.
        start_discharging_list : object
            Start discharging list used by this operation.
        total_parts : object
            Total parts used by this operation.
        njobs : object
            Njobs used by this operation."""
        date_input = self.cfg.date_input
        data_folder_name = str(self.ev_dir)

        tasks = []
        for part in range(total_parts):
            for ch_refuel in ch_refuel_list:
                for part in range(total_parts):
                    for sd in start_discharging_list:
                        file_name = f"ch_profiles_{ch_refuel}_DEU_{date_input}_2012_cal2012_{self.cfg.region_name}_part0{part}"
                        tasks.append((file_name, ch_refuel, sd))
    
        Parallel(n_jobs=njobs)(
            delayed(self._run_start_discharging)(
                data_folder_name=data_folder_name,
                date_input=date_input,
                file_name=file_name,
                ch_refuel=ch_refuel,
                start_discharging=sd,
            )
            for (file_name, ch_refuel, sd) in tasks
        )
        

    def _run_start_discharging(
            self,
            data_folder_name:str,
            date_input:str,
            file_name:str,
            ch_refuel:str,
            start_discharging:float,
    ):
        """Run start discharging.
        
        Parameters
        ----------
        data_folder_name : str
            Data folder name used by this operation.
        date_input : str
            Date input used by this operation.
        file_name : str
            File name used by this operation.
        ch_refuel : str
            Plug-in motivation parameter associated with the scenario.
        start_discharging : float
            Start discharging used by this operation."""
        print("loading file:", file_name)
        df = load_parquet(data_folder_name, file_name)

        updated_df = df.copy()
    
        soc_min_all = df.xs('soc_min', level=1, axis=1)
        soc_max_all = df.xs('soc_max', level=1, axis=1)
        ch_avail_all = df.xs('ch_avail', level=1, axis=1)
        mask_p1 = (ch_avail_all != 0)
        updated_soc_min = np.minimum(soc_min_all.where(~mask_p1, np.maximum(soc_min_all, start_discharging)), soc_max_all)

        soc_min_p2 = df.xs('soc_min_plug2', level=1, axis=1)
        soc_max_p2 = df.xs('soc_max_plug2', level=1, axis=1)
        ch_avail_p2 = df.xs('ch_avail_plug2', level=1, axis=1)
        mask2 = (ch_avail_p2 != 0)
        updated_soc_min_p2 = np.minimum(soc_min_p2.where(~mask2, np.maximum(soc_min_p2, start_discharging)), soc_max_p2)

        if not isinstance(updated_df.columns, pd.MultiIndex):
            updated_df.columns = pd.MultiIndex.from_tuples(updated_df.columns)
        colnames = updated_soc_min.columns
        updated_soc_min.columns = pd.MultiIndex.from_product([colnames, ['soc_min']])
        updated_soc_min_p2.columns = pd.MultiIndex.from_product([colnames, ['soc_min_plug2']])

        updated_df.loc[:, updated_soc_min.columns] = updated_soc_min
        updated_df.loc[:, updated_soc_min_p2.columns] = updated_soc_min_p2
        
        output_file = f"ch_profiles_{ch_refuel}_{start_discharging}_DEU_{date_input}_{file_name[-22:]}.parquet"
        updated_df.to_parquet(os.path.join(data_folder_name, output_file))
        print("Saved updated file:", output_file)
            

