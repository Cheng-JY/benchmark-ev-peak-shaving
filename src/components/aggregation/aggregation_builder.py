"""Aggregation builder utilities for the EV/V2G analysis workflow."""

import pandas as pd
from pathlib import Path
from itertools import product

from ...utils._aggregation import run_aggregation_profile

class AggregationBuilder:

    """Represent aggregation builder and its associated operations."""
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

        self.hh_dir = scenario_data.household_dir
        self.hp_dir = scenario_data.heat_pump_dir
        self.ev_dir = scenario_data.ev_dir

        self.agg_dir = scenario_data.aggregation_dir

    def run(self, ch_refuel_list, start_discharging_list, sheet_name='scenario_1'):

        # read the scenario file / which ev get which contract ...
        # read the combination of the ch_refuel and start_discharging_list
        # do the aggregation with run_aggregation 
        # may be need to do the aggregation multiple time before go into the optimization

        """Run .
        
        Parameters
        ----------
        ch_refuel_list : object
            Ch refuel list used by this operation.
        start_discharging_list : object
            Start discharging list used by this operation.
        sheet_name : object
            Sheet name used by this operation."""
        ev_hh_fp = self.ev_dir / "plug_in_duration_with_hh.csv"
        self.ev_hh = pd.read_csv(ev_hh_fp, sep=";")

        data_folder_ev = self.ev_dir
        data_folder_agg = self.agg_dir

        region_name = self.cfg.region_name
        date_input = self.cfg.date_input
        cal_date_input = 20251125

        keys = self.ev_participation_assignment(
                                participation_filename="assigned_1",
                                participation_sheetname=sheet_name)

        sampled_ev_dict = {
                "uncontrolled": keys[0],
                "controlled_v2g": keys[1]
            }
        
        for key in sampled_ev_dict.keys():
            for thr, sl in product(ch_refuel_list, start_discharging_list):
                # The uncontrolled reference is fixed at ch_refuel=0.4 for
                # every controlled ch_refuel scenario.
                thr_use = 0.4 if "uncontrolled" in key else thr
                data_input_name = f"ch_profiles_{thr_use}_{sl}_DEU_{cal_date_input}_2012_cal2012_{region_name}"
                if sl == 0:
                    data_input_name = f"ch_profiles_{thr_use}_DEU_{cal_date_input}_2012_cal2012_{region_name}"
                data_output_name = f"ch_agg_{thr_use}_{sl}_DEU_{cal_date_input}_2012_cal2012_{region_name}_{sheet_name}"

                sampled_ev_list = sampled_ev_dict.get(key, [])
                run_aggregation_profile(
                    data_folder_ev,
                    data_folder_agg,
                    data_input_name,
                    data_output_name,
                    total_part=1,
                    sampled_ev_list=sampled_ev_list,
                    sampled_ev_group_name=key,
                    hourly_profiles=False,
                )
                    
    def aggregate(self, list_of_aggregation_file_name, output_filename):
        """Aggregate .
        
        Parameters
        ----------
        list_of_aggregation_file_name : object
            List of aggregation file name used by this operation.
        output_filename : object
            Output filename used by this operation."""
        agg_result = None
        n_profile_result = 0
        for filename in list_of_aggregation_file_name:
            data_folder_agg = self.agg_dir
            file_path = data_folder_agg / filename
            profile_agg = pd.read_csv(file_path, sep=";")
            nprofiles = profile_agg.loc[agg_result.index[0], 'nprofiles']
            if agg_result is None:
                agg_result = profile_agg
            else:
                agg_result = agg_result + profile_agg
            n_profile_result += nprofiles
        
        profile_output = f'{output_filename}.csv'
        file_path = data_folder_agg / profile_output
        agg_result.to_csv(file_path, sep=";")

    def ev_participation_assignment(self, 
                                    participation_filename, 
                                    participation_sheetname):
        # 1. alles bei ch_refuel = 0,9 (02.12. 21:00)
        # 2. the same order with the plug-in with hh
        """Ev participation assignment.
        
        Parameters
        ----------
        participation_filename : object
            Participation filename used by this operation.
        participation_sheetname : object
            Participation sheetname used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(keys_uncontrolled, keys_controlled)``."""
        ev_hh_fp = self.ev_dir / "plug_in_duration_with_hh.csv"
        self.ev_hh = pd.read_csv(ev_hh_fp, sep=";")

        participation_fp = self.ev_dir / f"{participation_filename}.xlsx"
        participation_df = pd.read_excel(participation_fp,
                                         sheet_name=participation_sheetname)
        
        merged_df = pd.merge(
            self.ev_hh,
            participation_df,
            left_index=True, 
            right_index=True,
            suffixes=('_plugin', '_part')
        )

        grouped_keys = merged_df.groupby('Take_participation')['key'].apply(list)

        keys_uncontrolled = grouped_keys[0]
        keys_controlled = grouped_keys[1]

        return keys_uncontrolled, keys_controlled

    def participated_household(self,
                               participation_filename, 
                               participation_sheetname):
        """Participated household.
        
        Parameters
        ----------
        participation_filename : object
            Participation filename used by this operation.
        participation_sheetname : object
            Participation sheetname used by this operation.
        
        Returns
        -------
        object
            Result computed as ``participation_counts.reset_index()``."""
        participation_fp = self.ev_dir / f"{participation_filename}.xlsx"
        participation_df = pd.read_excel(participation_fp,
                                         sheet_name=participation_sheetname)
        
        participation_counts = participation_df.groupby('hh_id')['Take_participation'].sum()

        return participation_counts.reset_index()
    

        

            







    

