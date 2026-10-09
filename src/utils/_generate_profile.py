"""Generate profile utilities for the EV/V2G analysis workflow."""

import pandas as pd
import os
from math import ceil
import numpy as np

from pathlib import Path
input_charge = str(Path(__file__).with_name("Input_Ladeprofile.xlsx"))

def read_input_mob_groups():
    """Read input mob groups.
    
    Returns
    -------
    object
        Result computed as ``share_mob_group``."""
    share_mob_group = pd.read_excel(
        input_charge,
        sheet_name='Mobilitätsgruppen',
        engine='openpyxl',
    )
    return share_mob_group

def get_veh_per_mob(ev_stats:pd.DataFrame, 
                    region_name:str,):
    """Return veh per mob.
    
    Parameters
    ----------
    ev_stats : pd.DataFrame
        Ev stats used by this operation.
    region_name : str
        Region name used by this operation.
    
    Returns
    -------
    object
        Result computed as ``ev_stats``."""
    mob_groups_df = read_input_mob_groups()

    def get_mob_group(hh, eink, region_name):
        """Return mob group.
        
        Parameters
        ----------
        hh : object
            Hh used by this operation.
        eink : object
            Eink used by this operation.
        region_name : object
            Region name used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        region_num = ''.join(filter(str.isdigit, region_name))
        hh_num = ''.join(filter(str.isdigit, hh))
        eink_num = ''.join(filter(str.isdigit, eink))
        return int(region_num) + int(hh_num) + int(eink_num), int(region_num)*100 + int(hh_num)*10 + int(eink_num)

    share_region = mob_groups_df.loc[mob_groups_df['R_Gruppe'] == region_name, 'DEU'].sum()
    idxs, veh_counts, share_mob_list, mob_groups = [], [], [], []

    for i, row in ev_stats.iterrows():
        hh, eink = i
        n_vehicles = row['total_ev_count']
        idx, mob_group = get_mob_group(hh, eink, region_name)
        idxs.append(idx)
        mob_groups.append(mob_group)

        if idx in mob_groups_df.index:
            share_mob_group = mob_groups_df.loc[idx, 'DEU'] / share_region
            veh_per_region = ceil((n_vehicles / share_mob_group) * 3) if share_mob_group > 0 else np.nan

        share_mob_list.append(share_mob_group)
        veh_counts.append(veh_per_region)

    ev_stats['mob_group'] = mob_groups
    ev_stats['share_mob_group'] = share_mob_list
    ev_stats['veh_per_region'] = veh_counts

    return ev_stats
