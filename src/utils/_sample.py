"""Sample utilities for the EV/V2G analysis workflow."""

import os
from copy import deepcopy
import pandas as pd
import numpy as np
from jcheng.utils._calculation import * 
from jcheng.utils._data import *

def get_ev_id_overview(
    data_folder_name:str,
    data_input_filename:str,
):
    """Return ev id overview.
    
    Parameters
    ----------
    data_folder_name : str
        Data folder name used by this operation.
    data_input_filename : str
        Data input filename used by this operation.
    
    Returns
    -------
    object
        Result computed as ``df['key'].tolist()``.
    
    Raises
    ------
    KeyError
        If an explicit validation or processing check in this operation fails."""
    folder = data_path(data_folder_name)
    path = os.path.join(folder, f"{data_input_filename}.csv")
    df = pd.read_csv(path, sep=";")

    if "key" not in df.columns:
        raise KeyError("Column 'key' not found in the input file.")
    return df["key"].tolist()

def sample_ev_profiles(data_folder_name:str,
                       profiles_overview_filename:str,
                       region_group:str,
                       ev_statistics:pd.DataFrame,
                       id_patterns:list[str]|None=None,
                       filter_name:str=""):
    """Sample ev profiles.
    
    Parameters
    ----------
    data_folder_name : str
        Data folder name used by this operation.
    profiles_overview_filename : str
        Profiles overview filename used by this operation.
    region_group : str
        Region group used by this operation.
    ev_statistics : pd.DataFrame
        Ev statistics used by this operation.
    id_patterns : list[str] | None
        Id patterns used by this operation.
    filter_name : str
        Filter name used by this operation.
    
    Returns
    -------
    object
        Result computed as ``df_result``."""
    sampled_filename = f"sampled_ev_profiles_{region_group}_{filter_name}" 
    if os.path.exists(os.path.join(data_path(data_folder_name), f"{sampled_filename}.csv")):
        print(f"{sampled_filename} already exists, skipping sampling.")
        return
    ev_id_list = pd.read_csv(os.path.join(data_path(data_folder_name),
                                         f"{profiles_overview_filename}.csv"))["key"].tolist()
    
    def modify_id_pattern(hh, eink, patterns):
        """Modify id pattern.
        
        Parameters
        ----------
        hh : object
            Hh used by this operation.
        eink : object
            Eink used by this operation.
        patterns : object
            Patterns used by this operation.
        
        Returns
        -------
        object
            Result computed as ``[p.replace('-....-', f'-.{region_num}{hh_num}{eink_num}-') for p in patterns]``."""
        region_num = ''.join(filter(str.isdigit, region_group))
        hh_num = ''.join(filter(str.isdigit, hh))
        eink_num = ''.join(filter(str.isdigit, eink))
        return [p.replace("-....-", f'-.{region_num}{hh_num}{eink_num}-') for p in patterns]
    
    df_result = ev_statistics.copy()
    df_result['id_pattern'] = [modify_id_pattern(hh, eink, id_patterns) for hh, eink in ev_statistics.index]
    df_result['total_ev_count'] = ev_statistics['total_ev_count']

    # Reset index for a clean table
    df_result = df_result.reset_index()[['HHtyp', 'Einktyp', 'total_ev_count', 'id_pattern']]

    print(df_result)

    def sample_evs_with_patterns(id_patterns, ev_id_list, n_sampled_ev):
        """Sample evs with patterns.
        
        Parameters
        ----------
        id_patterns : object
            Id patterns used by this operation.
        ev_id_list : object
            Ev id list used by this operation.
        n_sampled_ev : object
            N sampled ev used by this operation.
        
        Returns
        -------
        object
            Result computed as ``sample_list``.
        
        Raises
        ------
        ValueError
            If an explicit validation or processing check in this operation fails."""
        if id_patterns is not None:
            if len(id_patterns[0]) < 34:
                raise ValueError("not the right filter length")
            if id_patterns[0][2] != '-' or id_patterns[0][8] != '-' or id_patterns[0][13] != '-' :
                raise ValueError("block size not as required or wrong spacers. needs to be 2-5-4-6-10-2")
            filtered_profile_list = []
            for filter in id_patterns:
                filtered_profile_list += [col for col in ev_id_list if re.match(re.compile(filter), col)]
        else:
            filtered_profile_list = ev_id_list

        n_ev = len(filtered_profile_list)
        n_sampled_ev = np.minimum(n_sampled_ev, n_ev)
        sample_list = np.random.choice(filtered_profile_list, n_sampled_ev, replace=False).tolist()
        return sample_list
    
    all_sampled_evs = []
    sampled_counts = [] 
    for _, row in df_result.iterrows():
        n_ev_to_sample = int(row['total_ev_count'])
        id_patterns = row['id_pattern']
        sampled_evs = sample_evs_with_patterns(id_patterns, ev_id_list, n_ev_to_sample)
        all_sampled_evs.extend(sampled_evs)
        sampled_counts.append(len(sampled_evs))

    df_result["sampled_count"] = sampled_counts
    pd.Series(all_sampled_evs, name="vehicle_id").to_csv(os.path.join(data_path(data_folder_name), 
                                                f"{sampled_filename}.csv"), index=False)

    
    return df_result


def get_ev_statistics(net_pop_data: pd.DataFrame) -> pd.DataFrame:
    """Return ev statistics.
    
    Parameters
    ----------
    net_pop_data : pd.DataFrame
        Net pop data used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``(df, grouped)``."""
    df = net_pop_data.copy()
    def group_family(row):
        """Group family.
        
        Parameters
        ----------
        row : object
            Record containing the fields used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        ef20, ef650 = row['ef20'], row['ef650']
        if ef20 + ef650 == 1:
            return 'HH1'
        elif ef20 > 1 and ef650 == 0:
            return 'HH2'
        elif ef20 + ef650 > 1 and ef650 > 0:
            return 'HH3'
        else:
            return 'Unknown'
    
    def group_income(row):
        """Group income.
        
        Parameters
        ----------
        row : object
            Record containing the fields used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        ef707 = row['ef707']
        if ef707 <= 10:
            return 'Eink1'
        elif ef707 <= 16:
            return 'Eink2'
        elif ef707 <= 24:
            return 'Eink3'
        else:
            return 'Unknown'
    
    df['HHtyp'] = df.apply(group_family, axis=1)
    df['Einktyp'] = df.apply(group_income, axis=1)

    grouped = df.groupby(['HHtyp', 'Einktyp']).agg({
        'total_ev_count': 'sum' if 'total_ev_count' in df.columns else lambda x: 0,
        'cp_single_ev_count': 'sum' if 'cp_single_ev_count' in df.columns else lambda x: 0,
        'cp_double_ev_count': 'sum' if 'cp_double_ev_count' in df.columns else lambda x: 0,
    }).rename(columns={'HHtyp': 'HHtyp', 'Einktyp': 'Einktyp', 'total_ev_count': 'total_ev_count', 'cp_single_ev_count': 'cp_single_ev_count', 'cp_double_ev_count': 'cp_double_ev_count'})

    return df, grouped