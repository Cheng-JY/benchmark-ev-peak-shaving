import numpy as np
import pandas as pd

import os
from copy import deepcopy
import logging
from _calculation import * 
from _data import data_path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_aggregation_profile(data_folder_name:str, 
                            date_input:str, 
                            region_group:str,
                            charging_profiles:pd.DataFrame,
                            n_profiles:int, 
                            controlled_portion:dict[str, float],
                            profile_input:str="ch_profiles",
                            profile_output:str="ch_profiles_agg",
                            hourly_profiles:bool=True,
                            sampled_filename:str|None=None,
                            random_seed:int=42,
                            is_standard:bool=True):
    """
    Aggregate charging profiles for a given region and controlled portion distribution.

    This function loads charging profiles from a given folder, aggregates them according to the controlled portion,
    and saves the aggregated profiles as CSV and Parquet files.

    Parameters
    ----------
    data_folder_name : str
        Path to the folder containing the data files.
    date_input : str
        The input date in "YYYYMMDD" format used to identify the input file.
    region_group : str
        The name of the region group for which the aggregation is performed.
    n_profiles : int
        Number of profiles to aggregate. If the available profiles are fewer, it uses the maximum available profiles.
    controlled_portion : dict[str, float]
        A dictionary defining the portion of profiles for each controlled category (e.g., uncontrolled, controlled_wo_v2g, controlled_v2g, etc.).
        The sum of the values in this dictionary should ideally equal 1.
    profile_input : str, optional
        Prefix for the input profile filename, by default "ch_profiles".
    profile_output : str, optional
        Prefix for the output profile filename, by default "ch_profiles_agg_".

    Returns
    -------
    None
        Saves aggregated profiles as CSV and Parquet files in the specified folder.
    """
    profile, n_ev = deepcopy(charging_profiles), len(charging_profiles.columns.levels[0])

    if n_ev == 0:
        logger.warning(f"No profiles found. Generate more profiles.")
        return

    # logger.info(f'loaded region: {region_group} with {n_ev} profiles')

    n_profiles = np.minimum(n_profiles, n_ev)

    n_prev = 0
    ev_id_list = profile.columns.get_level_values(0).unique()
    np.random.seed(random_seed)

    if os.path.exists(os.path.join(data_path(data_folder_name), 
                                                  f"{sampled_filename}.csv")):
        sampled_elements = pd.read_csv(os.path.join(data_path(data_folder_name), 
                                                  f"{sampled_filename}.csv")).squeeze().tolist()
        n_sampled_elements = len(sampled_elements)
        remaining_elements = list(set(ev_id_list) - set(sampled_elements))
        
        n_new_sampled = n_profiles - n_sampled_elements
        if n_new_sampled > 0:
            new_sampled_elements = np.random.choice(remaining_elements, n_new_sampled, replace=False).tolist()
            sampled_elements += new_sampled_elements
    else:
        sampled_elements = np.random.choice(ev_id_list, n_profiles, replace=False)
    
    sampled_index = pd.Index(sampled_elements)
    sampled_filename = "sampled_ev_profiles" if sampled_filename is None else sampled_filename
    sampled_index.to_series().to_csv(os.path.join(data_path(data_folder_name), 
                                                    f"{sampled_filename}.csv"), index=False)
    
    for key in controlled_portion.keys():
        if controlled_portion.get(key) <= 1:
            n_ev_part = int(n_profiles * controlled_portion.get(key))
        else:
            n_ev_part = int(controlled_portion.get(key))

        if n_ev_part < 1e-8:
            A = profile[ev_id_list[0]]
            controll_profile_agg = pd.DataFrame(0, index=A.index, columns=A.columns)
        else:
            controll_profile = profile[sampled_index[n_prev:n_prev+n_ev_part]]
            controll_profile_agg = controll_profile.T.groupby(level=1).sum().T
            controll_profile_agg = controll_profile_agg / 1000 # convert to MW
            n_prev += n_ev_part
        
        if is_standard:
            controll_profile_agg['nprofiles'] = np.nan
            controll_profile_agg.at[controll_profile_agg.index[0], 'nprofiles'] = n_ev_part
            
            if hourly_profiles:
                controll_profile_agg = controll_profile_agg.resample('h').agg({
                        'ch_avail': 'mean', 'ch_avail_plug2': 'mean', 
                        'SOCmax': 'first', 'SOCmin': 'first', 'SOCmax_plug2': 'first', 'SOCmin_plug2': 'first',
                        'distance': 'sum','ch_direct': 'sum', 'ch_refuel': 'sum', 'ch_travel': 'sum', 
                        'consumption': 'sum', 'dem_temp': 'sum', 'max_el_distance': 'sum',
                        })
        else:
            controll_profile_agg.rename(columns={'Power_Availability': 'ch_avail'}, inplace=True)
            controll_profile_agg.rename(columns={'charging_direct': 'ch_direct'}, inplace=True)
            controll_profile_agg.rename(columns={'charging_refuel': 'ch_refuel'}, inplace=True)
            controll_profile_agg.rename(columns={'demand_temperature': 'dem_temp'}, inplace=True)

            
        controll_profile_agg.to_csv(os.path.join
                                    (data_path(data_folder_name), profile_output + '_' + key + '.csv'), sep=";")
        controll_profile_agg.to_parquet(os.path.join
                                        (data_path(data_folder_name), profile_output + '_' + key + '.parquet'))
        logger.info(f'aggregated profiles saved to {profile_output + key}.csv in {data_folder_name}')

def aggregate_household(
        data_folder_name:str|None=None,
        profile_input:str="case_study_time_series",
        profile_output:str="case_study_time_series_sum",
        hourly_profiles:bool=True):
    
    input_file = os.path.join(data_path(data_folder_name), profile_input + '.csv')
    
    profile_df = pd.read_csv(input_file, sep=",", header=[0,1], parse_dates=[0], index_col=0)
    
    df_filt, nprofiles = deepcopy(profile_df), len(profile_df.columns.levels[0])

    df_filt = df_filt.apply(pd.to_numeric, errors='coerce')

    profile_df_agg = df_filt.T.groupby(level=1).sum().T
    if not isinstance(profile_df_agg.index, pd.DatetimeIndex):
        profile_df_agg.index = pd.to_datetime(profile_df_agg.index, format="%d.%m.%Y %H:%M")

    profile_df_agg['nprofiles'] = np.nan
    profile_df_agg[profile_df_agg.index[0], 'nprofiles'] = nprofiles
    
    if hourly_profiles:
        profile_df_agg = profile_df_agg.resample('h').agg({
            'PV and price optimized': 'sum',  'PV generation': 'sum', 'PV optimized': 'sum',
            'direct': 'sum', 'household load': 'sum', 'price optimized': 'mean'
        })
    
    profile_df_agg.to_csv(os.path.join(
        data_path(data_folder_name), profile_output + '.csv'), sep=";")
    profile_df_agg.to_parquet(os.path.join(
        data_path(data_folder_name), profile_output + '.parquet'))
    
    logger.info(f'aggregated profiles saved to {profile_output}.csv in {data_folder_name}')

def aggregate_profile(
        charging_profiles:pd.DataFrame,
        profile_output:str="ch_profiles_agg",
        hourly_profiles:bool=True,      
):  
    profile_agg = charging_profiles.T.groupby(level=1).sum().T
    profile_agg = profile_agg / 1000 # convert to MW

    if hourly_profiles:
        profile_agg = profile_agg.resample('h').agg({
                'ch_avail': 'mean', 'ch_avail_plug2': 'mean', 
                'SOCmax': 'first', 'SOCmin': 'first', 'SOCmax_plug2': 'first', 'SOCmin_plug2': 'first',
                'distance': 'sum','ch_direct': 'sum', 'ch_refuel': 'sum', 'ch_travel': 'sum', 
                'consumption': 'sum', 'dem_temp': 'sum', 'max_el_distance': 'sum',
                })
            
    profile_agg.to_csv(profile_output + '_' + '.csv', sep=";")
    profile_agg.to_parquet(profile_output + '_' + '.parquet')
    logger.info(f'aggregated profiles saved to {profile_output}.csv')


if __name__ == "__main__":
    date_input = '20241127'
    data_folder_name = f'data_{date_input}'

    controlled_portion = {
        "uncontrolled": 0,
        "controlled_wo_v2g": 0,
        "controlled_v2g": 1,
    }
    n_ev_profiles = 100
    filename = "ch_profiles_DEU_20241122_2012_cal2012_R2"
    refuel=""
    
    run_aggregation_profile(data_folder_name, date_input, 'R2', n_ev_profiles, controlled_portion, hourly_profiles=False, filename_input=filename, refuel=refuel)


