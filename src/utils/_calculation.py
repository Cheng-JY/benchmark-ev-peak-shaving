import numpy as np
import pandas as pd
from ._data import load_csv, data_path, get_household_load

def calculate_SoC(SoC_prev, charging, consumption):
    return np.max(SoC_prev + charging - consumption, 0)

def get_updated_load(
        result_df,
        plug_state:str,
        charging_threshold:float,
        household_input:str="case_study_time_series_sum",
        hp_portion:float=1,
):  
    hh_profile = load_csv(data_path(), household_input, sep=";")
    household_load = get_household_load(hh_profile, hh_keyword="hh", hp_portion=hp_portion, hp_keyword="hp")
    load = household_load

    for key in ["uncontrolled", "controlled_wo_v2g", "controlled_v2g"]:
        load += result_df.get(f'charging_update_{plug_state}_{charging_threshold}_{key}')

    return load

def get_influence_length(
    status_load,
    updated_load,
):
    diff_ranges = []
    start = None

    for i in range(len(status_load)):
        if status_load[i] != updated_load[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                diff_ranges.append((start, i-1))
                start = None
    
    if start is not None:
        diff_ranges.append((start, len(status_load)-1))
    
    lengths = [(t[1]-t[0]+1) for t in diff_ranges]
    return lengths, diff_ranges

def get_ch_avail_hours(
        charging_profile:pd.DataFrame,
        plug_state:str="_plug2",
        filter:list=None,        
):
    if filter is not None:
        charging_profile, n_ev = filter_id(charging_profile, filter, "home")
    
    ev_id_list = charging_profile.columns.get_level_values(0).unique()

    n_ev = len(ev_id_list)
    n_length = len(charging_profile.index)

    ch_avail_df = charging_profile.iloc[:, charging_profile.columns.get_level_values(1)==f'ch_avail{plug_state}']
    n_ch_avail = np.count_nonzero(ch_avail_df)
    return n_ch_avail, (n_ch_avail / (n_length*n_ev))* (30*24)

def get_ch_avail_hours_per_ev(
        charging_profile:pd.DataFrame,
        plug_state:str="_plug2",
        filter:list=None
):
    if filter is not None:
        charging_profile, _ = filter_id(charging_profile, filter, "home")

    n_length = len(charging_profile.index)

    ch_avail_df = charging_profile.iloc[:, charging_profile.columns.get_level_values(1)==f'ch_avail{plug_state}']
    n_ch_avail = np.count_nonzero(ch_avail_df, axis=0)
    return (n_ch_avail / n_length) * (30*24)

def get_ch_avail_hours_per_ev_destination(
        charging_profile:pd.DataFrame,
        plug_state:str="_plug2",
        filter:list=None
):
    if filter is not None:
        charging_profile, _ = filter_id(charging_profile, filter, "home")

    n_length = len(charging_profile.index)

    column_mask = charging_profile.columns.get_level_values(1).isin([f'ch_avail{plug_state}', 'destination'])
    ch_avail_df = charging_profile.iloc[:, column_mask]
    ch_avail = ch_avail_df.xs(f'ch_avail{plug_state}',   axis=1, level=1)
    destination = ch_avail_df.xs('destination',  axis=1, level=1)

    mask = (destination == 3) & (ch_avail != 0)

    counts = np.count_nonzero(mask.values, axis=0)

    return (counts / n_length) * (30*24)

def get_ch_avail_hours_per_ev_month(
        charging_profile:pd.DataFrame,
        plug_state:str="_plug2",
        filter:list=None,
        previous_result:dict=None,
):
    if filter is not None:
        charging_profile, _ = filter_id(charging_profile, filter, "home")
    
    ch_avail_df = charging_profile.iloc[:, charging_profile.columns.get_level_values(1)==f'ch_avail{plug_state}']
    result_month = {}
    for i in range(1, 13):
        month_mask = (charging_profile.index.month == i)
        n_ch_avail = np.count_nonzero(ch_avail_df[month_mask], axis=0)
        n_length = len(charging_profile[month_mask].index)
        n_ch_avail_month = (n_ch_avail / n_length) * (30*24)
        if previous_result is not None:
            n_ch_avail_month = np.concatenate((previous_result[i], n_ch_avail_month), axis=None)
        result_month[i] = n_ch_avail_month
    return result_month

def filter_id(profiles, filters: str, key_filter: str) -> pd.DataFrame:
    """
    example filter: ['01-.....-7123-..1111-H..A00S000-1.']
    key composition (meaning of the characters):
     0- 1: profile generator version number
     3- 7: profile id
     9-10: regional type
    11-12: encoded location
    Note: 3-12 form a unique identifier for a profile within a set generated in one run.
    14-15: indicator federal state (within DEU)
    16:    indicator scale group (1-5)
    17:    indicator commuting vehicle (0 - no, 1 - yes)
    18:    indicator whether car is used for travelling (0 - no, 1 - yes)
    19:    indicator market size (0 - 0..7000 km/a, 1 - 7001..14000 km/a, 2 - 14001..100000 km/a)
    21-30: H<available charging power at home in kW>
           A<available charging power at work (Arbeit)>
           S<available charging power elsewhere (Sonstiges)>
    32: indicator propulsion (1 - BEV, 2 - PHEV, 3 - REEV)
    33: indicator vehicle size (1 - small, 2 - medium, 3 - large, 4 - lcv,
                                5 - small, 2nd car in household, 6 - medium, 2nd car)
    Note: wild card in regex is '.' (matches any character at this position).
    """
    if len(filters[0]) < 34:
        raise ValueError("not the right filter length")
    if filters[0][2] != '-' or filters[0][8] != '-' or filters[0][13] != '-' :
        raise ValueError("block size not as required or wrong spacers. needs to be 2-5-4-6-10-2")
    filtered_profile_list = []
    for filter in filters:
        filtered_profile_list += [col for col in profiles.columns.levels[0] if re.match(re.compile(filter), col)]
    profiles_filt = profiles[filtered_profile_list]
    profiles_filt.columns = profiles_filt.columns.remove_unused_levels()
    nprofiles = len(profiles_filt.columns.levels[0])
    print(f'filtered {nprofiles} out of {len(profiles.columns.levels[0])} profiles for {key_filter}: {filters}')
    return profiles_filt, nprofiles

def get_annual_mileage_per_ev(
        charging_profile:pd.DataFrame,
        filter:list=None
):
    if filter is not None:
        charging_profile, _ = filter_id(charging_profile, filter, "home")

    distance_df = charging_profile.iloc[:, charging_profile.columns.get_level_values(1)=='distance']
    annual_mileage = np.sum(distance_df, axis=0)
    return annual_mileage