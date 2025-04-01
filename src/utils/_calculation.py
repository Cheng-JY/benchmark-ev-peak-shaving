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