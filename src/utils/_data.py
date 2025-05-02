import numpy as np
import pandas as pd
import os
import datetime
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

cols_mapping = {
    'plug1': ('ch_avail', 'soc_max', 'soc_min', 'ch_direct', 'consumption'),
    'plug2': ('ch_avail_plug2', 'soc_max', 'soc_min', 'ch_refuel', 'consumption'),
}

def get_state(profile: pd.DataFrame, plug_state='plug1', index=None):
    cols = cols_mapping[plug_state]
    if index is None:
        return (profile[cols[0]], profile[cols[1]], 
                profile[cols[2]], profile[cols[3]], 
                profile[cols[4]])
    else:
        return (profile[cols[0]][:index], profile[cols[1]][:index], 
                profile[cols[2]][:index], profile[cols[3]][:index], 
                profile[cols[4]][:index])

def load_parquet(path: str, file_name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(path, f'{file_name}.parquet'))

def load_csv(path:str, file_name:str, sep=";") -> pd.DataFrame:
    return pd.read_csv(os.path.join(path, f'{file_name}.csv'), sep=sep)

def load_result_df(
    result_df, 
    plug_state: str,
    charging_threshold: float,
    data_folder_name: str,
    profile_output: str='ch_update',
    ):

    result_df.to_csv(os.path.join(data_path(data_folder_name), f'{profile_output}_{plug_state}_{charging_threshold}.csv'), sep=";")
    result_df.to_parquet(os.path.join(data_path(data_folder_name), f'{profile_output}_{plug_state}_{charging_threshold}.parquet'))

def ch_profile_filename(
        profile_type:str,
        mob_group: str|None = None,
        date_: str|None = None,
        country:str|None = None,
        weather_year: int|None = None,
        calendar_year: int|None = None
        ) -> str:
    """
    Example
    -------
    >>> ch_profile_filename(
    ...     "ch_avail", "DEU", "20240314", "311", weather_year=2012, calender_year=2013)
    ch_avail_DEU_20240314_2012_cal2013_311
    """
    date_ = date_ if date_ is not None else datetime.date.today().strftime("%Y%m%d")
    country = country if country is not None else "DEU"
    weather_year = weather_year if weather_year is not None else "2012"
    calendar_year = calendar_year if calendar_year is not None else "2012"
    mob_group_str = "" if mob_group is None else f"_{mob_group}"
    return f"{profile_type}_{country}_{date_}_{weather_year}_cal{calendar_year}{mob_group_str}"

_ecv2g_path = os.path.dirname(os.path.realpath(__file__))
print(_ecv2g_path)
ecv2g_path = os.path.abspath(os.path.join(_ecv2g_path,"..", ".."))
print(ecv2g_path)

def data_path(data_folder_name:str|None=None):
    data_folder = os.path.join(ecv2g_path, 'data')
    if data_folder_name is not None:
        return os.path.join(data_folder, data_folder_name)
    else:
        return data_folder
    
def get_household_load(hh_profile:pd.DataFrame,
                    hh_keyword="hh", 
                    year="2012", 
                    hp_portion:float=1, 
                    hp_keyword="hp"):
    if not isinstance(hh_profile.index, pd.DatetimeIndex):
        start_date = f"{year}-01-01"
        hh_profile.index = pd.date_range(start=start_date, periods=len(hh_profile), freq='15T')
    return (hh_profile[hh_keyword] + hh_profile[hp_keyword] * hp_portion) * 0.25


