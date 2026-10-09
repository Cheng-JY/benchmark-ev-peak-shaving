"""Aggregation utilities for the EV/V2G analysis workflow."""

import os
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
from jcheng.utils._calculation import * 
from jcheng.utils._data import *
import ast
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_aggregation_profile(data_folder_ev: Path,
                            data_folder_agg: Path,
                            data_input_filename: str,
                            data_output_filename: str,
                            total_part: int,
                            sampled_ev_list: list,
                            sampled_ev_group_name: str="",
                            hourly_profiles: bool=True):
    """Run aggregation profile.
    
    Parameters
    ----------
    data_folder_ev : Path
        Data folder ev used by this operation.
    data_folder_agg : Path
        Data folder agg used by this operation.
    data_input_filename : str
        Data input filename used by this operation.
    data_output_filename : str
        Data output filename used by this operation.
    total_part : int
        Total part used by this operation.
    sampled_ev_list : list
        Sampled ev list used by this operation.
    sampled_ev_group_name : str
        Sampled ev group name used by this operation.
    hourly_profiles : bool
        Hourly profiles used by this operation."""
    agg_result = None
    for part in range(total_part):
        filename = f"{data_input_filename}_part0{part}.parquet"
        fp = data_folder_ev / filename
        profile_df = pd.read_parquet(fp)
        profile_agg = aggregate_profile(profile_df,
                                    sampled_ev_list,
                                    hourly_profiles)
        if agg_result is None:
            agg_result = profile_agg
        else:
            agg_result = agg_result + profile_agg            
        
    nprofiles = len(sampled_ev_list)
    agg_result.loc[agg_result.index[0], 'nprofiles'] = nprofiles
        
    profile_output = f'{data_output_filename}_{sampled_ev_group_name}_{int(nprofiles)}.csv'
    file_path = data_folder_agg / profile_output
    agg_result.to_csv(file_path, sep=";")

    logger.info(f'aggregated profiles saved to {profile_output}.csv')

def aggregate_profile(
        profile_df:pd.DataFrame,
        sample_ev_id_list:list[str],
        hourly_profiles:bool=True,    
):  
    """Aggregate profile.
    
    Parameters
    ----------
    profile_df : pd.DataFrame
        Profile df used by this operation.
    sample_ev_id_list : list[str]
        Sample ev id list used by this operation.
    hourly_profiles : bool
        Hourly profiles used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch."""
    if not isinstance(profile_df.index, pd.DatetimeIndex):
        # assume index is minutes
        profile_df = profile_df.copy()
        profile_df.index = pd.to_timedelta(profile_df.index, unit="m") + pd.DatetimeIndex('2012-01-01')

    if not isinstance(profile_df.columns, pd.MultiIndex):
        tuples = [ast.literal_eval(col_str) for col_str in profile_df.columns]
        mi = pd.MultiIndex.from_tuples(tuples, names=['ev_id', 'variable'])
        profile_df.columns = mi
    existing_ids = profile_df.columns.get_level_values(0).unique()
    valid_ids = [ev for ev in sample_ev_id_list if ev in existing_ids]
    if valid_ids == []:
        logger.warning("No valid EV IDs found in the profile DataFrame.")
        return pd.DataFrame()
    idx = pd.IndexSlice
    df_sub = profile_df.loc[:, idx[valid_ids, :]]
    
    profile_df_agg = df_sub.T.groupby(level=1).sum().T
    profile_df_agg['nprofiles'] = np.nan
    profile_df_agg = profile_df_agg / 1000 # convert to MW
    profile_df_agg.at[profile_df_agg.index[0], 'nprofiles'] = len(df_sub.columns.get_level_values(0).unique())

    columns_to_drop = [
        'destination', 'ch_avail_travel', 'SOCmax_travel',
        'dem_temp_travel', 'consumption_travel'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in profile_df_agg.columns]
    profile_df_agg = profile_df_agg.drop(columns=columns_to_drop)

    profile_df_agg = profile_df_agg.rename(columns={
    "Strecke": "distance", "Power_Availability": "ch_avail", 
    "SOC_refuel": "SOCmax_plug2", "charging_direct": "ch_direct","charging_refuel": "ch_refuel",
    "demand_temperature": "dem_temp"})

    if hourly_profiles:
        aggregation_methods = {
            'ch_avail': 'mean',
            'ch_avail_plug2': 'mean',
            'soc_min': 'first',
            'soc_max': 'first',
            'soc_max_plug2': 'first',
            'soc_min_plug2': 'first',
            'soc_refuel': 'first',
            'soc_max_travel': 'first',
            'distance': 'sum',
            'ch_direct': 'sum',
            'ch_refuel': 'sum',
            'ch_travel': 'sum',
            'consumption': 'sum',
            'dem_temp': 'sum',
            'max_el_distance': 'sum',
            'nprofiles': 'first',
        }

        # Filter the aggregation methods to only include existing columns
        existing_agg_methods = {col: method for col, method in aggregation_methods.items() 
                                if col in profile_df_agg.columns}

        # Perform the resampling and aggregation
        profile_df_agg = profile_df_agg.resample('1h').agg(existing_agg_methods)

    return profile_df_agg

def find_extra_ev_rows(file_1127_path, file_1203_path):
    # 1. 读取两个 CSV 文件，指定分隔符为分号
    """Find extra ev rows.
    
    Parameters
    ----------
    file_1127_path : object
        File 1127 path used by this operation.
    file_1203_path : object
        File 1203 path used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch."""
    df_1127 = pd.read_csv(file_1127_path, delimiter=';')
    df_1203 = pd.read_csv(file_1203_path, delimiter=';')

    # 2. 计算每个 hh_id 在两个文件中的出现次数 (EV数量)
    counts_1127 = df_1127['hh_id'].value_counts()
    counts_1203 = df_1203['hh_id'].value_counts()

    print(counts_1127 - counts_1203)

    extra_rows_list = []

    # 3. 遍历 1127 中的每个 hh_id，找出数量比 1203 多的情况
    for hh_id in counts_1127.index:
        count_in_1127 = counts_1127[hh_id]
        # 获取该 id 在 1203 中的数量，如果没有则为 0
        count_in_1203 = counts_1203.get(hh_id, 0)

        # 如果 1127 中的数量更多，则提取多出来的行
        if count_in_1127 > count_in_1203:
            diff = count_in_1127 - count_in_1203
            
            
            # 获取该 hh_id 在 1127 中的所有行
            rows = df_1127[df_1127['hh_id'] == hh_id]
            
            # 截取最后 diff 行作为“多余”的行
            # (注意：这里假设多出来的行是追加在后面的；如果没有特定顺序，通常取最后几行作为差集)
            extras = rows.iloc[-diff:]
            extra_rows_list.append(extras)

    # 4. 如果找到了多余的行，合并并保存
    if extra_rows_list:
        result_df = pd.concat(extra_rows_list)
        print(f"找到了 {len(result_df)} 行多余的数据。")
        print("涉及的 hh_id 及其多余数量：")
        print(result_df['hh_id'].value_counts())
        
        return result_df
    else:
        print("未发现 1127 文件中有基于 hh_id 的多余行。")
        return pd.DataFrame()
