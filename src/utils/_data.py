"""Data utilities for the EV/V2G analysis workflow."""

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
    'plug2': ('ch_avail_plug2', 'soc_max_plug2', 'soc_min_plug2', 'ch_refuel', 'consumption'),
}

def get_state(profile: pd.DataFrame, plug_state='plug1', index=None):
    """Return state.
    
    Parameters
    ----------
    profile : pd.DataFrame
        Profile used by this operation.
    plug_state : object
        Plug state used by this operation.
    index : object
        Index used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch."""
    cols = cols_mapping[plug_state]
    if index is None:
        return (profile[cols[0]], profile[cols[1]], 
                profile[cols[2]], profile[cols[3]], 
                profile[cols[4]])
    else:
        return (profile[cols[0]][:index], profile[cols[1]][:index], 
                profile[cols[2]][:index], profile[cols[3]][:index], 
                profile[cols[4]][:index])

def load_parquet(path: str, file_name: str, **kwargs) -> pd.DataFrame:
    """Load parquet.
    
    Parameters
    ----------
    path : str
        File or directory path used by this operation.
    file_name : str
        File name used by this operation.
    kwargs : object
        Kwargs used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``pd.read_parquet(os.path.join(path, f'{file_name}.parquet'), **kwargs)``."""
    return pd.read_parquet(os.path.join(path, f'{file_name}.parquet'), **kwargs)

def load_csv(path:str, file_name:str, sep=";", **kwargs) -> pd.DataFrame:
    """Load csv.
    
    Parameters
    ----------
    path : str
        File or directory path used by this operation.
    file_name : str
        File name used by this operation.
    sep : object
        Sep used by this operation.
    kwargs : object
        Kwargs used by this operation.
    
    Returns
    -------
    pd.DataFrame
        Result computed as ``pd.read_csv(os.path.join(path, f'{file_name}.csv'), sep=sep, **kwargs)``."""
    return pd.read_csv(os.path.join(path, f'{file_name}.csv'), sep=sep, **kwargs)

def load_result_df(
    result_df, 
    plug_state: str,
    charging_threshold: float,
    data_folder_name: str,
    profile_output: str='ch_update',
    ):

    """Load result df.
    
    Parameters
    ----------
    result_df : object
        Result df used by this operation.
    plug_state : str
        Plug state used by this operation.
    charging_threshold : float
        Charging threshold used by this operation.
    data_folder_name : str
        Data folder name used by this operation.
    profile_output : str
        Profile output used by this operation."""
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
ecv2g_path = os.path.abspath(os.path.join(_ecv2g_path,"..", ".."))

def data_path(data_folder_name:str|None=None):
    """Data path.
    
    Parameters
    ----------
    data_folder_name : str | None
        Data folder name used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch."""
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
    """Return household load.
    
    Parameters
    ----------
    hh_profile : pd.DataFrame
        Hh profile used by this operation.
    hh_keyword : object
        Hh keyword used by this operation.
    year : object
        Year used by this operation.
    hp_portion : float
        Hp portion used by this operation.
    hp_keyword : object
        Hp keyword used by this operation.
    
    Returns
    -------
    object
        Result computed as ``(hh_profile[hh_keyword] + hh_profile[hp_keyword] * hp_portion) * 0.25``."""
    if not isinstance(hh_profile.index, pd.DatetimeIndex):
        start_date = f"{year}-01-01"
        hh_profile.index = pd.date_range(start=start_date, periods=len(hh_profile), freq='15T')
    return (hh_profile[hh_keyword] + hh_profile[hp_keyword] * hp_portion) * 0.25

def get_combined_load(data_folder_name:str,
                    net_name:str,
                    station_name:str):
    """Return combined load.
    
    Parameters
    ----------
    data_folder_name : str
        Data folder name used by this operation.
    net_name : str
        Net name used by this operation.
    station_name : str
        Station name used by this operation.
    
    Returns
    -------
    object
        Result computed as ``df_combined['total_MWh']``."""
    household_path = os.path.join(data_path(data_folder_name), 'household_load.csv')
    heatpump_path = os.path.join(data_path(data_folder_name), 'heat_pump.csv')

    df_hp = pd.read_csv(heatpump_path, 
                        sep=";", 
                        index_col=0,
                        )
    df_hp.index = pd.to_datetime(df_hp.index)

    df_hp["hp_MWh"] = df_hp["total_all"] * 0.25/1000

    df_hh = pd.read_csv(household_path, 
                        sep=";", 
                        index_col=0,
    )
    df_hh.index = pd.to_datetime(df_hh.index)
    df_hh["hh_MWh"] = df_hh[f"{station_name}"] * 0.25

    df_combined = pd.DataFrame(index=pd.date_range(
        start='2012-01-01', periods=len(df_hh), freq='15min'))
    df_combined["hp_MWh"] = df_hp["hp_MWh"].values
    df_combined["hh_MWh"] = df_hh["hh_MWh"].values
    df_combined["total_MWh"] = df_hh["hh_MWh"] + df_hp["hp_MWh"]

    return df_combined["total_MWh"]


def read_temperature_dependency():
    """Read temperature dependency.
    
    Returns
    -------
    object
        Result computed as ``pd.read_excel(file, sheet_name='Temperature dependence', header=0, index_col=0)``."""
    file = os.path.join(_ecv2g_path, 'Input_Dependency.xlsx')
    return pd.read_excel(
        file,
        sheet_name='Temperature dependence',
        header=0,
        index_col=0,
    )

def read_battery_capacities():
    """Read battery capacities.
    
    Returns
    -------
    object
        Result computed as ``pd.read_excel(file, sheet_name='Battery capacity', header=[0, 1], index_col=0)``."""
    file = os.path.join(_ecv2g_path, 'Input_Dependency.xlsx')
    return pd.read_excel(
        file,
        sheet_name='Battery capacity',
        header=[0, 1],
        index_col=0,
    )

def read_speed():
    """Read speed.
    
    Returns
    -------
    object
        Result computed as ``pd.read_excel(file, sheet_name='Speed', header=0, index_col=0)``."""
    file = os.path.join(_ecv2g_path, 'Input_Dependency.xlsx')
    return pd.read_excel(
        file,
        sheet_name='Speed',
        header=0,
        index_col=0,
    )

def save_household_load(hh_load:np.ndarray,
                        data_folder_name:str,
                        net_name:str,
                        station_name:str):
    
    """Save household load.
    
    Parameters
    ----------
    hh_load : np.ndarray
        Hh load used by this operation.
    data_folder_name : str
        Data folder name used by this operation.
    net_name : str
        Net name used by this operation.
    station_name : str
        Station name used by this operation."""
    time_index = pd.date_range("2012-01-01", periods=len(hh_load), freq="15min")

    df = pd.DataFrame({
        "time": time_index.strftime("%Y-%m-%d %H:%M"),
        station_name: hh_load,
    })
    df.to_csv(os.path.join(data_path(data_folder_name), f'{net_name}_{station_name}', 'household_load.csv'), sep=";", index=False)


def generate_price_csv():
    # 1. 创建时间索引：2012年是闰年，共366天
    # start: 2012-01-01 00:00, end: 2012-12-31 23:45
    """Generate price csv."""
    timestamps = pd.date_range(start='2012-01-01 00:00:00', 
                               end='2012-12-30 23:45:00', 
                               freq='15min', 
                               inclusive='left')
    
    df = pd.DataFrame(index=timestamps)
    df.index.name = 'timestamp'
    
    # 2. 定义获取价格的函数 (根据图片逻辑)
    def get_price_ct(hour):
        # 23:00 - 6:00 -> 1.64 (跨午夜，所以包括 0,1,2,3,4,5 和 23)
        """Return price ct.
        
        Parameters
        ----------
        hour : object
            Hour used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        if 0 <= hour < 6 or hour == 23:
            return 1.64
        # 6:00 - 11:00 -> 5.48
        elif 6 <= hour < 11:
            return 5.48
        # 11:00 - 15:00 -> 1.64
        elif 11 <= hour < 15:
            return 1.64
        # 15:00 - 17:00 -> 5.48
        elif 15 <= hour < 17:
            return 5.48
        # 17:00 - 22:00 -> 10.52
        elif 17 <= hour < 22:
            return 10.52
        # 22:00 - 23:00 -> 5.48
        elif 22 <= hour < 23:
            return 5.48
        else:
            return 0.0 # Should not happen

    # 3. 应用逻辑
    # df.index.hour 会自动处理每一行的小时数
    df['price_ct_per_kwh'] = [get_price_ct(h) for h in df.index.hour]
    
    # 4. 转换为 Euro/kWh (你的优化器可能需要这个单位)
    # 1 ct = 0.01 Euro
    df['price_euro_per_kwh'] = df['price_ct_per_kwh'] / 100.0
    
    # 5. 保存 CSV
    filename = "electricity_price.csv"
    df.to_csv(filename, sep=";")
    print(f"CSV file '{filename}' generated successfully with {len(df)} rows.")

def read_price():
    """Read price.
    
    Returns
    -------
    object
        Result computed as ``(price_df['price_ct_per_kwh'], price_df['price_euro_per_kwh'])``."""
    file = os.path.join(_ecv2g_path, 'electricity_price.csv')
    price_df = pd.read_csv(
        file,
        sep=";"
    )
    return price_df['price_ct_per_kwh'], price_df['price_euro_per_kwh']


def generate_feedin_price_csv():
    # 1. 创建 2012 全年时间索引 (每15分钟)
    """Generate feedin price csv."""
    timestamps = pd.date_range(start='2012-01-01 00:00:00', 
                               end='2013-01-01 00:00:00', 
                               freq='15min', 
                               inclusive='left')
    
    df = pd.DataFrame(index=timestamps)
    df.index.name = 'timestamp'
    
    # 2. 定义价格逻辑 (模拟 EPEX Spot 现货市场特征)
    # 单位: Euro/kWh (例如 0.035 代表 35 Euro/MWh)
    def get_feedin_price(ts):
        """Return feedin price.
        
        Parameters
        ----------
        ts : object
            Ts used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        hour = ts.hour
        is_weekend = ts.dayofweek >= 5 # 5=周六, 6=周日
        
        if is_weekend:
            # 周末：需求低，价格普遍较低 (模拟图表中的周日低谷)
            # 白天 (10-16点) 稍高，其余很低
            if 10 <= hour < 16:
                return 0.020 # 20 Euro/MWh
            else:
                return 0.010 # 10 Euro/MWh
        else:
            # 工作日：典型的早晚双高峰模式
            if 0 <= hour < 6:       # 深夜低谷
                return 0.020        # 20 Euro/MWh
            elif 6 <= hour < 9:     # 早高峰爬坡
                return 0.035        # 35 Euro/MWh
            elif 9 <= hour < 17:    # 白天平稳期 (太阳能压低价格)
                return 0.028        # 28 Euro/MWh
            elif 17 <= hour < 22:   # 晚高峰 (图表中最高的波峰)
                return 0.040        # 40 Euro/MWh
            else:                   # 22-24点 回落
                return 0.025        # 25 Euro/MWh

    # 3. 应用逻辑生成数据
    df['price_euro_per_kwh'] = [get_feedin_price(ts) for ts in df.index]
    
    # 4. (可选) 添加一些随机波动，让曲线看起来更像真实的"灰色曲线"
    # 波动范围 +/- 0.002 Euro/kWh
    np.random.seed(42) # 固定种子保证结果可复现
    noise = np.random.uniform(-0.002, 0.002, size=len(df))
    df['price_euro_per_kwh'] = df['price_euro_per_kwh'] + noise
    
    # 确保价格不小于 0
    df['price_euro_per_kwh'] = df['price_euro_per_kwh'].clip(lower=0.0)

    # 5. 保存 CSV
    filename = os.path.join(_ecv2g_path, 'feedin_price.csv')
    df.to_csv(filename, sep=";")
    print(f"Successfully generated '{filename}' with {len(df)} rows.")
    print("Preview:")
    print(df.head(10))

def read_feedin_price():
    """Read feedin price.
    
    Returns
    -------
    object
        Result computed as ``price_df['price_euro_per_kwh']``."""
    file = os.path.join(_ecv2g_path, 'feedin_price.csv')
    price_df = pd.read_csv(
        file,
        sep=";"
    )
    return price_df['price_euro_per_kwh']

if __name__ == "__main__":
    generate_feedin_price_csv()
