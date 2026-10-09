import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import plotly.graph_objects as go

mapping_dict = {
    'EFH': 'EZFH',
    'MFH': 'KMFH',
    'GMH': 'KMFH',
    'RH_': 'NWG',
    'teilsaniert': 'partly-renovated',
    'vollsaniert': 'renovated',
    'saniert': 'renovated',
    'unsaniert': 'unrenovated',
}

def read_heatpump_timeseries(filepath, sheet_ts, sheet_sim):
    # Read timeseries from excel file and return 2 dataframes: one with timeseries and one with simultaneity factors
    
    # Read timeseries
    df_ts = pd.read_excel(
        filepath, 
        sheet_name=sheet_ts, 
        index_col=0,
        usecols=lambda col: str(col).startswith(("EZFH", "NWG", "KMFH")) or col.startswith("time index")
    )

    # Read simultaneity factors
    df_sim = pd.read_excel(filepath, sheet_name=sheet_sim, index_col=0)
    return df_ts, df_sim

def get_row(df_config, building_type, hp_type, is_renovated):
    row = df_config[
        (df_config['building_type'] == building_type) &
        (df_config['hp_type'] == hp_type) &
        (df_config['is_renovated'] == is_renovated)
    ]
    if row.empty:
        return row
    return row.iloc[0]

def get_simultaneity_factor(row: pd.Series, df_sim: pd.DataFrame, sum_hp: int):
    """Extract simultaneity factor for a given number of heatpumps."""
    if row.empty:
        return None
    col_excel_sim = row["col_excel_sim"]
    if sum_hp in df_sim.index:
        return df_sim.loc[sum_hp, col_excel_sim]
    larger_indices = df_sim.index[df_sim.index > sum_hp]
    if len(larger_indices) > 0:
        next_larger = larger_indices.min()
        return df_sim.loc[next_larger, col_excel_sim]
    max_index = df_sim.index.max()
    return df_sim.loc[max_index, col_excel_sim]

def adjust_timeseries(
    df_config: pd.DataFrame,
    df_ts: pd.DataFrame,
    building_type: str,
    hp_type: str,
    is_renovated: str,
    heatpump_stat:pd.DataFrame,
    plot: bool = False,
    time_interval: int = 96 * 7,
):
    """
    Scale and shape a 15-min heatpump timeseries to meet target annual energy (kWh)
    and simultaneity-adjusted peak load.
    """
    building_type_mapped = mapping_dict.get(building_type)
    is_renovated_mapped = mapping_dict.get(is_renovated)

    heatpump_row = heatpump_stat[
        (heatpump_stat["building_type"] == building_type) 
        & (heatpump_stat["sanierung"] == is_renovated)]
    floor_area = heatpump_row["floor_area_m2"].values[0]
    building_count = heatpump_row["building_count"].values[0]

    if building_type_mapped in ['KMFH', 'NWG']:
        is_renovated_mapped = 'unknown'

    row = get_row(df_config, building_type_mapped, hp_type, is_renovated_mapped)
    if row.empty:
        return None

    # parameters
    col_excel = row["col_excel"]
    annual_consumption_m2 = row["annual_consumption_m2"]
    annual_consumption_target = annual_consumption_m2 * floor_area / building_count
    glz_factor = row["sim_factor"]

    ts_original = df_ts[col_excel].values.astype(float)

    # --- Step 1: scale to target annual energy ---
    sum_current_kwh = np.sum(ts_original) * 0.25
    if sum_current_kwh <= 0:
        raise ValueError("Sum of base timeseries <= 0.")
    ts_scaled = ts_original * (annual_consumption_target / sum_current_kwh)

    # --- Step 2: adjust shape interval-wise ---
    n_steps = len(ts_scaled)
    n_intervals = math.ceil(n_steps / time_interval)
    ts_final_all = []

    for d in range(n_intervals):
        start_idx = d * time_interval
        end_idx = min((d + 1) * time_interval, len(ts_scaled))
        ts_day = ts_scaled[start_idx:end_idx]

        # Some summer intervals may contain only zero heat-pump load.
        positive_values = ts_day[ts_day > 0]

        if positive_values.size == 0:
            ts_final_all.extend(np.zeros_like(ts_day, dtype=float))
            continue

        consumption_target = np.sum(ts_day) * 0.25
        peak_target = glz_factor * np.max(ts_day)

        # Protect the optimization against division by zero.
        if consumption_target <= 0 or peak_target <= 0:
            ts_final_all.extend(np.zeros_like(ts_day, dtype=float))
            continue

        mean_ts = float(np.mean(positive_values))

        def objective(p):
            ts_tmp = mean_ts * np.power(ts_day / mean_ts, p)
            sum_tmp = np.sum(ts_tmp) * 0.25
            ts_tmp *= consumption_target / sum_tmp
            peak_err = ((np.max(ts_tmp) - peak_target) ** 2) / peak_target
            energy_err = ((np.sum(ts_tmp) * 0.25 - consumption_target) ** 2) / consumption_target
            return peak_err + energy_err

        res = minimize_scalar(objective, bounds=(0.01, 20.0), method="bounded", options={"xatol": 1e-6})
        p_opt = float(res.x)

        ts_final = mean_ts * np.power(ts_day / mean_ts, p_opt)
        ts_final *= consumption_target / (np.sum(ts_final) * 0.25)
        ts_final_all.extend(ts_final)

    ts_final_all = np.asarray(ts_final_all, dtype=float)

    if not np.isfinite(ts_final_all).all():
        invalid_count = int(
            (~np.isfinite(ts_final_all)).sum()
        )
        raise ValueError(
            f"Generated HP profile contains {invalid_count} "
            f"NaN/Inf values for "
            f"{building_type}/{hp_type}/{is_renovated}"
        )
    _plot_timeseries_if_requested(plot, df_ts, ts_scaled, ts_final_all, building_type, hp_type, is_renovated, glz_factor)
    return ts_final_all


def _plot_timeseries_if_requested(plot, df_ts, ts_scaled, ts_final_all, building_type, hp_type, is_renovated, glz_factor):
    if not plot:
        return
    time_index = pd.date_range(start="2012-01-01", periods=len(ts_final_all), freq="15min")
    peak_target = glz_factor * np.max(ts_scaled)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=ts_final_all, mode="lines", name="Adjusted", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=time_index, y=ts_scaled, mode="lines", name="Scaled", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=time_index, y=[peak_target]*len(ts_final_all), mode="lines", name="Target Peak",
                             line=dict(color="red", dash="dash")))
    fig.update_layout(title=f"{building_type}-{hp_type}-{is_renovated}", height=400, width=900)
    fig.show()

def get_heat_pump_statistics(net_pop_data: pd.DataFrame) -> pd.DataFrame:
    """
    Gruppiert net_pop_data nach Gebäudetyp und gibt Statistiken aus.
    
    Args:
        net_pop_data (pd.DataFrame): DataFrame mit Populationsdaten, muss Spalten 'gebauedetyp', 'volumen', 'nutzflaeche' enthalten
        
    Returns:
        pd.DataFrame: DataFrame mit Gruppierungsstatistiken nach Gebäudetyp
    """
    
    # Gruppiere nach Gebäudetyp
    hp_data = net_pop_data[net_pop_data['has_hp'] == True].copy()

    if hp_data.empty:
        print("Keine Gebäude mit Wärmepumpe in den Daten gefunden.")
        return pd.DataFrame(columns=[
            "building_type", "sanierung", "volume_m3", "floor_area_m2", "building_count"
        ])

    grouped = hp_data.groupby(['gebauedetyp', 'sanierung']).agg({
        'volumen': 'sum' if 'volumen' in net_pop_data.columns else lambda x: 0,
        'nutzflaeche': 'sum' if 'nutzflaeche' in net_pop_data.columns else lambda x: 0,
        'gebauedetyp': 'count',  # Anzahl der Zeilen
        'has_hp': 'sum'
    }).rename(columns={'gebauedetyp': 'building_count', 'sanierung':'renovated', 'volumen': 'volume_m3', 'nutzflaeche': 'floor_area_m2', 'has_hp': 'hp_count'})
    
    # Sortiere nach Anzahl der Gebäude (absteigend)
    grouped = grouped.sort_values('building_count', ascending=False)
    
    # Runde numerische Werte auf 2 Dezimalstellen
    numeric_columns = ['volume_m3', 'floor_area_m2']
    for col in numeric_columns:
        if col in grouped.columns:
            grouped[col] = grouped[col].round(2)
    
    grouped.reset_index(inplace=True)
    grouped.rename(columns={'gebauedetyp': 'building_type'}, inplace=True)
    
    return grouped

