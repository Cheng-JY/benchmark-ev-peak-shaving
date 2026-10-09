"""Analysis builder utilities for the EV/V2G analysis workflow."""

import os
import pandas as pd
import numpy as np
from ...utils._data import (
    read_temperature_dependency,
    read_battery_capacities,
)

class AnalysisBuilder:

    """Represent analysis builder and its associated operations."""
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
        self.summary_path = scenario_data.analysis_dir / "summary.csv"
    
    def get_uncontrolled_peak(self, filter_name="home", ch_refuel=0.4):
        """Return the fixed ch_refuel=0.4 uncontrolled reference peak."""
        df_fp = self.scenario_data.analysis_dir / "summary.csv"
        df = pd.read_csv(df_fp, sep=";")

        filtered_data = df[
        (df['region_name'] == self.cfg.region_name) &
        (df['commuting'] == filter_name)
        ]
        filtered_data = filtered_data[
            np.isclose(filtered_data["ch_refuel"], float(ch_refuel))
        ]

        if filtered_data.empty:
            return 0
        peaks = pd.to_numeric(
            filtered_data["uncontrolled_peak"], errors="coerce"
        ).dropna().unique()
        if len(peaks) != 1:
            raise ValueError(
                "More than one uncontrolled peak matches the requested "
                "scenario. Pass ch_refuel explicitly or use the row-wise "
                "calculate_relevant_reduction method."
            )
        return float(peaks[0])

    def update_uncontrolled_peaks(self):
        """Apply the ch_refuel=0.4 uncontrolled peak to every ch_refuel row."""

        file_path = self.scenario_data.analysis_dir / "summary.csv"
        df = pd.read_csv(file_path, sep=";")
        baseline = df[np.isclose(df["ch_refuel"], 0.4)].copy()
        if baseline.empty:
            raise ValueError(
                "summary.csv contains no ch_refuel=0.4 rows for the "
                "uncontrolled peak baseline."
            )

        # Keep separate references for physically different feeder/fleet
        # cases, but never separate them by the controlled ch_refuel value.
        match_columns = [
            column
            for column in ("region_name", "commuting", "n_controlled")
            if column in df.columns
        ]
        baseline = (
            baseline[match_columns + ["uncontrolled_peak"]]
            .groupby(match_columns, as_index=False, dropna=False)
            .first()
            .rename(columns={"uncontrolled_peak": "baseline_uncontrolled_peak"})
        )
        df = df.merge(baseline, on=match_columns, how="left")
        if df["baseline_uncontrolled_peak"].isna().any():
            raise ValueError(
                "Some rows have no matching ch_refuel=0.4 uncontrolled baseline."
            )
        df["uncontrolled_peak"] = df.pop("baseline_uncontrolled_peak")
        df.to_csv(file_path, sep=";", index=False)
        
    def calculate_relevant_reduction(self):
        """
        Calculate reduction against the fixed ch_refuel=0.4 baseline.
        """
        file_path = self.scenario_data.analysis_dir / "summary.csv"
        df = pd.read_csv(file_path, sep=";")
        uncontrolled = pd.to_numeric(df["uncontrolled_peak"], errors="coerce")
        controlled = pd.to_numeric(df["controlled_peak"], errors="coerce")
        df["reduced_relevant"] = (
            (uncontrolled - controlled)
            .div(uncontrolled.where(uncontrolled != 0))
            .mul(100.0)
            .fillna(0.0)
        )
        # Save the updated DataFrame to a new CSV file
        df.to_csv(file_path, sep=";", index=False)
    
    def get_plug_in_duration(self,
                             ch_refuel,
                             sampled_ids):
        """Return plug in duration.
        
        Parameters
        ----------
        ch_refuel : object
            Plug-in motivation parameter associated with the scenario.
        sampled_ids : object
            Sampled ids used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch.
        
        Raises
        ------
        ValueError
            If an explicit validation or processing check in this operation fails."""
        overview_path = self.scenario_data.ev_dir / "plug_in_duration_with_hh.csv"
        plug_df = pd.read_csv(overview_path, sep=";")

        col = f"home_avg_{ch_refuel}"

        if col not in plug_df.columns:
            raise ValueError(f"Column {col} not found in plug_in_duration_with_hh.csv")
        
        filtered = plug_df[plug_df["key"].isin(sampled_ids)]

        if filtered.empty:
            return 0
        return filtered[col].mean()
    
    def add_plug_in_duration_to_summary(self):
        # load summary
        """Add plug in duration to summary.
        
        Returns
        -------
        object
            Result computed as ``df``."""
        df = pd.read_csv(self.summary_path, sep=";")
        overview_path = self.scenario_data.ev_dir / "plug_in_duration_with_hh.csv"
        plug_df = pd.read_csv(overview_path, sep=";")

        # ↓ store results
        plug_values = []

        for _, row in df.iterrows():
            ch_refuel = row["ch_refuel"]
            n_controlled = int(row["n_controlled"])

            sampled_ids = plug_df["key"].tolist()[:n_controlled] if "key" in plug_df.columns else []

            avg_plug = self.get_plug_in_duration(
                ch_refuel=ch_refuel,
                sampled_ids=sampled_ids
            )
            plug_values.append(avg_plug)

        df["plug_in_duration"] = plug_values

        # save result
        out_path = self.scenario_data.analysis_dir / "summary.csv"
        df.to_csv(out_path, sep=";", index=False)

        return df
    
    def temp_dep_consumption(self, temperature=0):
        """Temp dep consumption.
        
        Parameters
        ----------
        temperature : object
            Temperature used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(e_temp_stat, P_instat)``."""
        temp = read_temperature_dependency()
        Q_ohne_sol = temp.loc[temperature, "Q_total in kW"]
        P_stat = temp.loc[temperature, "COP WP"]
        P_instat = temp.loc[temperature, "Vorkonditionierung in kWh"]

        e_temp_stat = Q_ohne_sol / P_stat
        return e_temp_stat, P_instat

    def get_efficiency(self, year, model):
        """Return efficiency.
        
        Parameters
        ----------
        year : object
            Year used by this operation.
        model : object
            Model used by this operation.
        
        Returns
        -------
        object
            Result computed as ``df.loc[year, ('durch. elektrischer Verbrauch in kWh/km', model)]``."""
        df = read_battery_capacities()
        return df.loc[year, ("durch. elektrischer Verbrauch in kWh/km", model)]

    def minimum_distance(self, capacity, model, temperature=0):
        """Minimum distance.
        
        Parameters
        ----------
        capacity : object
            Capacity used by this operation.
        model : object
            Model used by this operation.
        temperature : object
            Temperature used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(capacity - e_pre) / eff``."""
        e_temp_stat, e_pre = self.temp_dep_consumption(temperature)
        eff = self.get_efficiency(self.cfg.scenario_year, model)
        return (capacity - e_pre) / eff  # km

    def add_distance_column(self, model_list=["BEV-Mittel"]):
        """Add distance column.
        
        Parameters
        ----------
        model_list : object
            Model list used by this operation.
        
        Returns
        -------
        object
            Result computed as ``df_final``."""
        df = pd.read_csv(self.summary_path, sep=";")

        rows = []
        for model in model_list:
            for cap in sorted(df["start_discharging_level"].unique()):
                if cap == 0:
                    dist = 0
                else:
                    dist = self.minimum_distance(cap, model)
                rows.append(
                    {
                        "start_discharging_level": cap,
                        "minimum_range": dist,
                    }
                )

        df_dist = pd.DataFrame(rows)

        df_final = df.merge(
            df_dist, on="start_discharging_level", how="left"
        )

        final_path = self.scenario_data.analysis_dir / "summary.csv"
        df_final.to_csv(final_path, sep=";", index=False)
        print(f"[Analysis] summary_final written → {final_path}")
        return df_final

    def run_all(self):
        """
        完成所有 a7 分析步骤：
        1. replace uncontrolled peaks with the ch_refuel=0.4 baseline
        2. peak reduction calculation
        3. minimum distance calculation
        """
        self.update_uncontrolled_peaks()
        self.calculate_relevant_reduction()
        self.add_plug_in_duration_to_summary()
        final_df = self.add_distance_column()
        return final_df




