"""Greedy optimizer utilities for the EV/V2G analysis workflow."""

from .base_optimizer import BaseEVOptimizer

import numpy as np
import pandas as pd
import math

from ...utils import get_state

"""
# How to use
    opt_cfg = OptimizationConfig()
    opt_cfg.controlled_name = "controlled_v2g"

    vehicle_scenarios = {
        "uncontrolled": (0.1, 15, 50),
        "controlled_v2g": (0.1, 10, 35)
    }

    optimizer = GreedyEVOptimizer(
        opt_config=opt_cfg,
        scenario_data=scenario_data,
        scenario_config=cfg,
    )

    optimizer.build_model(input_data=None, vehicle=vehicle_scenarios)

    results = optimizer.optimize()

    optimizer.export_results(results)
"""


class GreedyEVOptimizer(BaseEVOptimizer):

    """Represent greedy evoptimizer and its associated operations."""
    def __init__(self, 
                 opt_config, 
                 scenario_config, 
                 scenario_data,
                 charging_efficiency:float=0.95,
                 discharging_efficiency:float|None=None):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        opt_config : object
            Opt config used by this operation.
        scenario_config : object
            Scenario config used by this operation.
        scenario_data : object
            Scenario data used by this operation.
        charging_efficiency : float
            Charging efficiency used by this operation.
        discharging_efficiency : float | None
            Discharging efficiency used by this operation."""
        super().__init__(opt_config)
        self.cfg = scenario_config
        self.scenario_data = scenario_data
        self.tolerance = 0.0015
        # Numerical feasibility tolerances. Power values are expressed in MW
        # and SoC values in MWh. A 1e-6 MW tolerance corresponds to 1 W and
        # prevents CSV/float rounding at an otherwise feasible peak boundary
        # from being treated as a physical violation.
        self.power_feasibility_tolerance = 1e-6
        self.soc_feasibility_tolerance = 1e-3
        self.charging_efficiency = self._validate_efficiency(
            "charging_efficiency", charging_efficiency)
        if discharging_efficiency is None:
            self.discharging_efficiency = self.charging_efficiency
        else:
            self.discharging_efficiency = self._validate_efficiency(
                "discharging_efficiency", discharging_efficiency)

    @staticmethod
    def _validate_efficiency(name: str, value: float) -> float:
        """Validate the supplied efficiency value before using it in calculations.
        
        Parameters
        ----------
        name : str
            Name used by this operation.
        value : float
            Value used by this operation.
        
        Returns
        -------
        float
            Result computed as ``value``.
        
        Raises
        ------
        ValueError
            If an explicit validation or processing check in this operation fails."""
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in the interval (0, 1], got {value!r}.")
        return value

    def grid_to_battery_energy(self, grid_energy):
        """Convert signed grid-side energy to signed battery-side energy."""

        values = np.asarray(grid_energy, dtype=float)
        converted = np.where(
            values >= 0.0,
            values * self.charging_efficiency,
            values / self.discharging_efficiency,
        )
        return float(converted) if converted.ndim == 0 else converted

    def battery_to_grid_energy(self, battery_energy):
        """Convert a required signed battery-energy change to grid energy."""

        values = np.asarray(battery_energy, dtype=float)
        converted = np.where(
            values >= 0.0,
            values / self.charging_efficiency,
            values * self.discharging_efficiency,
        )
        return float(converted) if converted.ndim == 0 else converted


    def build_model(self, vehicle: dict[str: (float, float, int, str)]):
        
        """Build model.
        
        Parameters
        ----------
        vehicle : dict[str:(float, float, int, str)]
            Vehicle used by this operation."""
        self.vehicle = vehicle

    def get_soc(self, 
                ch_avail: pd.Series, 
                soc_init: pd.Series,
                charging: pd.Series, 
                consumption: pd.Series) -> np.ndarray:
        """Return soc.
        
        Parameters
        ----------
        ch_avail : pd.Series
            Ch avail used by this operation.
        soc_init : pd.Series
            Soc init used by this operation.
        charging : pd.Series
            Charging used by this operation.
        consumption : pd.Series
            Consumption used by this operation.
        
        Returns
        -------
        np.ndarray
            Result computed as ``soc``."""
        soc = np.zeros(len(ch_avail))
        soc[0] = soc_init
        for t in range(1, len(ch_avail)):
            soc[t] = np.maximum(soc[t-1] 
                                + self.grid_to_battery_energy(charging[t-1]) 
                                - consumption[t-1], 
                                0)
        return soc

    def update_charging_smart(self, 
                        soc: np.ndarray,
                        soc_max: np.ndarray,
                        soc_min: np.ndarray,
                        ch_avail: np.ndarray,
                        threshold: np.ndarray,
                        charging: np.ndarray,
                        consumption: np.ndarray):
        
        """Update charging smart.
        
        Parameters
        ----------
        soc : np.ndarray
            Soc used by this operation.
        soc_max : np.ndarray
            Soc max used by this operation.
        soc_min : np.ndarray
            Soc min used by this operation.
        ch_avail : np.ndarray
            Ch avail used by this operation.
        threshold : np.ndarray
            Threshold used by this operation.
        charging : np.ndarray
            Charging used by this operation.
        consumption : np.ndarray
            Consumption used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(soc, charging_updated, shifted_charging, check_max_min_soc, feasible, charging_threshold)``."""
        charging_threshold = np.minimum(threshold, ch_avail)
        T = len(soc)

        charging_updated = np.zeros(T)
        shifted_charging = np.zeros(T)
        charging_should = np.zeros(T)
        check_max_min_soc = np.zeros(T)
        feasible = np.ones(len(ch_avail), dtype=bool)
        
        for t in range(T-1):
            charging_should[t] = charging[t] + shifted_charging[t]
            charging_temp = np.minimum(charging_should[t], charging_threshold[t])
            soc_state = soc[t] + self.grid_to_battery_energy(charging_temp) - consumption[t]
            check_max_min_soc[t+1] = soc_state - np.clip(soc_state, soc_min[t+1], soc_max[t+1])
            charging_temp = charging_should[t] - check_max_min_soc[t]/self.charging_efficiency
    
            if np.isclose(
                check_max_min_soc[t+1],
                0.0,
                atol=self.soc_feasibility_tolerance,
            ):
                if charging_threshold[t] < -self.power_feasibility_tolerance:
                    feasible[t] = False
                charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)

            elif check_max_min_soc[t+1] > 0:
                if charging_threshold[t] < -self.power_feasibility_tolerance:
                    feasible[t] = False
                charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)

            elif check_max_min_soc[t+1] < 0:
                if charging_temp <= charging_threshold[t]:
                    if charging_threshold[t] < -self.power_feasibility_tolerance:
                        feasible[t] = False
                    charging_updated[t] = np.maximum(charging_threshold[t], 0)
                    shifted_charging[t+1] = charging_should[t] - charging_updated[t]
                else:
                    if charging_threshold[t] < -self.power_feasibility_tolerance:
                        feasible[t] = False
                    charging_updated[t] = np.maximum(charging_threshold[t], 0)
            shifted_charging[t+1] = charging_should[t] - charging_updated[t]

            soc[t+1] = (soc[t] 
                        + self.grid_to_battery_energy(charging_updated[t]) 
                        - consumption[t])
            
            if (
                soc[t+1]
                > soc_max[t+1] + self.soc_feasibility_tolerance
            ):
                feasible[t] = False
                charging_updated[t] = np.maximum(self.battery_to_grid_energy(soc_max[t+1] - soc[t] + consumption[t])
                                                 , 0)
                soc[t+1] = soc_max[t+1]
            elif (
                soc[t+1]
                < soc_min[t+1] - self.soc_feasibility_tolerance
            ):
                feasible[t] = False
                charging_updated[t] = np.maximum(self.battery_to_grid_energy(soc_min[t+1] - soc[t] + consumption[t])
                                                 , 0)
                soc[t+1] = soc_min[t+1]

        charging_updated[-1] = np.minimum(charging[-1] + shifted_charging[-1], charging_threshold[-1])

        return soc, charging_updated, shifted_charging, check_max_min_soc, feasible, charging_threshold

        
    
    def update_charging_v2g(self, 
                        soc: np.ndarray,
                        soc_max: np.ndarray,
                        soc_min: np.ndarray,
                        ch_avail: np.ndarray,
                        threshold: np.ndarray,
                        charging: np.ndarray,
                        consumption: np.ndarray):

        # ``ch_avail`` is the grid-side connection capacity selected for the
        # current option.  It must constrain both directions of V2G power:
        #
        #     -ch_avail[t] <= charging_updated[t] <= ch_avail[t]
        #
        # The previous implementation only applied the positive upper bound.
        # Consequently, a negative grid threshold could request more
        # discharging power than the connected vehicles could provide.
        """Update charging v2g.
        
        Parameters
        ----------
        soc : np.ndarray
            Soc used by this operation.
        soc_max : np.ndarray
            Soc max used by this operation.
        soc_min : np.ndarray
            Soc min used by this operation.
        ch_avail : np.ndarray
            Ch avail used by this operation.
        threshold : np.ndarray
            Threshold used by this operation.
        charging : np.ndarray
            Charging used by this operation.
        consumption : np.ndarray
            Consumption used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(soc, charging_updated, shifted_charging, check_max_min_soc, feasible, charging_threshold)``.
        
        Raises
        ------
        ValueError
            If an explicit validation or processing check in this operation fails."""
        ch_avail = np.asarray(ch_avail, dtype=float)
        threshold = np.asarray(threshold, dtype=float)

        if np.any(~np.isfinite(ch_avail)):
            raise ValueError("ch_avail contains non-finite values.")
        if np.any(ch_avail < -self.power_feasibility_tolerance):
            raise ValueError("ch_avail must be non-negative.")

        ch_avail = np.maximum(ch_avail, 0.0)
        charging_threshold = np.minimum(threshold, ch_avail)
        discharging_threshold = -ch_avail
        T = len(soc)

        charging_updated = np.zeros_like(soc)
        shifted_charging = np.zeros_like(soc)
        charging_should = np.zeros_like(soc)
        check_max_min_soc = np.zeros_like(soc)
        feasible = np.ones(T, dtype=bool)

        for t in range(T-1):
            charging_should[t] = charging[t] + shifted_charging[t]

            lower_power = discharging_threshold[t]
            upper_power = charging_threshold[t]

            # If the requested feeder threshold lies below the maximum
            # available V2G discharge, the feeder target is infeasible.  Use
            # the largest physically available discharge for a consistent
            # diagnostic trajectory and mark this time step infeasible.
            if upper_power < lower_power - self.power_feasibility_tolerance:
                charging_temp = lower_power
                feasible[t] = False
            else:
                charging_temp = np.clip(
                    charging_should[t],
                    lower_power,
                    upper_power,
                )

            soc_state = (soc[t]
                        + self.grid_to_battery_energy(charging_temp)
                        - consumption[t]) # update soc
            check_max_min_soc[t+1] = soc_state - np.clip(soc_state, soc_min[t+1], soc_max[t+1])

            if np.isclose(
                check_max_min_soc[t+1],
                0.0,
                atol=self.soc_feasibility_tolerance,
            ):
                requested_power = charging_temp
            elif soc_state > soc_max[t+1]:
                battery_change = soc_max[t+1] - soc[t] + consumption[t]
                requested_power = self.battery_to_grid_energy(battery_change)
            elif soc_state < soc_min[t+1] - self.soc_feasibility_tolerance:
                battery_change = soc_min[t+1] - soc[t] + consumption[t]
                requested_power = self.battery_to_grid_energy(battery_change)

            # A correction required by the SoC bounds must still respect both
            # the connection capacity and the feeder threshold.  If those
            # requirements are incompatible, keep the closest admissible
            # power and report the step as infeasible.
            if upper_power < lower_power - self.power_feasibility_tolerance:
                charging_updated[t] = lower_power
            else:
                if (
                    requested_power
                    < lower_power - self.power_feasibility_tolerance
                    or requested_power
                    > upper_power + self.power_feasibility_tolerance
                ):
                    feasible[t] = False
                charging_updated[t] = np.clip(
                    requested_power,
                    lower_power,
                    upper_power,
                )

            shifted_charging[t+1] = charging_should[t] - charging_updated[t]
            soc[t+1] = soc[t] + self.grid_to_battery_energy(charging_updated[t]) - consumption[t]

            if (
                soc[t+1]
                > soc_max[t+1] + self.soc_feasibility_tolerance
            ):
                feasible[t] = False
            elif (
                soc[t+1]
                < soc_min[t+1] - self.soc_feasibility_tolerance
            ):
                feasible[t] = False

            if (
                charging_updated[t]
                > charging_threshold[t] + self.power_feasibility_tolerance
                or charging_updated[t]
                < discharging_threshold[t] - self.power_feasibility_tolerance
            ):
                feasible[t] = False

        last_lower = discharging_threshold[-1]
        last_upper = charging_threshold[-1]
        last_requested = charging[-1] + shifted_charging[-1]
        if last_upper < last_lower - self.power_feasibility_tolerance:
            charging_updated[-1] = last_lower
            feasible[-1] = False
        else:
            charging_updated[-1] = np.clip(
                last_requested,
                last_lower,
                last_upper,
            )

        # Final defensive check: a run can never be reported feasible when
        # its V2G power exceeds the option-specific connection capacity.
        availability_violation = (
            np.abs(charging_updated)
            > ch_avail + self.power_feasibility_tolerance
        )
        feeder_violation = (
            charging_updated
            > charging_threshold + self.power_feasibility_tolerance
        )
        feasible[availability_violation | feeder_violation] = False

        return soc, charging_updated, shifted_charging, check_max_min_soc, feasible, charging_threshold


    def get_coordination(self,
                         household_load: pd.DataFrame,
                         charging: dict[str, pd.Series],
                         soc: dict[str, pd.Series],
                         soc_max: dict[str, pd.Series],
                         soc_min: dict[str, pd.Series],
                         ch_avail: dict[str, pd.Series],
                         consumption: dict[str, pd.Series],
                         grid_threshold: float):
        """Return coordination.
        
        Parameters
        ----------
        household_load : pd.DataFrame
            Household load used by this operation.
        charging : dict[str, pd.Series]
            Charging used by this operation.
        soc : dict[str, pd.Series]
            Soc used by this operation.
        soc_max : dict[str, pd.Series]
            Soc max used by this operation.
        soc_min : dict[str, pd.Series]
            Soc min used by this operation.
        ch_avail : dict[str, pd.Series]
            Ch avail used by this operation.
        consumption : dict[str, pd.Series]
            Consumption used by this operation.
        grid_threshold : float
            Grid threshold used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(soc_upd, charging_upd, feas, thr_dict, shifted)``."""
        threshold = grid_threshold - household_load

        soc_upd = {}
        charging_upd = {}
        feas = {}
        thr_dict = {}

        for key in charging.keys():
            if "uncontrolled" in key:
                soc_upd[key] = soc.get(key, 0)
                charging_upd[key] = charging.get(key, 0)
                feas[key] = (
                    charging.get(key, 0)
                    <= threshold + self.power_feasibility_tolerance
                )
                thr_dict[key] = threshold
            elif "smart_charging" in key:
                soc_updated, charging_updated, _, _, feasible, _ = self.update_charging_smart(
                    soc=soc.get(key),
                    soc_max=soc_max.get(key),
                    soc_min=soc_min.get(key),
                    ch_avail=ch_avail.get(key),
                    threshold=threshold,
                    charging=charging.get(key),
                    consumption=consumption.get(key),
                )
                soc_upd[key] = soc_updated
                charging_upd[key] = charging_updated
                feas[key] = feasible
                thr_dict[key] = threshold
            elif "v2g" in key:
                """
                soc_updated, charging_updated, _, _, feasible, _ = self.update_charging_smart(
                    soc=soc.get(key),
                    soc_max=soc_max.get(key),
                    soc_min=soc_min.get(key),
                    ch_avail=ch_avail.get(key),
                    threshold=threshold,
                    charging=charging.get(key),
                    consumption=consumption.get(key),
                )
                soc_upd[f"{key}_wo"] = soc_updated
                charging_upd[f"{key}_wo"] = charging_updated
                feas[f"{key}_wo"] = feasible
                thr_dict[f"{key}_wo"] = threshold
                """
                soc_updated, charging_updated, shifted, _, feasible, _ = self.update_charging_v2g(
                    soc=soc.get(key),
                    soc_max=soc_max.get(key),
                    soc_min=soc_min.get(key),
                    ch_avail=ch_avail.get(key),
                    threshold=threshold,
                    charging=charging.get(key),
                    consumption=consumption.get(key),
                )
                soc_upd[key] = soc_updated
                charging_upd[key] = charging_updated
                feas[key] = feasible
                thr_dict[key] = threshold

            threshold = threshold - charging_upd[key]
        
        return soc_upd, charging_upd, feas, thr_dict, shifted
    
    def get_peak_reduction(self,
                           load: pd.Series,
                           ch_avail_dict,
                           soc_max_dict,
                           soc_min_dict,
                           charging_dict,
                           consumption_dict,
                           soc_dict,
                           uncontrolled_charging_dict=None,
                           ):
        """Return peak reduction.
        
        Parameters
        ----------
        load : pd.Series
            Load used by this operation.
        ch_avail_dict : object
            Ch avail dict used by this operation.
        soc_max_dict : object
            Soc max dict used by this operation.
        soc_min_dict : object
            Soc min dict used by this operation.
        charging_dict : object
            Charging dict used by this operation.
        consumption_dict : object
            Consumption dict used by this operation.
        soc_dict : object
            Soc dict used by this operation.
        uncontrolled_charging_dict : object
            Uncontrolled charging dict used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(result_df, target_load_upper_bound, uncontrolled_peak)``.
        
        Raises
        ------
        RuntimeError
            If an explicit validation or processing check in this operation fails."""
        print("excute in greedy optimizer")
        # ``charging_dict`` is the charging demand used by the selected
        # flexibility option.  The comparison baseline may intentionally be a
        # different, fixed uncontrolled scenario (for example the
        # ch_refuel + ch_avail case).  Keep both concepts separate.
        reference_charging = (
            charging_dict
            if uncontrolled_charging_dict is None
            else uncontrolled_charging_dict
        )
        uncontrolled_peak = np.max(
            load + sum((reference_charging[k] for k in reference_charging.keys()))
        )
        option_uncontrolled_peak = np.max(
            load + sum((charging_dict[k] for k in charging_dict.keys()))
        )
        # The initial trial must be feasible even if the selected option has a
        # higher native charging peak than the reference baseline.
        upper_base = max(
            float(uncontrolled_peak), float(option_uncontrolled_peak)
        )

        # CSV rounding can make a mathematically feasible boundary fail by a
        # few watts. Start with a 1 W margin and, only if necessary, retry up
        # to a 1 kW margin. This is a numerical safeguard, not a relaxation of
        # the battery, SoC, or connection-capacity constraints.
        upper_margins_MW = (1e-6, 1e-5, 1e-4, 1e-3)
        feasible_upper = False
        for upper_margin in upper_margins_MW:
            target_load_upper_bound = upper_base + upper_margin
            _, _, _, _, feasible_upper, s = self.compute_feasibility(
                target_load_upper_bound,
                load,
                ch_avail_dict,
                soc_max_dict,
                soc_min_dict,
                charging_dict,
                consumption_dict,
                soc_dict,
                self.opt_cfg.controlled_name,
            )
            if feasible_upper:
                break

        if not feasible_upper:
            raise RuntimeError(
                f"Initial upper bound remains infeasible after numerical "
                f"margins up to {upper_margins_MW[-1]:.6f} MW "
                f"(tested bound: {target_load_upper_bound:.6f} MW). "
                "This indicates an actual inconsistency in charging "
                "availability, shifted demand, or SoC limits."
            )

        target_load_lower_bound = 0.25 * upper_base
        
        _,_,_,_, feasible_lower, s = self.compute_feasibility(target_load_lower_bound,
                                                           load,
                                                           ch_avail_dict,
                                                           soc_max_dict,
                                                           soc_min_dict,
                                                           charging_dict,
                                                           consumption_dict,
                                                           soc_dict,
                                                           self.opt_cfg.controlled_name)
        
        iteration = 0
        mid = target_load_upper_bound
        print(feasible_upper, feasible_lower)
        while (target_load_upper_bound - target_load_lower_bound > self.tolerance) and (iteration < 50):
            print(target_load_upper_bound, target_load_lower_bound)
            mid = (target_load_lower_bound + target_load_upper_bound) / 2 * 1000 / 1000
            print(f"Iteration {iteration}: Testing threshold {mid: .3f}")
        
            _, _, _, _, feasible_mid,s = self.compute_feasibility(mid,
                                                                load,
                                                                ch_avail_dict,
                                                                soc_max_dict,
                                                                soc_min_dict,
                                                                charging_dict,
                                                                consumption_dict,
                                                                soc_dict,
                                                                self.opt_cfg.controlled_name)
            if feasible_mid:
                target_load_upper_bound = mid
            else:
                target_load_lower_bound = mid
            iteration += 1
        
        su_dict, cu_dict, feas_dict, th_dict, _, s = self.compute_feasibility(target_load_upper_bound,
                                                                           load,
                                                                           ch_avail_dict,
                                                                           soc_max_dict,
                                                                           soc_min_dict,
                                                                           charging_dict,
                                                                           consumption_dict,
                                                                           soc_dict,
                                                                           self.opt_cfg.controlled_name)

        result_df = pd.DataFrame({'load': load})
        for key in ch_avail_dict.keys():
            if "controlled_v2g" in key:
                result_df[f'charging_status_{key}'] = charging_dict.get(key)
                controlled_part = key
            result_df[f'SOC_update_{key}'] = su_dict.get(key)
            result_df[f'charging_updated_{key}'] = cu_dict.get(key)
            result_df[f'feasible_{key}'] = feas_dict.get(key)
        result_df['shifted'] = s
        
        infeasible_locations = np.where(feas_dict.get(controlled_part) == False)
        print("Infeasible indices for controlled_v2g:", infeasible_locations)
        
        return result_df, target_load_upper_bound, uncontrolled_peak
            


    def compute_feasibility(self,
                            threshold,
                            load,
                            ch_avail_dict,
                            soc_max_dict,
                            soc_min_dict,
                            charging_dict,
                            consumption_dict,
                            soc_dict,
                            controlled_name,
                            ):
        """Check whether the proposed trajectory satisfies the modeled feasibility conditions.
        
        Parameters
        ----------
        threshold : object
            Threshold used by this operation.
        load : object
            Load used by this operation.
        ch_avail_dict : object
            Ch avail dict used by this operation.
        soc_max_dict : object
            Soc max dict used by this operation.
        soc_min_dict : object
            Soc min dict used by this operation.
        charging_dict : object
            Charging dict used by this operation.
        consumption_dict : object
            Consumption dict used by this operation.
        soc_dict : object
            Soc dict used by this operation.
        controlled_name : object
            Controlled name used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(su_dict, cu_dict, feas_dict, th_dict, feasible, shifted)``."""
        print("excute compute_feasibility in greedy optimizer")
        su_dict, cu_dict, feas_dict, th_dict, shifted = self.get_coordination(
            household_load=load,
            charging=charging_dict,
            soc=soc_dict,
            soc_max=soc_max_dict,
            soc_min=soc_min_dict,
            ch_avail=ch_avail_dict,
            consumption=consumption_dict,
            grid_threshold=threshold,
        )
        feasible = np.all(feas_dict[controlled_name])
        return su_dict, cu_dict, feas_dict, th_dict, feasible, shifted

    
    def optimize(self):
        """Optimize .
        
        Returns
        -------
        object
            Result computed as ``results``."""
        load = self.get_load()
        vehicle = self.vehicle

        ch_avail_dict = {}
        soc_max_dict = {}
        soc_min_dict = {}
        charging_dict = {}
        consumption_dict = {}
        soc_dict = {}
        for key in vehicle.keys():
            ch_refuel, start_discharging, n_controlled = vehicle.get(key)
            ch_refuel_use = 0.9 if ch_refuel == 1 else ch_refuel
            group = key
            filename = f"ch_profiles_agg_home_DEU_" + \
                       f"20260720_2026_cal2026_R3.csv"
            ch_avail, soc_max, soc_min, charging, consumption = self.get_ev_load(
                ch_refuel, start_discharging, n_controlled, filename, group
            )
            ch_avail_dict[key] = ch_avail.to_numpy().flatten()
            soc_max_dict[key] = soc_max.to_numpy().flatten()
            soc_min_dict[key] = soc_min.to_numpy().flatten()
            charging_dict[key] = charging.to_numpy().flatten()
            consumption_dict[key] = consumption.to_numpy().flatten()
            soc = self.get_soc(ch_avail_dict[key], soc_max_dict[key][0], charging_dict[key], consumption_dict[key])
            soc_dict[key] = soc
        
        results = self.get_peak_reduction(load,
                                          ch_avail_dict,
                                          soc_max_dict,
                                          soc_min_dict,
                                          charging_dict,
                                          consumption_dict,
                                          soc_dict)
        return results
    
    def export_results(self,
                       results,
                       filename):
        
        """Export results.
        
        Parameters
        ----------
        results : object
            Results used by this operation.
        filename : object
            Filename used by this operation."""
        result_df, reducted_peak, uncontrolled_peak = results
        result_filename = f"result_{filename}_con_{reducted_peak:.3f}_from_{uncontrolled_peak:.3f}.csv"
        result_fp = self.scenario_data.optimization_dir / result_filename
        result_df.to_csv(result_fp, sep=';')
