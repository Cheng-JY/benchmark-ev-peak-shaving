import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from ..base import AggregateCoordinateStrategy

class Wang(AggregateCoordinateStrategy):
    def __init__(self, random_state=None):
        super.__init__(random_state)

    def get_SoC_charging(
            self,
            ch_avail: np.ndarray, 
            SoC_init: float,
            consumption: np.ndarray,
            target_load_ev: np.ndarray,
            SoC_max: np.ndarray,
            SoC_min: np.ndarray,
            charging_status_quo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        a = np.zeros(len(ch_avail))
        b = np.zeros(len(ch_avail))
        y = np.zeros(len(ch_avail))
        charging = np.zeros(len(ch_avail))
        feasible = np.ones(len(ch_avail), dtype=bool)
        diff_ev_load = charging_status_quo - target_load_ev

        for t in range(len(ch_avail)):
            a[t] = np.minimum(SoC_max[t] - SoC[t], ch_avail[t])
            b[t] = np.minimum(SoC[t] - SoC_min[t], ch_avail[t])

            if diff_ev_load[t] == 0:
                y[t] = 0
            elif diff_ev_load[t] > 0:
                # need discharging
                y[t] = (
                    -np.abs(diff_ev_load[t])
                    if np.abs(diff_ev_load[t]) < b[t]
                    else -b[t]
                )
            elif np.abs(diff_ev_load[t]) < a[t]:
                # need charging
                y[t] = np.abs(target_load_ev[t])
            else:
                y[t] = a[t]
            charging[t] = y[t] + charging_status_quo[t]
            SoC[t+1] = np.maximum(SoC[t] + charging[t] - consumption[t], 0)

        return SoC, charging

    def get_coordinate_SoC_charging(self,
            threshold:float,
            household_load: pd.DataFrame,
            charging_status_quo: Dict[str, np.ndarray],
            soc_status_quo: Dict[str, np.ndarray],
            soc_max: Dict[str, np.ndarray],
            soc_min: Dict[str, np.ndarray],
            ch_avail: Dict[str, np.ndarray],
            consumption: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # handling uncontrolled part
        adjusted_threshold = threshold - household_load

        soc_updated_dict = {"uncontrolled": soc_status_quo.get("uncontrolled", np.array([0]))}
        charging_updated_dict = {"uncontrolled": charging_status_quo.get("uncontrolled", np.array([0]))}
        feasible_dict = {
            "uncontrolled": (charging_status_quo.get("uncontrolled", np.array([0])) < adjusted_threshold).astype(bool)}
        threshold_dict = {"uncontrolled": adjusted_threshold}

        uncontrolled_charging = charging_status_quo.get("uncontrolled", np.zeros_like(adjusted_threshold))
        threshold_dict["controlled"] = adjusted_threshold - uncontrolled_charging

        soc_init = soc_max["controlled"][0]
        soc_controlled, charging_controlled = self.get_SoC_charging(
            ch_avail["controlled"],
            soc_init,
            consumption["controlled"],
            threshold_dict["controlled"],
            soc_max["controlled"],
            soc_min["controlled"],
            charging_status_quo["controlled"],
        )
        soc_updated_dict["controlled"] = soc_controlled
        charging_updated_dict["controlled"] = charging_controlled

        feasible_dict["controlled"] = (charging_controlled <= threshold_dict["controlled"]).astype(bool)

        return soc_updated_dict, charging_updated_dict, feasible_dict, threshold_dict


            
                
        

    