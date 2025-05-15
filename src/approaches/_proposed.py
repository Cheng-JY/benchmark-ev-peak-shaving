import numpy as np
import pandas as pd
from ..base import AggregateCoordinateStrategy

class Proposed(AggregateCoordinateStrategy):
    def __init__(self, 
                 random_state=None, 
                 threshold: float=1e-7):
        super.__init__(random_state)
        self.THESHOLD = threshold
    
    def get_SoC_charging_smart_charging(
            self,
            ch_avail: np.ndarray,
            SoC_init:float,
            consumption: np.ndarray,
            target_load_ev: np.ndarray,
            SoC_max: np.ndarray,
            SoC_min: np.ndarray,
            charging_status_quo: np.ndarray) -> np.ndarray:
        
        charging_threshold = np.minimum(target_load_ev, ch_avail) 

        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        charging_updated = np.zeros(len(ch_avail))

        check_max_min_SoC = np.zeros(len(ch_avail))
        shifted_charging = np.zeros(len(ch_avail))
        feasible = np.ones(len(ch_avail), dtype=bool)

        for t in range(len(charging_status_quo)-1):
            charging_should = charging_status_quo[t] + charging_threshold[t]
            charging = np.minimum(charging_should, charging_threshold[t])
            SOC_state = SoC[t] + charging - consumption[t]
            check_max_min_SoC[t+1] = SOC_state - np.clip(SOC_state, SoC_min[t+1], SoC_max[t+1])
            charging_temp = charging_should - check_max_min_SoC[t]
    
            if np.isclose(check_max_min_SoC[t+1], 0.0, atol=1e-7):
                charging_temp = charging_status_quo[t] + shifted_charging[t]
                if charging_threshold[t] < -1e-8:
                    feasible[t] = False
                charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif check_max_min_SoC[t+1] > 0:
                if charging_threshold[t] < -1e-8:
                    feasible[t] = False
                charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif check_max_min_SoC[t+1] < 0:
                if charging_temp <= charging_threshold[t]:
                    if charging_threshold[t] < -1e-8:
                        feasible[t] = False
                    charging_updated[t] = np.maximum(charging_threshold[t], 0)
                    shifted_charging[t+1] = charging_should - charging_updated[t]
                else:
                    if charging_threshold[t] < -1e-8:
                        feasible[t] = False
                    charging_updated[t] = np.maximum(charging_threshold[t], 0)
                    shifted_charging[t+1] = charging_should - charging_updated[t]

            SoC[t+1] = SoC[t] + charging_updated[t] - consumption[t]
            
            if (SoC[t+1] > SoC_max[t+1] + 1e-7):
                feasible[t] = False
                charging_updated[t] = np.maximum(SoC_max[t+1] - SoC[t] + consumption[t], 0)
                SoC[t+1] = SoC_max[t+1]
            elif (SoC[t+1] < SoC_min[t+1] - 1e-7):
                feasible[t] = False
                charging_updated[t] = np.maximum(SoC_min[t+1] - SoC[t] + consumption[t], 0)
                SoC[t+1] = SoC_min[t+1]

        charging_updated[-1] = np.minimum(charging_status_quo[-1] + shifted_charging[-1], charging_threshold[-1])

        return SoC, charging_updated, shifted_charging, check_max_min_SoC, feasible, charging_threshold
    
    def get_SoC_charging_v2g(self,
            ch_avail: np.ndarray,
            SoC_init:float,
            consumption: np.ndarray,
            target_load_ev: np.ndarray,
            SoC_max: np.ndarray,
            SoC_min: np.ndarray,
            charging_status_quo: np.ndarray) -> np.ndarray:
        
        charging_threshold = np.minimum(target_load_ev, ch_avail) 

        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        charging_updated = np.zeros(len(ch_avail)) 

        check_max_min_SoC = np.zeros(len(ch_avail))
        shifted_charging = np.zeros(len(ch_avail))
        feasible = np.ones(len(ch_avail), dtype=bool)

        for t in range(len(charging_status_quo)-1):
            charging_should = charging_status_quo[t] + shifted_charging[t]
            charging = np.minimum(charging_should, charging_threshold[t])
            SoC_state = SoC[t] + charging - consumption[t]
            check_max_min_SoC[t+1] = SoC_state - np.clip(SoC_state, SoC_min[t+1], SoC_max[t+1])
            if np.isclose(check_max_min_SoC[t+1], 0.0, atol=1e-7):
                charging_updated[t] = charging
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif SoC_state > SoC_max[t+1]:
                battery_capacity = SoC_max[t+1] - SoC[t]
                charging_updated[t] = battery_capacity + consumption[t]
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif SoC_state < SoC_min[t+1]:
                battery_capacity = SoC_min[t+1] - SoC[t]
                charging_updated[t] = np.minimum(battery_capacity + consumption[t], ch_avail[t])
                shifted_charging[t+1] = charging_should - charging_updated[t]

            SoC[t+1] = SoC[t] + charging_updated[t] - consumption[t]
            
            if (charging_updated[t] > charging_threshold[t]):
                feasible[t] = False

        charging_updated[-1] = np.minimum(charging_status_quo[-1] + shifted_charging[-1], charging_threshold[-1])
        return SoC, charging_updated, shifted_charging, check_max_min_SoC, feasible, charging_threshold
    
    def get_coordinate_SoC_charging(self,
            threshold:float,
            household_load: pd.DataFrame,
            charging_status_quo: dict[str, np.ndarray],
            soc_status_quo: dict[str, np.ndarray],
            soc_max: dict[str, np.ndarray],
            soc_min: dict[str, np.ndarray],
            ch_avail: dict[str, np.ndarray],
            consumption: dict[str, np.ndarray]):
        # sourcery skip: merge-dict-assign, move-assign-in-block
        # sourcery skip: merge-dict-assign, move-assign-in-block
        
        threshold = threshold - household_load

        charging_updated_dict = {}
        feasible_dict = {}
        threshold_dict = {}

        soc_updated_dict = {"uncontrolled": soc_status_quo.get("uncontrolled", 0)}
        charging_updated_dict["uncontrolled"] = charging_status_quo.get("uncontrolled", 0)
        feasible_dict["uncontrolled"] = charging_status_quo.get("uncontrolled") < threshold
        threshold_dict["uncontrolled"] = threshold
        threshold_dict["controlled_wo_v2g"] = threshold_dict["uncontrolled"] - charging_updated_dict["uncontrolled"]

        soc_init = soc_max.get("controlled_wo_v2g")[0]

        # controlled the ev wo v2g in the controll situation without v2g
        soc_updated, charging_updated, _, _, feasible, _ = self.get_SoC_charging_smart_charging(
            ch_avail=ch_avail.get("controlled_wo_v2g", 0),
            SoC_init=soc_init,
            consumption=consumption.get("controlled_wo_v2g", 0),
            target_load_ev=threshold_dict["controlled_wo_v2g"],
            SoC_max=soc_max.get("controlled_wo_v2g", 0),
            SoC_min=soc_min.get("controlled_wo_v2g", 0),
            charging_status_quo=charging_status_quo.get("controlled_wo_v2g", 0)
        )

        soc_updated_dict["controlled_wo_v2g"] = soc_updated
        charging_updated_dict["controlled_wo_v2g"] = charging_updated
        feasible_dict["controlled_wo_v2g"] = feasible

        threshold = threshold - charging_updated
        threshold_dict["controlled_v2g"] = threshold

        # ev group controlled_v2g in the situation controlling without v2g

        soc_init = soc_max.get("controlled_v2g")[0]
        SOC_updated, charging_updated, _, _, feasible, _ = self.get_SoC_charging_smart_charging(
            ch_avail=ch_avail.get("controlled_v2g"),
            SoC_init = soc_init,
            target_load_ev=threshold_dict["controlled_v2g"],
            consumption=consumption.get("controlled_v2g"),
            SoC_max=soc_max.get("controlled_v2g"),
            SoC_min=soc_min.get("controlled_v2g"),
            charging_status_quo=charging_status_quo.get("controlled_v2g"),
        )

        soc_updated_dict["controlled_v2g_wo"] = SOC_updated
        charging_updated_dict["controlled_v2g_wo"] = charging_updated
        feasible_dict["controlled_v2g_wo"] = feasible

        SOC_updated, charging_updated, _, _, feasible, _ = self.get_SoC_charging_v2g(
            ch_avail=ch_avail.get("controlled_v2g"),
            SoC_init=soc_init,
            charging_status_quo=charging_status_quo.get("controlled_v2g"),
            target_load_ev=threshold_dict.get("controlled_v2g"),
            consumption=consumption.get("controlled_v2g"),
            SoC_max=soc_max.get("controlled_v2g"),
            SoC_min=soc_min.get("controlled_v2g"),
        )
        soc_updated_dict["controlled_v2g"] = SOC_updated
        charging_updated_dict["controlled_v2g"] = charging_updated
        feasible_dict["controlled_v2g"] = feasible
        threshold_dict["controlled_v2g"] = threshold

        return soc_updated_dict, charging_updated_dict, feasible_dict, threshold_dict