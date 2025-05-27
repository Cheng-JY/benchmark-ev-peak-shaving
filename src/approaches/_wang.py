import pandas as pd
import numpy as np
from src.utils._calculation import calculate_SoC
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
            charging_status_quo: np.ndarray,
            SoC_max: np.ndarray,
            SoC_min: np.ndarray) -> np.ndarray:
        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        a = np.zeros(len(ch_avail))
        b = np.zeros(len(ch_avail))
        charging = np.zeros(len(ch_avail))

        for t in range(len(ch_avail)-1):
            a[t] = np.minimum(SoC_max[t] - SoC[t], ch_avail[t])
            b[t] = np.minimum(SoC[t] - SoC_min[t], ch_avail[t])
            diff = charging_status_quo[t] - target_load_ev[t]
            if np.isclose(diff, 0.0, atol=1e-7):
                charging[t] = charging_status_quo[t]
            elif diff > 0:
                charging[t] = (
                    -np.minimum(np.abs(target_load_ev[t]), ch_avail[t])
                    if np.abs(target_load_ev[t]) < b[t]
                    else -b[t]
                )
            else:
                # target_load_ev[t] < 0
                if np.abs(diff) < a[t]:
                    charging[t] = np.minimum(np.abs(target_load_ev[t]), ch_avail[t])
                else:
                    charging[t] = np.minimum(a[t], ch_avail[t])
            SoC[t+1] = np.maximum(SoC[t] + charging[t] - consumption[t], 0)
        
        return SoC, charging

    def get_coordinate_SoC_charging(self,
            threshold:float,
            household_load: pd.DataFrame,
            charging_status_quo: dict[str, np.ndarray],
            soc_status_quo: dict[str, np.ndarray],
            soc_max: dict[str, np.ndarray],
            soc_min: dict[str, np.ndarray],
            ch_avail: dict[str, np.ndarray],
            consumption: dict[str, np.ndarray]):
        
        threshold = threshold - household_load

        charging_updated_dict = {}
        feasible_dict = {}
        threshold_dict = {}

        soc_updated_dict = {"uncontrolled": soc_status_quo.get("uncontrolled", 0)}
        charging_updated_dict["uncontrolled"] = charging_status_quo.get("uncontrolled", 0)
        feasible_dict["uncontrolled"] = charging_status_quo.get("uncontrolled") < threshold
        threshold_dict["uncontrolled"] = threshold
        threshold_dict["controlled_v2g"] = threshold_dict["uncontrolled"] - charging_updated_dict["uncontrolled"]

        soc_init = soc_max.get("controlled_v2g")[0]
        SOC_updated, charging_updated = self.get_SoC_charging(
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
        feasible_dict["controlled_v2g"] = charging_updated < threshold_dict["controlled_v2g"]
        threshold_dict["controlled_v2g"] = threshold

        return soc_updated_dict, charging_updated_dict, feasible_dict, threshold_dict

            
                
        

    