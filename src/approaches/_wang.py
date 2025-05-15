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
            SoC_max: np.ndarray,
            SoC_min: np.ndarray) -> np.ndarray:
        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        a = np.zeros(len(ch_avail))
        b = np.zeros(len(ch_avail))
        charging = np.zeros(len(ch_avail))
        feasible = np.ones(len(ch_avail), dtype=bool)

        for t in range(len(ch_avail)):
            a[t] = np.minimum(SoC_max[t] - SoC[t], ch_avail[t])
            b[t] = np.minimum(SoC[t] - SoC_min[t], ch_avail[t])
            if target_load_ev[t] == 0:
                charging[t] = 0
            elif target_load_ev[t] > 0:
                charging[t] = (
                    -np.minimum(np.abs(target_load_ev[t]), ch_avail[t])
                    if np.abs(target_load_ev[t]) < b[t]
                    else -b[t]
                )
            elif np.abs(target_load_ev[t]) < a[t]:
                charging[t] = np.abs(target_load_ev[t])
            else:
                charging[t] = a[t]
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
            consumption: dict[str, np.ndarray]
    ):
        pass

            
                
        

    