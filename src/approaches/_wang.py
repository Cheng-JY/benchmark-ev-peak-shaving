import pandas as pd
import numpy as np
from src.utils._calculate_soc import calculate_SoC

class Wang():
    def __init__(self):
        pass

    def get_SoC_charging(
            self,
            ch_avail: pd.Series, 
            SoC_init: float,
            consumption: pd.Series,
            target_load_ev: pd.Series,
            SoC_max: pd.Series,
            SoC_min: pd.Series) -> np.ndarray:
        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        a = np.zeros(len(ch_avail))
        b = np.zeros(len(ch_avail))
        charging = np.zeros(len(ch_avail))

        for t in range(len(ch_avail)):
            a[t] = np.minimum(SoC_max[t] - SoC[t], ch_avail[t])
            b[t] = np.minimum(SoC[t] - SoC_min[t], ch_avail[t])
            if target_load_ev[t] == 0:
                charging[t] = 0
            elif target_load_ev[t] > 0:
                charging[t] = (
                    -np.minimum(np.abs(target_load_ev[t]), ch_avail[t])
                    if np.abs(target_load_ev[t]) < b[t]
                    else -np.minimum(b[t], ch_avail[t])
                )
            elif np.abs(target_load_ev[t]) < a[t]:
                charging[t] = np.minimum(np.abs(target_load_ev[t]), ch_avail[t])
            else:
                charging[t] = np.minimum(a[t], ch_avail[t])
            SoC[t+1] = np.maximum(SoC[t] + charging[t] - consumption[t], 0)

        return SoC, charging

            
                
        

    