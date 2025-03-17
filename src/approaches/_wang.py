import pandas as pd
import numpy as np
from src.utils._calculate_soc import calculate_SoC

class Wang():
    def __init__(self):
        super.__init__(self)

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
            a[t] = np.min(SoC_max[t] - SoC[t], ch_avail[t])
            b[t] = np.min(SoC[t] - SoC_min[t], ch_avail[t])
            if target_load_ev[t] == 0:
                charging[t] = 0
            elif target_load_ev[t] > 0:
                if np.abs(target_load_ev[t]) < b[t]:
                    charging[t] = -target_load_ev[t]
                else:
                    charging[t] = -b[t]
            else:
                # target_load_ev[t] < 0
                if np.abs(target_load_ev[t]) < a[t]:
                    charging[t] = np.abs(target_load_ev[t])
                else:
                    charging[t] = a[t]
            SoC[t+1] = np.max(SoC[t] + charging[t] - consumption[t], 0)
        
        return SoC, charging

            
                
        

    