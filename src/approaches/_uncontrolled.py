import pandas as pd
import numpy as np

class Uncontrolled():
    def __init__(self):
        super.__init__(self)

    def get_SoC(self, 
                ch_avail: pd.Series, 
                SoC_init: float,
                charging: pd.Series, 
                consumption: pd.Series) -> np.ndarray:
        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        for t in range(1, len(ch_avail)):
            SoC[t] = np.max(SoC[t-1] + charging[t-1] - consumption[t-1], 0)
        return SoC

