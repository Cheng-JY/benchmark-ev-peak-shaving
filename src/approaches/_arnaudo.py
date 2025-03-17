import pandas as pd
import numpy as np

class Arnaudo():
    def __init__(self, start_charging, stop_charging, start_discharging):
        self.start_charging = start_charging
        self.stop_charging = stop_charging
        self.start_discharging = start_discharging

    def get_SoC_charging(self, 
                ch_avail: pd.Series,
                SoC_init: float,
                battery_capacity: float,
                consumption: pd.Series) -> np.ndarray:
        """
        if SoC < 50%, start charging, until 90% stop
        if SoC > 80%, start discharging, until 80% stop
        :return:
        """
        SoC =  SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        charging = np.zeros(len(ch_avail))

        for t in range(len(ch_avail)):
            if SoC[t] < self.start_charging * battery_capacity:
                charging[t] = np.min(ch_avail[t], battery_capacity - SoC[t])
            elif SoC[t] > self.stop_charging * battery_capacity:
                charging[t] = 0
            elif SoC[t] > self.start_discharging * battery_capacity:
                charging[t] = -np.min(ch_avail[t], SoC[t] - self.start_discharging * battery_capacity)
            elif self.start_charging * battery_capacity <= SoC[t] <= self.start_discharging_charging * battery_capacity:
                charging[t] = 0
            SoC[t+1] = np.max(SoC[t] + charging[t] - consumption[t], 0)
        
        return SoC, charging

