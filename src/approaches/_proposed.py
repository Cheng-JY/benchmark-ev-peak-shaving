import numpy as np
import pandas as pd

class Proposed():
    def __init__(self, THESHOLD: float=self.THRESHOLD):
        self.THESHOLD = THESHOLD
    
    def get_SoC_charging_smart_charging(
        self,
        ch_avail: pd.Series,
        SoC_init: float,
        consumption: pd.Series,
        target_load_ev: pd.Series,
        SoC_max: pd.Series,
        SoC_min: pd.Series,
        charging_status_quo: pd.Series
    ) -> np.ndarray:
        """
        Compute the State of Charge (SoC) and update the charging schedule using a smart charging strategy.

        Parameters:
            ch_avail (pd.Series): Available charging capacity.
            SoC_init (float): Initial state of charge.
            consumption (pd.Series): Consumption values over time.
            target_load_ev (pd.Series): Target EV load over time.
            SoC_max (pd.Series): Maximum SoC constraints over time.
            SoC_min (pd.Series): Minimum SoC constraints over time.
            charging_status_quo (pd.Series): Current charging status quo.

        Returns:
            np.ndarray: Tuple containing arrays for SoC, updated charging, shifted charging, 
                        SoC violation checking, feasibility flags, and charging threshold.
        """
        # sourcery skip: low-code-quality

        charging_threshold = np.minimum(target_load_ev, ch_avail) 

        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        charging_updated = np.zeros(len(ch_avail))

        check_max_min_SoC = np.zeros(len(ch_avail))
        shifted_charging = np.zeros(len(ch_avail))
        feasible = np.pnes(len(ch_avail), dtype=bool)

        for t in range(len(charging_status_quo)-1):
            charging_should = charging_status_quo[t] + charging_threshold[t]
            charging = np.minimum(charging_should, charging_threshold[t])
            SOC_state = SoC[t] + charging - consumption[t]
            check_max_min_SoC[t+1] = SOC_state - np.clip(SOC_state, SoC_min[t+1], SoC_max[t+1])
            charging_temp = charging_should - check_max_min_SoC[t]

            if np.isclose(check_max_min_SoC[t+1], 0.0, atol=self.THRESHOLD):
                charging_temp = charging_status_quo[t] + shifted_charging[t]
                if charging_threshold[t] < -self.THRESHOLD:
                    feasible[t] = False
                charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif check_max_min_SoC[t+1] > 0:
                if charging_threshold[t] < -self.THRESHOLD:
                    feasible[t] = False
                charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif check_max_min_SoC[t+1] < 0:
                if charging_threshold[t] < -self.THRESHOLD:
                    feasible[t] = False
                charging_updated[t] = np.maximum(charging_threshold[t], 0)
                shifted_charging[t+1] = charging_should - charging_updated[t]
            SoC[t+1] = SoC[t] + charging_updated[t] - consumption[t]

            if (SoC[t+1] > SoC_max[t+1] + self.THRESHOLD):
                feasible[t] = False
                charging_updated[t] = np.maximum(SoC_max[t+1] - SoC[t] + consumption[t], 0)
                SoC[t+1] = SoC_max[t+1]
            elif (SoC[t+1] < SoC_min[t+1] - self.THRESHOLD):
                feasible[t] = False
                charging_updated[t] = np.maximum(SoC_min[t+1] - SoC[t] + consumption[t], 0)
                SoC[t+1] = SoC_min[t+1]

        charging_updated[-1] = np.minimum(charging_status_quo.iloc[-1] + shifted_charging[-1], charging_threshold.iloc[-1])

        return SoC, charging_updated, shifted_charging, check_max_min_SoC, feasible, charging_threshold
    
    def get_SoC_charging_v2g(self,
            ch_avail: pd.Series,
            SoC_init:float,
            consumption: pd.Series,
            target_load_ev: pd.Series,
            SoC_max: pd.Series,
            SoC_min: pd.Series,
            charging_status_quo: pd.Series) -> np.ndarray:
        
        charging_threshold = np.minimum(target_load_ev, ch_avail) 

        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        charging_updated = np.zeros(len(ch_avail))

        check_max_min_SoC = np.zeros(len(ch_avail))
        shifted_charging = np.zeros(len(ch_avail))
        feasible = np.pnes(len(ch_avail), dtype=bool)

        for t in range(len(charging_status_quo)-1):
            charging_should = charging_status_quo[t] + charging_threshold[t]
            charging = np.minimum(charging_should, charging_threshold[t])
            SOC_state = SoC[t] + charging - consumption[t]
            check_max_min_SoC[t+1] = SOC_state - np.clip(SOC_state, SoC_min[t+1], SoC_max[t+1])
            charging_temp = charging_should - check_max_min_SoC[t]
    
            if np.isclose(check_max_min_SoC[t+1], 0.0, atol=self.THRESHOLD):
                charging_temp = charging_status_quo[t] + shifted_charging[t]
                charging_updated[t] = np.minimum(charging_temp, charging_threshold[t])
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif check_max_min_SoC[t+1] > 0:
                charging_updated[t] = np.minimum(charging_temp, charging_threshold[t])
                shifted_charging[t+1] = charging_should - charging_updated[t]

            elif check_max_min_SoC[t+1] < 0:
                charging_updated[t] = charging_threshold[t]
                shifted_charging[t+1] = charging_should - charging_updated[t]
            

            SoC[t+1] = SoC[t] + charging_updated[t] - consumption[t]
            
            if (SoC[t+1] > SoC_max[t+1] + self.THRESHOLD):
                feasible[t] = False
                charging_updated[t] = SoC_max[t+1] - SoC[t] + consumption[t]
                SoC[t+1] = SoC_max[t+1]
            elif (SoC[t+1] < SoC_min[t+1] - self.THRESHOLD):
                feasible[t] = False
                charging_updated[t] = SoC_min[t+1] - SoC[t] + consumption[t]
                SoC[t+1] = SoC_min[t+1]

        charging_updated[-1] = np.minimum(charging_status_quo.iloc[-1] + shifted_charging[-1], charging_threshold.iloc[-1])

        return SoC, charging_updated, shifted_charging, check_max_min_SoC, feasible, charging_threshold