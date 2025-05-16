import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from ..base import AggregateCoordinateStrategy

class Proposed(AggregateCoordinateStrategy):
    def __init__(self, random_state=None, threshold: float = 1e-7):
        super().__init__(random_state)
        self.THRESHOLD = threshold

    def _common_charging_logic(
        self,
        ch_avail: np.ndarray,
        SoC_init: float,
        consumption: np.ndarray,
        target_load_ev: np.ndarray,
        SoC_max: np.ndarray,
        SoC_min: np.ndarray,
        charging_status_quo: np.ndarray,
        is_v2g: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        charging_threshold = np.minimum(target_load_ev, ch_avail)
        SoC = np.zeros(len(ch_avail))
        SoC[0] = SoC_init
        charging_updated = np.zeros(len(ch_avail))
        shifted_charging = np.zeros(len(ch_avail))
        check_max_min_SoC = np.zeros(len(ch_avail))
        feasible = np.ones(len(ch_avail), dtype=bool)

        for t in range(len(charging_status_quo) - 1):
            charging_should = charging_status_quo[t] + shifted_charging[t]
            charging = np.minimum(charging_should, charging_threshold[t])
            SoC_state = SoC[t] + charging - consumption[t]
            check_max_min_SoC[t + 1] = SoC_state - np.clip(SoC_state, SoC_min[t + 1], SoC_max[t + 1])

            if is_v2g:
                if np.isclose(check_max_min_SoC[t + 1], 0.0, atol=self.THRESHOLD):
                    charging_updated[t] = charging
                elif SoC_state > SoC_max[t + 1]:
                    charging_updated[t] = SoC_max[t + 1] - SoC[t] + consumption[t]
                elif SoC_state < SoC_min[t + 1]:
                    charging_updated[t] = np.minimum(SoC_min[t + 1] - SoC[t] + consumption[t], ch_avail[t])
                shifted_charging[t + 1] = charging_should - charging_updated[t]
                SoC[t + 1] = SoC[t] + charging_updated[t] - consumption[t]
                feasible[t] = charging_updated[t] <= charging_threshold[t]
            else:
                charging_temp = charging_should - check_max_min_SoC[t]
                if np.isclose(check_max_min_SoC[t + 1], 0.0, atol=self.THRESHOLD):
                    charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)
                elif check_max_min_SoC[t + 1] > 0:
                    charging_updated[t] = np.maximum(np.minimum(charging_temp, charging_threshold[t]), 0)
                elif check_max_min_SoC[t + 1] < 0:
                    charging_updated[t] = np.maximum(charging_threshold[t], 0)
                shifted_charging[t + 1] = charging_should - charging_updated[t]
                SoC[t + 1] = SoC[t] + charging_updated[t] - consumption[t]
                if SoC[t + 1] > SoC_max[t + 1] + self.THRESHOLD:
                    charging_updated[t] = SoC_max[t + 1] - SoC[t] + consumption[t]
                    SoC[t + 1] = SoC_max[t + 1]
                    feasible[t] = False
                elif SoC[t + 1] < SoC_min[t + 1] - self.THRESHOLD:
                    charging_updated[t] = SoC_min[t + 1] - SoC[t] + consumption[t]
                    SoC[t + 1] = SoC_min[t + 1]
                    feasible[t] = False

        charging_updated[-1] = np.minimum(charging_status_quo[-1] + shifted_charging[-1], charging_threshold[-1])
        return SoC, charging_updated, shifted_charging, check_max_min_SoC, feasible, charging_threshold

    def get_SoC_charging_smart_charging(
        self,
        ch_avail: np.ndarray,
        SoC_init: float,
        consumption: np.ndarray,
        target_load_ev: np.ndarray,
        SoC_max: np.ndarray,
        SoC_min: np.ndarray,
        charging_status_quo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._common_charging_logic(
            ch_avail,
            SoC_init,
            consumption,
            target_load_ev,
            SoC_max,
            SoC_min,
            charging_status_quo,
            is_v2g=False
        )

    def get_SoC_charging_v2g(
        self,
        ch_avail: np.ndarray,
        SoC_init: float,
        consumption: np.ndarray,
        target_load_ev: np.ndarray,
        SoC_max: np.ndarray,
        SoC_min: np.ndarray,
        charging_status_quo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._common_charging_logic(
            ch_avail,
            SoC_init,
            consumption,
            target_load_ev,
            SoC_max,
            SoC_min,
            charging_status_quo,
            is_v2g=True
        )

    def get_coordinate_SoC_charging(
        self,
        threshold: float,
        household_load: pd.DataFrame,
        charging_status_quo: Dict[str, np.ndarray],
        soc_status_quo: Dict[str, np.ndarray],
        soc_max: Dict[str, np.ndarray],
        soc_min: Dict[str, np.ndarray],
        ch_avail: Dict[str, np.ndarray],
        consumption: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        threshold -= household_load
        soc_updated_dict = {"uncontrolled": soc_status_quo.get("uncontrolled", 0)}
        charging_updated_dict = {"uncontrolled": charging_status_quo.get("uncontrolled", 0)}
        feasible_dict = {"uncontrolled": charging_status_quo.get("uncontrolled") < threshold}
        threshold_dict = {"uncontrolled": threshold}
        threshold_dict["controlled_wo_v2g"] = threshold - charging_updated_dict["uncontrolled"]

        soc_init = soc_max["controlled_wo_v2g"][0]
        soc_updated, charging_updated, _, _, feasible, _ = self.get_SoC_charging_smart_charging(
            ch_avail["controlled_wo_v2g"], soc_init, consumption["controlled_wo_v2g"],
            threshold_dict["controlled_wo_v2g"], soc_max["controlled_wo_v2g"], soc_min["controlled_wo_v2g"],
            charging_status_quo["controlled_wo_v2g"]
        )
        soc_updated_dict["controlled_wo_v2g"] = soc_updated
        charging_updated_dict["controlled_wo_v2g"] = charging_updated
        feasible_dict["controlled_wo_v2g"] = feasible

        threshold -= charging_updated
        threshold_dict["controlled_v2g"] = threshold

        soc_init = soc_max["controlled_v2g"][0]
        soc_updated, charging_updated, _, _, feasible, _ = self.get_SoC_charging_smart_charging(
            ch_avail["controlled_v2g"], soc_init, consumption["controlled_v2g"],
            threshold_dict["controlled_v2g"], soc_max["controlled_v2g"], soc_min["controlled_v2g"],
            charging_status_quo["controlled_v2g"]
        )
        soc_updated_dict["controlled_v2g_wo"] = soc_updated
        charging_updated_dict["controlled_v2g_wo"] = charging_updated
        feasible_dict["controlled_v2g_wo"] = feasible

        soc_updated, charging_updated, _, _, feasible, _ = self.get_SoC_charging_v2g(
            ch_avail["controlled_v2g"], soc_init, consumption["controlled_v2g"],
            threshold_dict["controlled_v2g"], soc_max["controlled_v2g"], soc_min["controlled_v2g"],
            charging_status_quo["controlled_v2g"]
        )
        soc_updated_dict["controlled_v2g"] = soc_updated
        charging_updated_dict["controlled_v2g"] = charging_updated
        feasible_dict["controlled_v2g"] = feasible

        return soc_updated_dict, charging_updated_dict, feasible_dict, threshold_dict