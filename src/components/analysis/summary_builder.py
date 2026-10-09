"""Summary builder utilities for the EV/V2G analysis workflow."""

import re
import pandas as pd
import os

class SummaryBuilder:

    """Represent summary builder and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.cfg = scenario_config
        self.opt_dir = scenario_data.optimization_dir
        self.analysis_dir = scenario_data.analysis_dir

        self.pattern = re.compile(
                r"^result_"
                r"(?P<ch_refuel>[^_]+)_"                  # 0.9
                r"(?P<start_discharging_level>[^_]+)_"    # 25
                r"DEU_[0-9]{8}_"                          # DEU_20250514_
                r"(?P<region_name>[^_]+)_"                # R7
                r"(?P<commuting>.+?)_85_con_"             # home_non_commuting  (non-greedy up to _100_con_)
                r"(?P<n_controlled>[^_]+)_"               # 100
                r"(?P<controlled_peak>.+?)_from_"
                r"(?P<uncontrolled_peak>.+?)\.csv$"       # 0.113
            )

    def build(self):
        """Build .
        
        Returns
        -------
        object
            Result computed as ``df``."""
        rows = []
        for fn in os.listdir(self.opt_dir):
            m = self.pattern.match(fn)
            if not m:
                continue
            info = m.groupdict()

            # 类型转换
            info["ch_refuel"] = float(info["ch_refuel"])
            info["start_discharging_level"] = float(info["start_discharging_level"])
            info["n_controlled"] = float(info["n_controlled"])
            info["controlled_peak"] = float(info["controlled_peak"])
            info["uncontrolled_peak"] = float(info["uncontrolled_peak"])

            rows.append(info)

        df = pd.DataFrame(rows)

        df.to_csv(self.analysis_dir / "summary.csv", sep=";", index=False)
        print(f"[SummaryBuilder] summary written → {self.analysis_dir}, rows={len(df)}")
        return df
