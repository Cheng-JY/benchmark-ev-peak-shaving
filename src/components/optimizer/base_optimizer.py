"""Base optimizer utilities for the EV/V2G analysis workflow."""

from abc import ABC, abstractmethod
import pandas as pd
from ...utils import get_state

class BaseEVOptimizer(ABC):

    """Represent base evoptimizer and its associated operations."""
    def __init__(self, optimizer_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        optimizer_config : object
            Optimizer config used by this operation."""
        self.opt_cfg = optimizer_config
        self.model = None
    
    @abstractmethod
    def build_model(self, vehicle):
        """Build model.
        
        Parameters
        ----------
        vehicle : object
            Vehicle used by this operation."""
        pass

    @abstractmethod
    def optimize(self):
        """Optimize ."""
        pass
    
    @abstractmethod
    def export_results(self):
        """Export results."""
        pass

    def get_load(self):
        """Return load.
        
        Returns
        -------
        object
            Result computed as ``df_combined['total_MWh']``."""
        hh_fp = self.scenario_data.household_dir / "household_load.csv"
        hp_fp = self.scenario_data.heat_pump_dir / "heat_pump_profiles.csv"

        df_hp = pd.read_csv(hp_fp, 
                        sep=";", 
                        index_col=0,
                        )
        df_hp.index = pd.to_datetime(df_hp.index)

        df_hp["hp_MWh"] = df_hp["total_all"] * 0.25/1000

        df_hh = pd.read_csv(hh_fp, 
                            sep=";", 
                            index_col=0,
        )
        df_hh.index = pd.to_datetime(df_hh.index)
        df_hh["hh_MWh"] = df_hh["hh_load (MW)"] * 0.25

        df_combined = pd.DataFrame(index=pd.date_range(
            start='2012-01-01', periods=len(df_hh), freq='15min'))
        df_combined["hp_MWh"] = df_hp["hp_MWh"].values
        df_combined["hh_MWh"] = df_hh["hh_MWh"].values
        df_combined["total_MWh"] = df_hh["hh_MWh"] + df_hp["hp_MWh"]

        return df_combined["total_MWh"]

    # @JC need to adjust
    def get_ev_load(self, filename):

        """Return ev load.
        
        Parameters
        ----------
        filename : object
            Filename used by this operation.
        
        Returns
        -------
        object
            Result computed as ``(ch_avail, soc_max, soc_min, charging_status_quo, consumption)``."""
        ev_fp = self.scenario_data.aggregation_dir / filename
        aggregated_profile = pd.read_csv(ev_fp, sep=";")
        ch_avail, soc_max, soc_min, charging_status_quo, consumption = get_state(
            aggregated_profile)
        return ch_avail, soc_max, soc_min, charging_status_quo, consumption

