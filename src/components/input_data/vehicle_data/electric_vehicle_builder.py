"""Electric vehicle builder utilities for the EV/V2G analysis workflow."""

from .driving_profile_generator import DrivingProfileGenerator
from .mix_profile_builder import MixProfileBuilder
from .soc_profile_builder import SOCProfileBuilder

class EVProfileBuilder:

    """Represent evprofile builder and its associated operations."""
    def __init__(self, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        self.scenario_data = scenario_data
        self.ev_dir = scenario_data.ev_dir
        self.cfg = scenario_config

    def build(self,
              net_pop_data,
              filters,
              ch_refuel_list,
              start_discharging_list,
              n_jobs=6,
    ):
        """Build .
        
        Parameters
        ----------
        net_pop_data : object
            Net pop data used by this operation.
        filters : object
            Filters used by this operation.
        ch_refuel_list : object
            Ch refuel list used by this operation.
        start_discharging_list : object
            Start discharging list used by this operation.
        n_jobs : object
            N jobs used by this operation."""
        driving_profile_generator = DrivingProfileGenerator(self.scenario_data, self.cfg)
        driving_profile_generator.generate_profiles(net_pop_data,
                                           filters,
                                           )
        
        mix_profile_builder = MixProfileBuilder(self.scenario_data, self.cfg)
        mix_profile_builder.generate(
            mob_groups_short=driving_profile_generator.mob_group_list,
            ch_refuel_list=ch_refuel_list,
            veh_per_region=10000,
            njobs=n_jobs
        )

        soc_profile_builder = SOCProfileBuilder(self.scenario_data, self.cfg)
        soc_profile_builder.generate(
            ch_refuel_list=ch_refuel_list,
            start_discharging_list=start_discharging_list,
            total_parts=1,
            njobs=n_jobs,
        )