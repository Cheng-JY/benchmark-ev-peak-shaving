"""Optimization config utilities for the EV/V2G analysis workflow."""

class OptimizationConfig:
    """Represent optimization config and its associated operations."""
    def __init__(self):

        # Set input data
        """Initialize the instance using the supplied configuration and data."""
        self.ac_energy_price_year = 2021 #2021
        self.ambient_temperature_year = 2021
        self.mobility_data_regional_type = 'R3'
        self.large_price_feedin_spread = False
        self.minimum_power_ac_charger_kw = 0.3

        # Set charging scenarios to compute. Options: 'basic', 'solar', 'bidirectional'
        self.charging_scenarios = ['bidirectional']
        self.ac_charging_strategy = 'ch_avail_scenario_1'
        self.amount_of_scenarios = 100 # default: 100
        self.intraday_auction = True
        self.adjusted_feedin_prices = False

        # Vehicle Selection
        self.vehicle_name = 'tesla_model_3_lr'

        # Export filepath
        self.h5_export_filepath = None

        # Optimization Parameters
        self.vehicle_initial_soc = 1.0
        self.vehicle_minimum_soc = 0.5
        self.vehicle_terminal_soc = 0.95
        self.lambda_minimum_soc = 100
        self.lambda_terminal_soc = 1000 #1000
        self.lambda_dc_charging_duration = 1000
        self.maximum_ac_consumption_grid_kw = 22.0
        self.minimum_ac_consumption_grid_kw = -22.0
        self.ac_charging_price_daily_surcharge_percent = 0.5 #0.5
        self.ac_feedin_price_daily_discount_percent = 0.5 #0.5
        self.solver = 'HiGHS'
        self.loss_linearization_forbidden_penalty_factor = 100
        self.loss_linearization_alpha_0 = 0.1440966 #[kW]
        self.loss_linearization_alpha_1 = 0.03754
        self.constraint_epsilon = 1e-8
        self.constraint_bigM = 1e5
        self.calender_weeks_to_optimize = 50 #50
        self.logging_enabled = False
        

    def __repr__(self):
        """Return the textual representation of the instance.
        
        Returns
        -------
        object
            Result computed as ``'Optimization Config'``."""
        return "Optimization Config"
    
    @property
    def controlled_name(self):
        """Controlled name.
        
        Returns
        -------
        object
            Result computed as ``self._controlled_name``."""
        return self._controlled_name

    @controlled_name.setter
    def controlled_name(self, name):
        """Controlled name.
        
        Parameters
        ----------
        name : object
            Name used by this operation."""
        self._controlled_name = name