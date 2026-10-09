"""Milp basic optimizer utilities for the EV/V2G analysis workflow."""

from .base_optimizer import BaseEVOptimizer
import pandas as pd
import numpy as np
import pyomo.environ as pyo
# from pyomo.util.infeasible import log_infeasible_constraints # 如果需要调试可以解开
import logging

from ...utils import get_state
from ...utils._data import read_price

class PyomoBasicEVOptimizer(BaseEVOptimizer):

    """Represent pyomo basic evoptimizer and its associated operations."""
    def __init__(self, opt_config, scenario_data, scenario_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        opt_config : object
            Opt config used by this operation.
        scenario_data : object
            Scenario data used by this operation.
        scenario_config : object
            Scenario config used by this operation."""
        super().__init__(opt_config)

        self.cfg = scenario_config
        self.scenario_data = scenario_data

        self._model = None
    
    def _prepare_optimization_dataframe(self, 
                                        load_series, 
                                        uncontrolled_ev_charging, 
                                        controlled_consumption, 
                                        controlled_ch_avail,
                                        controlled_soc_max,
                                        controlled_soc_min,):
        """Prepare optimization dataframe.
        
        Parameters
        ----------
        load_series : object
            Load series used by this operation.
        uncontrolled_ev_charging : object
            Uncontrolled ev charging used by this operation.
        controlled_consumption : object
            Controlled consumption used by this operation.
        controlled_ch_avail : object
            Controlled ch avail used by this operation.
        controlled_soc_max : object
            Controlled soc max used by this operation.
        controlled_soc_min : object
            Controlled soc min used by this operation.
        
        Returns
        -------
        object
            Result computed as ``df``."""
        df = pd.DataFrame(index=load_series.index)

        df["household_consumption_MWh"] = (load_series.values + uncontrolled_ev_charging)
        df["ambient_temperature_degC"] = 20 # from the climate scenario data
        df['maximum_ac_charging_MWh'] = controlled_ch_avail * 0.25
        price_ct, price_euro = read_price()
        df["price_ac_consumption_euro_MWh"] = price_euro.values * 1000
        df["driving_consumption_MWh"] = controlled_consumption
        df['maximum_soc_MWh'] = controlled_soc_max
        df["minimum_soc_MWh"] = controlled_soc_min

        return df
    
    def build_model(self, vehicle):
        """Build model.
        
        Parameters
        ----------
        vehicle : object
            Vehicle used by this operation."""
        self.vehicle = vehicle

    def _build_model(self, input_data, initial_soc=None, terminal_soc=None):
        """
        Builds the core structure of the Mixed Integer Linear Programming (MILP) model.
        Includes: Sets, Parameters, and Variables.
        """
        # 1. Create Model Container
        model = pyo.ConcreteModel(name='EV Charging Basic MILP')

        # 2. Define Sets (Time Horizon)
        # Corresponds to the number of rows in input_data
        model.T = pyo.Set(initialize=list(range(len(input_data))), ordered=True)

        # 3. Define Parameters (Mapped from input_data DataFrame to Pyomo)
        # Price Information
        model.price_cons_ac = list(input_data["price_ac_consumption_euro_MWh"])

        # Physical Constraint Series (Time-dependent)
        model.hh_cons_MWh = list(input_data['household_consumption_MWh'])
        model.max_charging_ac_MWh = list(input_data['maximum_ac_charging_MWh'])
        model.min_charging_ac_MWh = list(np.zeros(len(input_data)))
        model.driving_consumption_MWh = list(input_data['driving_consumption_MWh'])

        # Battery & Vehicle Static Parameters & SoC State Parameter
        model.soc_max = list(input_data['maximum_soc_MWh'])
        model.soc_min = list(input_data['minimum_soc_MWh'])
        model.soc_ini = pyo.Param(initialize=initial_soc if initial_soc is not None else model.soc_max[0],
                                  doc='Initial SoC', within=pyo.Reals)
        model.soc_term = pyo.Param(initialize=terminal_soc if terminal_soc is not None else model.soc_max[-1], 
                                   doc='Target Terminal SoC')
        bc_max = np.max(model.soc_max)
        model.bc_max = pyo.Param(initialize=bc_max,
                                 doc='Total EV battery capacity in kWh')

        # Power Limit Parameters (kW)
        model.emax_vehicle_ac = pyo.Param(initialize=np.max(model.max_charging_ac_MWh), 
                                           doc="EV's maximum AC charged energy")
        model.emin_vehicle_ac = pyo.Param(initialize=0, doc='EVs minimum AC charging power.')
        
        etarget = np.max(model.hh_cons_MWh + model.max_charging_ac_MWh)
        print(etarget)
        model.emax_grid_ac = pyo.Param(initialize=etarget,
                                       doc='Maximum AC power consumption from grid')
        model.emin_grid_ac = pyo.Param(initialize=self.opt_cfg.minimum_ac_consumption_grid_kw/1000,
                                       doc='Minimum AC power consumption from grid.')
        # 4. Define Decision Varaibless (The unknows we want to solve)

        # Power Variables
        model.energy_vehicle_ac_charging = pyo.Var(model.T, 
                                                  bounds=(0, model.emax_vehicle_ac),
                                                  doc='AC charging power at time t, connected to wallbox, during timeslot t')
        model.energy_grid_ac_consumption = pyo.Var(model.T,
                                                  bounds=(0, model.emax_grid_ac),
                                                  doc='Total grid consumption at time t')
        
        model.charging_idle_losses = pyo.Var(model.T, bounds=(0, None),
                                             doc='Power losses AC charging and vehicle idle')
        # Energy State Varible (kWh)
        model.soc = pyo.Var(model.T,
                            bounds=(None, model.bc_max),
                            doc='Battery energy content in MWh at time t')
        
        # Penalty Variables (Soft Constraints)
        model.terminal_soc_deviation = pyo.Var(bounds=(0, None),
                                               doc='Energy missing from target SoC at the end of simulation')
        
        # Comfort Penalty Switch: Becomes 1 if SoC drops below minimum
        model.comfort_binary = pyo.Var(model.T, domain=pyo.Binary, initialize=0,
                                       doc='Comfort binary value becomes 1 once SoC falls below soc_min to increment'
                                        'penalty.')
        
        model.comfort_binary_2 = pyo.Var(model.T, domain=pyo.Binary, initialize=0,
                                       doc='Comfort binary value becomes 1 once SoC falls over soc_max to increment'
                                        'penalty.')
        
        # for idle losses
        #Binary for intervall [0, epsilon], epsilon=1e-3
        model.ac_charging_threshold_binary_1 = pyo.Var(model.T, domain=pyo.Binary, initialize=0)
        #Binary for intervall [epsilon, 2.0], epsilon=1e-3
        model.ac_charging_threshold_binary_2 = pyo.Var(model.T, domain=pyo.Binary, initialize=0)
        #Binary for intervall [2.0, pmax]
        model.ac_charging_threshold_binary_3 = pyo.Var(model.T, domain=pyo.Binary, initialize=0)

        # Save model to self
        self._model = model

        print("Pyomo Variables & Parameters defined successfully.")
    
    # ==========================================
    # Pyomo Modeling: Constraints (Extracted)
    # ==========================================

    def _battery_capacity_update_constraint(self, model, t):
        """
        Updates the State of Charge (SoC).
        Equation (16) & (17) simplified: No AC loss term.
        """
        # AC Energy in = Power * 0.25h (Assuming 100% efficiency for AC as requested)
        ac_energy = (model.energy_vehicle_ac_charging[t] - model.charging_idle_losses[t])
        driving = model.driving_consumption_MWh[t]
        if t == model.T.first():
            current_energy = model.soc_ini + ac_energy
        else:
            current_energy = model.soc[t-1] + ac_energy
        return model.soc[t] == (current_energy - driving)
    
    def _grid_balancing_constraint(self, model, t):
        """
        Grid Balance Equation (8).
        Grid Power = EV Charging + Household Load.
        """
        return (model.energy_grid_ac_consumption[t] == 
                (model.energy_vehicle_ac_charging[t] + model.hh_cons_MWh[t] + model.charging_idle_losses[t]))

    def _minimum_soc_constraint(self, model, t):
        """
        Soft constraint for minimum SoC using Big-M.
        """
        bigM = self.opt_cfg.constraint_bigM
        # If comfort_binary is 1, the constraint is relaxed (penalized in objective).
        return (model.soc[t] - model.soc_min[t]) >= -bigM * model.comfort_binary[t]

    def _maximum_soc_constraint(self, model, t):
        """
        Soft constraint for minimum SoC using Big-M.
        """
        bigM = self.opt_cfg.constraint_bigM
        # If comfort_binary is 1, the constraint is relaxed (penalized in objective).
        return (model.soc_max[t] - model.soc[t]) >= -bigM * model.comfort_binary_2[t]
    
    def _terminal_soc_deviation_constraint(self, model, t):
        """
        Calculates missing energy at the last timestep.
        """
        last_t = model.T.last()
        return model.terminal_soc_deviation >= (model.soc_term - model.soc[last_t])
    
    def _charging_idle_losses_constraint(self, model, t):
        """Charging idle losses constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.charging_idle_losses[t] == model.energy_vehicle_ac_charging[t] * 0.01``."""
        optimization_config = self.opt_cfg
        """
        1.2 kW (1-phasig): 91% // 2.3 kW (1-phasig): 95,2% // 3.6 kW (1-phasig): 95,7% // 7.2 kW (2-phasig): 95,7%
        11 kW (3-phasig): 95,7% // 15.6 kW (3-phasig): 95.9% // 22 kW (3-phasig): 96.2%

        losses computation (only charger):
        1.2 kW = 108.0 W // 2.3 kW = 110.4 W // 3.6 kW = 154.8 W // 7.2 kW = 309.6 W // 11 kW = 473.0 W //
        15.6 kW = 639.6 W // 22 kW = 836.0 W

        piecewise linear fit only charger: 0.0440966 + 0.0369272 * x
        piecewise linear fit charger + 100 W base consumption: 0.1440966 + 0.03754 * x // 100 W come from cooling pump,
        communication controller, LV supply OBC and pulsed operation HV LV DCDC. (can be also calculated with 200 W total).
        """
        # alpha_0 = optimization_config.loss_linearization_alpha_0
        # alpha_1 = optimization_config.loss_linearization_alpha_1
        # penalty = optimization_config.loss_linearization_forbidden_penalty_factor

        # return model.charging_idle_losses[t] == (alpha_0 * model.ac_charging_threshold_binary_3[t] +
        #                                          alpha_1 * model.energy_vehicle_ac_charging[t])
        return model.charging_idle_losses[t] == model.energy_vehicle_ac_charging[t] * 0.01

    def _ac_charging_threshold_binary_1_constraint(self, model, t):
        # Intervall (0, epsilon)
        """Ac charging threshold binary 1 constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        optimization_config = self.opt_cfg
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac > epsilon --> binary == 0
        # if ac <= epsilon --> binary 0 or 1
        return (model.energy_vehicle_ac_charging[t] - epsilon) <= bigM * (1-model.ac_charging_threshold_binary_1[t])

    def _ac_charging_threshold_binary_2_constraint(self, model, t):
        # Intervall (epsilon, 2.0)
        """Ac charging threshold binary 2 constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        optimization_config = self.opt_cfg
        minimum_energy_ac_charger_MW = optimization_config.minimum_power_ac_charger_kw * 0.25 /1000
        bigM = optimization_config.constraint_bigM
        # if ac > 2.0 --> binary == 0
        # if ac <= 2.0 --> binary 0 or 1
        return (model.energy_vehicle_ac_charging[t] - minimum_energy_ac_charger_MW) <= bigM * (1-model.ac_charging_threshold_binary_2[t])

    def _ac_charging_threshold_binary_3_constraint(self, model, t):
        # Intervall (epsilon, 2.0)
        """Ac charging threshold binary 3 constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        optimization_config = self.opt_cfg
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac < epsilon --> binary == 0
        # if ac >= epsilon --> binary 0 or 1
        return (epsilon - model.energy_vehicle_ac_charging[t]) <= bigM * (1-model.ac_charging_threshold_binary_2[t])

    def _ac_charging_threshold_binary_4_constraint(self, model, t):
        # Intervall [2.0, :]
        """Ac charging threshold binary 4 constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        optimization_config = self.opt_cfg
        minimum_energy_ac_charger_MW = optimization_config.minimum_power_ac_charger_kw * 0.25 /1000
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac < 2.0 --> binary == 0
        # if ac >= 2.0 --> binary 0 or 1
        return (minimum_energy_ac_charger_MW - model.energy_vehicle_ac_charging[t]) <= bigM * (1-model.ac_charging_threshold_binary_3[t])

    def _ac_charging_threshold_binary_5_constraint(self, model, t):
        # Intervall [2.0, :]
        """Ac charging threshold binary 5 constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        optimization_config = self.opt_cfg
        minimum_energy_ac_charger_MW = optimization_config.minimum_power_ac_charger_kw * 0.25 /1000
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac >= 2.0 --> binary == 1
        return (model.energy_vehicle_ac_charging[t] - minimum_energy_ac_charger_MW) <= bigM * model.ac_charging_threshold_binary_3[t]

    def _ac_charging_total_threshold_constraint(self, model, t):
        """Ac charging total threshold constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        return (model.ac_charging_threshold_binary_1[t] + model.ac_charging_threshold_binary_2[t] + model.ac_charging_threshold_binary_3[t] == 1)
    
    def _max_ac_charging_power_constraint(self, model, t):
        """Hard limit on AC charging power."""
        return model.energy_vehicle_ac_charging[t] <= model.max_charging_ac_MWh[t]
    
    def _final_soc_constraint(self, model):
        """Final soc constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.soc[last_t] >= target_soc``."""
        last_t = model.T.last()
        target_soc = 0.2 * model.bc_max
        return model.soc[last_t] >= target_soc

    def _get_model_objective(self, model):
        """
        Defines the minimization objective: Cost + Penalties.
        """
        # 1. Energy Costs
        cost_ac = sum(model.energy_grid_ac_consumption[t] * model.price_cons_ac[t] for t in model.T)
        
        # 2. Penalties
        lambda_soc = self.opt_cfg.lambda_minimum_soc
        lambda_term = self.opt_cfg.lambda_terminal_soc
        
        penalty_soc = sum(model.comfort_binary[t] for t in model.T)
        penalty_soc_2 = sum(model.comfort_binary_2[t] for t in model.T)
        penalty_term = model.terminal_soc_deviation
        penalty_forbidden = sum(model.ac_charging_threshold_binary_2[t] for t in model.T)
        
        return cost_ac + lambda_soc * (penalty_soc + penalty_soc_2) + lambda_term * penalty_term + 1000 * penalty_forbidden

    def add_constraints_objective(self, model):
        """
        Registers all constraints and the objective function to the model.
        """
        # Register constraints using the private methods defined above
        model.battery_capacity_update = pyo.Constraint(model.T, rule=self._battery_capacity_update_constraint)
        model.grid_balancing_constraint = pyo.Constraint(model.T, rule=self._grid_balancing_constraint)
        model.minimum_soc_constraint = pyo.Constraint(model.T, rule=self._minimum_soc_constraint)
        model.maximum_soc_constraint = pyo.Constraint(model.T, rule=self._maximum_soc_constraint)
        
        #AC charging binary constraint
        # model.ac_charging_threshold_binary_1_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_1_constraint)
        # model.ac_charging_threshold_binary_2_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_2_constraint)
        # model.ac_charging_threshold_binary_3_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_3_constraint)
        # model.ac_charging_threshold_binary_4_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_4_constraint)
        # model.ac_charging_threshold_binary_5_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_5_constraint)
        # model.ac_charging_total_threshold_constraint = pyo.Constraint(model.T, rule=self._ac_charging_total_threshold_constraint)

        model.charging_idle_losses_constraint = pyo.Constraint(model.T, rule=self._charging_idle_losses_constraint)
        # Note: Terminal constraint is scalar (not indexed by T)
        model.terminal_soc_deviation_constraint = pyo.Constraint(rule=self._terminal_soc_deviation_constraint)
        
        model.max_ac_charging_power_constraint = pyo.Constraint(model.T, rule=self._max_ac_charging_power_constraint)

        model.final_soc_con = pyo.Constraint(rule=self._final_soc_constraint)
        # Set Objective
        model.objective = pyo.Objective(rule=self._get_model_objective, sense=pyo.minimize)

    def optimize(self):
        """Optimize .
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        load_df_combined = self.get_load()
        results_summary = {}

        if self.vehicle is None:
            return None

        ch_avail_dict = {}
        soc_max_dict = {}
        soc_min_dict = {}
        charging_dict = {}
        consumption_dict = {}
        for key in self.vehicle.keys():
            ch_refuel, start_discharging, n_controlled = self.vehicle.get(key)
            group = key
            filename = f"ch_agg_{ch_refuel}_{start_discharging}_DEU_" + \
                       f"{self.cfg.date_input}_2012_cal2012_{self.cfg.region_name}_{key}_{n_controlled}.csv"
            ch_avail, soc_max, soc_min, charging, consumption = self.get_ev_load(
                ch_refuel, start_discharging, n_controlled, filename, group
            )
            ch_avail_dict[key] = ch_avail.to_numpy()
            soc_max_dict[key] = soc_max.to_numpy()
            soc_min_dict[key] = soc_min.to_numpy()
            charging_dict[key] = charging.to_numpy()
            consumption_dict[key] = consumption.to_numpy()

        input_data = self._prepare_optimization_dataframe(
            load_series=load_df_combined,
            uncontrolled_ev_charging=charging_dict.get("uncontrolled"), 
            controlled_consumption=consumption_dict.get("controlled_v2g"), 
            controlled_ch_avail=ch_avail_dict.get("controlled_v2g"),
            controlled_soc_max=soc_max_dict.get("controlled_v2g"),
            controlled_soc_min=soc_min_dict.get("controlled_v2g"),
        )

        steps_per_day = 96
        horizon_days = 3
        step_size_days = 1

        horizon_steps = horizon_days * steps_per_day
        step_size_steps = step_size_days * steps_per_day

        total_steps = len(input_data)
        current_step=0
        results_list = []

        current_initial_soc = input_data['maximum_soc_MWh'].iloc[0]
        print("current", current_initial_soc)

        if self.opt_cfg.solver == 'HiGHS':
            solver = pyo.SolverFactory('appsi_highs')
            solver.options['time_limit'] = 60
            solver.options['mip_heuristic_effort'] = 0.2
            solver.options['mip_detect_symmetry'] = 'on'
            solver.options['output_flag'] = False

        elif self.opt_cfg.solver == 'GLPK':
            solver = pyo.SolverFactory('glpk')

        while current_step < total_steps:
            window_end = min(current_step + horizon_steps, total_steps)
            
            if window_end - current_step < 1:
                break
            
            input_slice = input_data.iloc[current_step:window_end].copy().reset_index(drop=True)
            
            self._build_model(input_slice, current_initial_soc)
            self.add_constraints_objective(self._model)
    
            results = solver.solve(self._model, tee=False, load_solutions=False)
        
            if (results.solver.status == pyo.SolverStatus.ok) and \
               (results.solver.termination_condition == pyo.TerminationCondition.optimal):
                self._model.solutions.load_from(results)
            else:
                try:
                    self._model.solutions.load_from(results)
                except:
                    print(f"Basic Optimization failed at step {current_step}")
                    break
            
            steps_to_keep = min(step_size_steps, len(input_slice))
            
            df_slice = self._extract_result_slice(steps_to_keep)
            df_slice.index = input_data.index[current_step : current_step + steps_to_keep]
            
            results_list.append(df_slice)
            
            # E. Update SoC
            last_idx = steps_to_keep - 1
            current_initial_soc = self._model.soc[last_idx].value
            
            # F. Slide
            current_step += steps_to_keep
            
            if (current_step // steps_per_day) % 50 == 0:
                print(f"  Processed {current_step}/{total_steps} steps...")
    
        if results_list:
            final_result = pd.concat(results_list)
            self._final_result = final_result
            print("Rolling Horizon Basic Optimization Completed.")
            return self._model
        else:
            print("No results.")
            return None
    
    def _extract_result_slice(self, steps_to_keep):
        """Extract result slice.
        
        Parameters
        ----------
        steps_to_keep : object
            Steps to keep used by this operation.
        
        Returns
        -------
        object
            Result computed as ``pd.DataFrame(df_dict)``."""
        model = self._model
        timeslot = list(range(steps_to_keep))

        # 提取结果 (Energy MWh)
        ac_charging_MWh = [model.energy_vehicle_ac_charging[t].value for t in timeslot]
        grid_consumption_MWh = [model.energy_grid_ac_consumption[t].value for t in timeslot]
        soc_MWh = [model.soc[t].value for t in timeslot]
        comfort_violation = [model.comfort_binary[t].value for t in timeslot]

        df_dict = {
            'household_consumption_MWh': [model.hh_cons_MWh[t] for t in timeslot],
            'driving_consumption_MWh': [model.driving_consumption_MWh[t] for t in timeslot],
            'max_ac_charging_MWh': [model.max_charging_ac_MWh[t] for t in timeslot],
            'price_consumption_ac_euro_MWh': [model.price_cons_ac[t] for t in timeslot],
            
            'ac_vehicle_charging_MWh': ac_charging_MWh,
            'ac_grid_consumption_MWh': grid_consumption_MWh,
            'battery_capacity_MWh': soc_MWh,
            'comfort_penalty_active': comfort_violation,
        }

        return pd.DataFrame(df_dict)
    
    def export_results(self):
        # 1. 检查是否有结果
        """Export results.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        if not hasattr(self, '_final_result') or self._final_result is None:
            print("Error: No results found. Please run optimize() first.")
            return None

        # 2. 构建文件路径 (使用 pathlib 的 / 操作符)
        output_path = self.scenario_data.optimization_dir / "price.csv"

        self._final_result.to_csv(output_path, sep=";")

        print(f"Rolling Horizon Results exported to: {output_path}")
        
        return self._final_result


