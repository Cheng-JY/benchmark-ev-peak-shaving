"""Milp bidi optimizer utilities for the EV/V2G analysis workflow."""

from .base_optimizer import BaseEVOptimizer
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from ...utils import get_state
from ...utils._data import read_price, read_feedin_price

class PyomoBidirectionalEVOptimizer(BaseEVOptimizer):

    """Represent pyomo bidirectional evoptimizer and its associated operations."""
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

        # --- Helper for Data Cleaning ---
        def clean_data(data_input):
            # Check for None first
            """Clean data.
            
            Parameters
            ----------
            data_input : object
                Data input used by this operation.
            
            Returns
            -------
            object
                Computed result for the selected input and execution branch."""
            if data_input is None:
                return 0.0
            
            if hasattr(data_input, 'values'):
                arr = data_input.values
            else:
                arr = data_input
            
            # Ensure float type and replace NaNs
            return np.nan_to_num(np.array(arr, dtype=float), nan=0.0)

        # Clean inputs to remove NaNs and handle None
        load_clean = clean_data(load_series)
        uncontrolled_clean = clean_data(uncontrolled_ev_charging)
        avail_clean = clean_data(controlled_ch_avail) # in MW (Power)
        cons_clean = clean_data(controlled_consumption) # MWh (Energy)
        soc_max_clean = clean_data(controlled_soc_max) # MWh (Energy)
        soc_min_clean = clean_data(controlled_soc_min) # 

        # 1. Load: Consumption = HH_load (MWh) + uncontrolled (MWh)
        df["household_consumption_MWh"] = (load_clean + uncontrolled_clean)

        df["ambient_temperature_degC"] = 20 # Default temp

        # 2. Charging Limits: Mw (Power)        
        df['maximum_ac_charging_MW'] = avail_clean
        df['maximum_ac_feedin_MW'] = avail_clean
        
        _, price_euro = read_price()
        df["price_ac_consumption_euro_MWh"] = price_euro.values * 1000
        df["price_ac_feedin_euro_MWh"] = read_feedin_price().values * 1000
        
        df["driving_consumption_MWh"] = cons_clean
        df['maximum_soc_MWh'] = soc_max_clean
        df["minimum_soc_MWh"] = soc_min_clean

        return df
    
    def build_model(self, vehicle):
        """Build model.
        
        Parameters
        ----------
        vehicle : object
            Vehicle used by this operation."""
        self.vehicle = vehicle

    def _build_model(self, input_data, inital_soc=None, terminal_soc=None):
        """Build model.
        
        Parameters
        ----------
        input_data : object
            Input data used by this operation.
        inital_soc : object
            Inital soc used by this operation.
        terminal_soc : object
            Terminal soc used by this operation."""
        print("Building Bidirectional Pyomo Model...")
        model = pyo.ConcreteModel(name='EV Charging Bidirectional MILP')

        # Sets
        model.T = pyo.Set(initialize=list(range(len(input_data))), ordered=True)

        # Parameters
        model.price_cons_ac = list(input_data['price_ac_consumption_euro_MWh'])
        model.price_feedin_ac = list(input_data['price_ac_feedin_euro_MWh'])

        model.hh_cons_kw = list(input_data['household_consumption_MWh'])
        model.max_charging_ac = list(input_data['maximum_ac_charging_MW'])
        model.max_feedin_ac = list(input_data['maximum_ac_feedin_MW'])
        model.driving_consumption = list(input_data["driving_consumption_MWh"])

        model.bc_max = pyo.Param(initialize=np.max(input_data["maximum_soc_MWh"]),
                                 within=pyo.NonNegativeReals)
        model.soc_min = list(input_data["minimum_soc_MWh"])
        model.soc_max = list(input_data["maximum_soc_MWh"])
        
        init_soc = model.soc_max[0]
        term_soc = model.soc_max[-1]

        model.soc_ini = pyo.Param(initialize=init_soc, within=pyo.NonNegativeReals)
        model.soc_term = pyo.Param(initialize=term_soc, within=pyo.Reals)

        # Vehicle AC Power Limits (Time-series list)
        model.pmax_veh_ac = list(input_data['maximum_ac_charging_MW'])
        
        # Grid Power Limit (Scalar)
        # Ensure inputs are arrays for element-wise addition if needed, or use safe max
        # Here assuming lists are compatible or taking global max
        pmax_grid = np.max(np.array(model.hh_cons_kw) + np.array(model.max_charging_ac)) * 4
        model.pmax_grid_ac = pyo.Param(initialize=pmax_grid, within=pyo.NonNegativeReals)

        # ---Variables---
        # Upper bound is max possible power found in data
        max_p_val = np.max(model.pmax_veh_ac) if len(model.pmax_veh_ac) > 0 else 0.011
        model.power_vehicle_ac_charging = pyo.Var(model.T, bounds=(0, max_p_val))
        max_p_feedin = np.max(model.max_feedin_ac) if len(model.max_feedin_ac) > 0 else 0.011
        model.power_vehicle_ac_feedin = pyo.Var(model.T, bounds=(0, max_p_feedin))

        model.soc = pyo.Var(model.T, bounds=(0, model.bc_max))
        model.terminal_soc_deviation = pyo.Var(bounds=(0, None))

        model.power_grid_ac_consumption = pyo.Var(model.T, bounds=(0, model.pmax_grid_ac))
        model.power_grid_ac_feedin = pyo.Var(model.T, bounds=(0, model.pmax_grid_ac))

        model.comfort_binary = pyo.Var(model.T, domain=pyo.Binary, initialize=0)
        model.is_charging_ac = pyo.Var(model.T, domain=pyo.Binary, initialize=0)
        model.is_discharging_ac = pyo.Var(model.T, domain=pyo.Binary, initialize=0)

        self._model = model

    def _battery_capacity_update_constraint(self, model, t):
        """Battery capacity update constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.soc[t] == current - driving``."""
        ac_in = model.power_vehicle_ac_charging[t]
        ac_out = model.power_vehicle_ac_feedin[t] 
        driving = model.driving_consumption[t]
        
        if t == model.T.first():
            current = model.soc_ini + ac_in - ac_out
        else:
            current = model.soc[t-1] + ac_in - ac_out
            
        return model.soc[t] == (current - driving)
    
    def _grid_balancing_constraint(self, model, t):
        """Grid balancing constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``net_grid == net_ev + model.hh_cons_kw[t]``."""
        net_grid = model.power_grid_ac_consumption[t] - model.power_grid_ac_feedin[t]
        net_ev = model.power_vehicle_ac_charging[t] - model.power_vehicle_ac_feedin[t]
        return net_grid == (net_ev + model.hh_cons_kw[t])
    
    def _link_charge_binary_constraint(self, model, t):
        # FIX: Use indexed access [t] for pmax_veh_ac list
        """Link charge binary constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.power_vehicle_ac_charging[t] <= model.pmax_veh_ac[t] * model.is_charging_ac[t]``."""
        return model.power_vehicle_ac_charging[t] <= model.pmax_veh_ac[t] * model.is_charging_ac[t]

    def _link_discharge_binary_constraint(self, model, t):
        # FIX: Use indexed access [t] for pmax_veh_ac list
        """Link discharge binary constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.power_vehicle_ac_feedin[t] <= model.pmax_veh_ac[t] * model.is_discharging_ac[t]``."""
        return model.power_vehicle_ac_feedin[t] <= model.pmax_veh_ac[t] * model.is_discharging_ac[t]
    
    def _exclusivity_constraint(self, model, t):
        """Exclusivity constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.is_charging_ac[t] + model.is_discharging_ac[t] <= 1``."""
        return model.is_charging_ac[t] + model.is_discharging_ac[t] <= 1

    def _get_model_objective(self, model):
        
        # Grid Costs
        """Return model objective.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        
        Returns
        -------
        object
            Result computed as ``cost_grid - revenue_feedin + lambda_soc * (penalty_soc + penalty_term)``."""
        cost_grid = sum(model.power_grid_ac_consumption[t] * 0.25 * model.price_cons_ac[t] for t in model.T)
        # Grid Revenue
        revenue_feedin = sum(model.power_grid_ac_feedin[t] * 0.25 * model.price_feedin_ac[t] for t in model.T)
        
        lambda_soc = 10000.0
        penalty_soc = sum(model.comfort_binary[t] for t in model.T)
        penalty_term = model.terminal_soc_deviation
        
        return (cost_grid - revenue_feedin) + lambda_soc * (penalty_soc + penalty_term)
    
    def add_constraints_objective(self, model):
        """Add constraints objective.
        
        Parameters
        ----------
        model : object
            Model used by this operation."""
        print("Adding Constraints...")
        
        model.battery_update = pyo.Constraint(model.T, rule=self._battery_capacity_update_constraint)
        model.grid_balance = pyo.Constraint(model.T, rule=self._grid_balancing_constraint)
        
        model.link_charge = pyo.Constraint(model.T, rule=self._link_charge_binary_constraint)
        model.link_discharge = pyo.Constraint(model.T, rule=self._link_discharge_binary_constraint)
        model.exclusivity = pyo.Constraint(model.T, rule=self._exclusivity_constraint)
        
        # Using lambda with index access for list parameters
        model.max_charge = pyo.Constraint(model.T, rule=lambda m, t: m.power_vehicle_ac_charging[t] <= m.max_charging_ac[t])
        model.max_discharge = pyo.Constraint(model.T, rule=lambda m, t: m.power_vehicle_ac_feedin[t] <= m.max_feedin_ac[t])
        
        # FIX: Added [t] index to soc_min
        bigM = model.bc_max * 1.1
        model.min_soc = pyo.Constraint(model.T, rule=lambda m, t: (m.soc[t] - m.soc_min[t]) >= -bigM * m.comfort_binary[t])
        
        # FIX: Terminal SoC constraint logic
        # Ensure terminal_soc_deviation is scalar variable
        last_t = model.T.last()
        model.term_soc = pyo.Constraint(rule=lambda m: m.terminal_soc_deviation >= (m.soc_term - m.soc[last_t]))

        model.objective = pyo.Objective(rule=self._get_model_objective, sense=pyo.minimize)
        print("Constraints Added.")

    def optimize(self):
        """Optimize .
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        from tqdm import tqdm
        load_df_combined = self.get_load()

        if self.vehicle is None:
            return None

        ch_avail_dict = {}
        soc_max_dict = {}
        soc_min_dict = {}
        charging_dict = {}
        consumption_dict = {}
        key_to_optimize = None
        
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
            
            if "controlled_v2g" in key:
                key_to_optimize = key

        if key_to_optimize is None:
            print("Error: No 'controlled_v2g' scenario found.")
            return None

        print(f"Optimizing scenario: {key_to_optimize} with Rolling Horizon...")

        full_input_data = self._prepare_optimization_dataframe(
            load_series=load_df_combined,
            uncontrolled_ev_charging=charging_dict.get("uncontrolled"), 
            controlled_consumption=consumption_dict.get(key_to_optimize), 
            controlled_ch_avail=ch_avail_dict.get(key_to_optimize),
            controlled_soc_max=soc_max_dict.get(key_to_optimize),
            controlled_soc_min=soc_min_dict.get(key_to_optimize),
        )
        
        # --- 2. Rolling Horizon 参数设置 ---
        # 步长 (Control Horizon): 1 天 = 96 个 15分钟
        # 视野 (Optimization Horizon): 3 天 = 288 个 15分钟
        step_size = 96 
        horizon_size = 96 * 3
        total_steps = len(full_input_data)
        
        current_step = 0
        results_list = []
        
        current_soc_start = full_input_data['maximum_soc_MWh'].iloc[0]
        
        # --- 3. 循环优化 (Sliding Window) ---
        # 这是一个手动实现的滚动循环
        pbar = tqdm(total=total_steps, desc="Rolling Optimization", unit="step", leave=True)

        while current_step < total_steps:
            # A. 计算窗口结束点 (防止越界)
            window_end = min(current_step + horizon_size, total_steps)
            
            # 如果剩余数据太少(不足1个步长)，可以选择跳过或者只优化剩余部分
            # 这里我们选择优化剩余部分
            if window_end - current_step < 1:
                break
                
            # B. 切片数据：取出未来 3 天的数据用于建立模型
            input_slice = full_input_data.iloc[current_step:window_end].copy().reset_index(drop=True)
            
            # C. 构建模型 (关键：传入上一轮计算出的 current_soc_start)
            # terminal_soc=None 表示不强制要求 3 天后必须满电 (避免短视导致不可行)
            self._build_model(input_slice, inital_soc=current_soc_start, terminal_soc=None)
            self.add_constraints_objective(self._model)
            
            # D. 求解
            if hasattr(self.opt_cfg, 'solver') and self.opt_cfg.solver == 'GLPK':
                 solver = pyo.SolverFactory('glpk')
            else:
                 solver = pyo.SolverFactory('appsi_highs')

            # load_solutions=False 防止 Windows 文件锁报错
            results = solver.solve(self._model, tee=False, load_solutions=False)
            
            # E. 加载结果
            if (results.solver.status == pyo.SolverStatus.ok) and \
               (results.solver.termination_condition == pyo.TerminationCondition.optimal):
                self._model.solutions.load_from(results)
            else:
                # 尝试加载可行解 (Feasible Solution)
                try:
                    self._model.solutions.load_from(results)
                except:
                    pbar.write(f"\n[Warning] Optimization failed at step {current_step}")
                    print(f"Optimization failed at step {current_step}")
                    break

            # --- 4. 提取并保存结果 ---
            # 虽然我们优化了 3 天，但只采纳前 1 天 (step_size) 的决策
            # 如果是最后一次循环，可能不满 1 天，取实际长度
            steps_to_keep = min(step_size, len(input_slice))
            df_slice = self._extract_result_slice(steps_to_keep)
            df_slice.index = full_input_data.index[current_step : current_step + steps_to_keep]
            results_list.append(df_slice)
            
            # --- 5. 更新状态 ---
            # 下一轮的初始电量 = 这一轮保留部分的最后一个时刻的电量
            # Pyomo 索引从 0 开始，所以最后一个索引是 steps_to_keep - 1
            raw_soc = self._model.soc[steps_to_keep - 1].value
            max_cap = self._model.bc_max.value
            import numpy as np
            current_soc_start = float(np.clip(raw_soc, 0.0, max_cap))
            
            # F. 滑动窗口
            current_step += steps_to_keep
            pbar.update(steps_to_keep)
            
            # (可选) 打印进度
            if (current_step % (step_size * 10)) == 0:
                print(f"Processed {current_step}/{total_steps} steps...")
        pbar.close()

        # --- 6. 合并结果 ---
        if len(results_list) > 0:
            final_result = pd.concat(results_list)
            print("Rolling Horizon Optimization complete.")
            self._final_result = final_result 
            return final_result
        else:
            print("No results generated.")
            return None
        
    # 请在类中添加这个新方法
    def _extract_result_slice(self, steps_to_keep):
        """
        从当前模型中提取前 steps_to_keep 个时间步的结果
        """
        model = self._model
        timeslot = list(range(steps_to_keep))
        
        data = {
            'grid_consumption_kw': [model.power_grid_ac_consumption[t].value for t in timeslot],
            'grid_feedin_kw': [model.power_grid_ac_feedin[t].value for t in timeslot],
            'ev_charging_kw': [model.power_vehicle_ac_charging[t].value for t in timeslot],
            'ev_discharging_kw': [model.power_vehicle_ac_feedin[t].value for t in timeslot],
            'soc': [model.soc[t].value for t in timeslot],
            'price_buy': [model.price_cons_ac[t] for t in timeslot],
            'price_sell': [model.price_feedin_ac[t] for t in timeslot],
            'is_charging': [model.is_charging_ac[t].value for t in timeslot],
            'is_discharging': [model.is_discharging_ac[t].value for t in timeslot],
            'comfort_penalty': [model.comfort_binary[t].value for t in timeslot]
        }
        
        return pd.DataFrame(data)

    # 修改 export_results 以支持直接导出拼接好的结果
    def export_results(self):
        """Export results.
        
        Returns
        -------
        object
            Computed result for the selected input and execution branch."""
        if hasattr(self, '_final_result'):
            df = self._final_result
            if hasattr(self.scenario_data, 'optimization_dir'):
                output_path = self.scenario_data.optimization_dir / "price_v2g_full_year.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, sep=";")
                print(f"Results exported to {output_path}")
            return df
        else:
            # 回退到旧逻辑
            return super().export_results() # 假设父类有，或者用之前的逻辑