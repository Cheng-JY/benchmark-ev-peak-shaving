"""Basic optimizer utilities for the EV/V2G analysis workflow."""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraits
from .base_optimizer import BaseEVOptimizer
import logging

"""
Pyomo solar optimizer class with full AC charging loss modeling and forbidden charging powers < 2 kW. Loss function is
derived from real world onboard-charger data. 
"""

class BasicEVOptimizer(BaseEVOptimizer):

    """Represent basic evoptimizer and its associated operations."""
    def __init__(self, optimizer_config):
        """Initialize the instance using the supplied configuration and data.
        
        Parameters
        ----------
        optimizer_config : object
            Optimizer config used by this operation."""
        super().__init__(optimizer_config)

    def build_day_ahead_model(self, input_data, vehicle, initial_soc, terminal_soc):
        
        # 内部小函数：价格附加项
        """Build day ahead model.
        
        Parameters
        ----------
        input_data : object
            Input data used by this operation.
        vehicle : object
            Vehicle used by this operation.
        initial_soc : object
            Initial soc used by this operation.
        terminal_soc : object
            Terminal soc used by this operation."""
        def _get_surcharge_vector_consumption_price(input_data):
            """Return surcharge vector consumption price.
            
            Parameters
            ----------
            input_data : object
                Input data used by this operation.
            
            Returns
            -------
            object
                Result computed as ``0``."""
            return 0
        
        model = pyo.ConcreteModel(name='EV Charging Basic Optimization')
        optimization_config = self.opt_cfg

        # 内部小函数：价格附加项
        surcharge_vector_price_cons = _get_surcharge_vector_consumption_price(input_data)

        # Set models Sets
        model.T = pyo.Set(doc='Range of timeslot t within simulation horizon',
                          initialize=list(range(len(input_data))), ordered=True)
        
        model.calender_week = list(input_data['calender_week'])
        model.hh_cons_kw = list(input_data['household_consumption_kw']) # getting from householdprofileBuilder
        model.ambient_temp = list(input_data['ambient_temperature_degC']) # getting with scenario and scenario year
        model.price_cons_ac = list(input_data['price_ac_consumption_day_ahead_euro_kwh'] + surcharge_vector_price_cons)
        model.price_cons_dc = list(input_data['price_dc_consumption_euro_kwh'] + surcharge_vector_price_cons)
        model.max_charging_ac = list(input_data['maximum_ac_charging_kw']) # ch_avail 随时间变化
        model.min_charging_ac = list(input_data['minimum_ac_charging_kw']) # 0
        model.max_charging_dc = list(input_data['maximum_dc_charging_kw']) # zhe ge kuai chong xu yao ma? 150kw
        model.driving_distance = list(input_data['driving_distance_km']) # in driving_profile
        model.driving_consumption = list(input_data['driving_consumption_kwh']) # consumption in aggregation

        # Set models Parameters
        model.bc_max = pyo.Param(initialize=vehicle.battery_size_kwh,
                                 doc='Total EV battery capacity in kwh') # soc_max
        model.soc_ini = pyo.Param(initialize=initial_soc,
                                  doc='Initial SOC at first timeslot') # soc_max[0]
        model.soc_min = pyo.Param(initialize=optimization_config.vehicle_minimum_soc,
                                  doc='Minimum SOC during simulation period') # soc_min
        model.soc_term = pyo.Param(initialize=terminal_soc,
                                   doc='Terminal SOC at the end of simulation period.')
        
        model.pmax_veh_ac = pyo.Param(initialize=vehicle.maximum_ac_charging_power_kw,
                                      doc="EVs maximum AC charging power.") # max(ch_avail) 是一个标量 11
        model.pmin_veh_ac = pyo.Param(initialize=0,
                                      doc="EVs minimum AC charging power") # no bidirectional charging
        model.pmax_grid_ac = pyo.Param(initialize=optimization_config.maximum_ac_consumption_grid_kw,
                                       doc='Maximum AC power consumption from grid.') # 电网允许你从 AC 侧最多取多少电（千瓦
        model.pmin_grid_ac = pyo.Param(initialize=optimization_config.minimum_ac_consumption_grid_kw,
                                       doc='Minimum AC power consumption from grid.') # 从电网最少必须消耗的 AC 参考功率 (不让优化结果出现负功率（例如禁止 AC side V2G）)
        model.pmax_veh_dc = pyo.Param(initialize=vehicle.maximum_dc_charging_power_kw,
                                      doc='EVs maximum DC charging power.') # 快充的值

        model.eta_dc = pyo.Param(initialize=vehicle.efficiency_dc_charging, doc='EVs DC charging efficiency.') # 电动车 DC 充电效率（比例

        # Set Decision Variables
        model.power_vehicle_ac_charging = pyo.Var(model.T, bounds=(0, model.pmax_veh_ac),
                                                  doc="EVs AC charging power in kw, connected to wallbox, during timeslot t.")
        model.power_vehicle_dc_charging = pyo.Var(model.T, bounds=(0, model.pmax_veh_dc),
                                                  doc='EV DC charging power from grid in kw during timeslot t.')
        model.power_grid_ac_consumption = pyo.Var(model.T, bounds=(0, model.pmax_grid_ac),
                                                  doc='Power consumption from ac grid.') # 电网总消耗要在target之下
        model.charging_idle_losses = pyo.Var(model.T, bounds=(0, model.pmax_veh_ac),
                                             doc='Power losses AC charging and vehicle idle') # 这个我应该不需要考虑
        model.bc = pyo.Var(model.T, bounds=(0, model.bc_max), doc='EVs battery capacity in kwh during timeslot t.') # soc

        model.terminal_soc_deviation = pyo.Var(bounds=(0, None)) # 最终（终端）SoC 偏差, 软约束（soft constraint）

        model.comfort_binary = pyo.Var(model.T, domain=pyo.Binary, initialize=0,
                                       doc='Comfort binary value becomes 1 once SoC falls below soc_min to increment'
                                        'penalty.') # 如果在某个时刻 t，电动车 SoC < SoC_min, 那么 comfort_binary[t] 必须 = 1 
                                                    # 这样优化器会对舒适度（comfort）的损失进行惩罚
        model.dc_charging_binary = pyo.Var(model.T, domain=pyo.Binary, initialize=0, doc='Binary value becomes 1 if'
                                                                                         'vehicle is in DC charging'
                                                                              ' mode, else 0')
        

    def build_intraday_auction_model(self, input_data, vehicle, initial_soc, terminal_soc, day_ahead_optimization_result):

        """Build intraday auction model.
        
        Parameters
        ----------
        input_data : object
            Input data used by this operation.
        vehicle : object
            Vehicle used by this operation.
        initial_soc : object
            Initial soc used by this operation.
        terminal_soc : object
            Terminal soc used by this operation.
        day_ahead_optimization_result : object
            Day ahead optimization result used by this operation."""
        def _get_surcharge_vector_consumption_price(input_data):
            """Return surcharge vector consumption price.
            
            Parameters
            ----------
            input_data : object
                Input data used by this operation.
            
            Returns
            -------
            object
                Result computed as ``surcharge_vector``."""
            optimizaton_config = self._optimization_config
            days = len(input_data) / 96
            # Intraday 价格更真实，但更波动
            daily_price_surcharge_factor = optimizaton_config.ac_charging_price_daily_surcharge_percent / 100.0
            average_price = input_data['price_ac_consumption_intraday_auction_euro_kwh'].mean().item()
            factor = average_price * daily_price_surcharge_factor
            
            surcharge_vector = pd.Series(np.linspace(0, days*factor, len(input_data)), index=input_data.index)
            return surcharge_vector

        def _get_vehicle_charging_events(day_ahead_optimization_results):
            """Return vehicle charging events.
            
            Parameters
            ----------
            day_ahead_optimization_results : object
                Day ahead optimization results used by this operation.
            
            Returns
            -------
            object
                Result computed as ``df.index``."""
            df = day_ahead_optimization_results[day_ahead_optimization_result['ac_vehicle_charging_kw'] >= 1.5]
            return df.index

        model = pyo.ConcreteModel(name='EV Charging Basic Optimization')

        optimization_config = self._optimization_config

        # Set models Sets
        model.T = pyo.Set(doc='Range of timeslots t within simulation horizon',
                          initialize=list(range(len(input_data))), ordered=True)

        surcharge_vector_price_cons = _get_surcharge_vector_consumption_price(input_data)

        # Set models static input vectors
        #Set models static input vectors
        model.calender_week = list(input_data['calender_week'])
        model.hh_cons_kw = list(input_data['household_consumption_kw'])
        model.ambient_temp = list(input_data['ambient_temperature_degC'])
        model.price_cons_ac = list(input_data['price_ac_consumption_intraday_auction_euro_kwh'] + surcharge_vector_price_cons)
        model.price_cons_dc = list(input_data['price_dc_consumption_euro_kwh'] + surcharge_vector_price_cons)
        model.max_charging_ac = list(input_data['maximum_ac_charging_kw'])
        model.min_charging_ac = list(input_data['minimum_ac_charging_kw'])
        model.max_charging_dc = list(input_data['maximum_dc_charging_kw'])
        model.min_charging_dc = list(input_data['minimum_dc_charging_kw'])
        model.driving_distance = list(input_data['driving_distance_km'])
        model.driving_consumption = list(input_data['driving_consumption_kwh'])

        #Set models Parameters
        model.bc_max = pyo.Param(initialize=vehicle.battery_size_kwh,
                                 doc='Total EV battery capacity in kwh')
        model.soc_ini = pyo.Param(initialize=initial_soc,
                                  doc='Initial SOC at first timeslot.')
        model.soc_min = pyo.Param(initialize=optimization_config.vehicle_minimum_soc,
                                  doc='Minimum SOC during simulation period.')
        model.soc_term = pyo.Param(initialize=terminal_soc,
                                   doc='Terminal SOC at the end of simulation period.')

        model.pmax_veh_ac = pyo.Param(initialize=vehicle.maximum_ac_charging_power_kw,
                                      doc='EVs maximum AC charging power.')
        model.pmin_veh_ac = pyo.Param(initialize=0, doc='EVs minimum AC charging power.') # no bidirectional charging
        model.pmax_grid_ac = pyo.Param(initialize=optimization_config.maximum_ac_consumption_grid_kw,
                                       doc='Maximum AC power consumption from grid.')
        model.pmin_grid_ac = pyo.Param(initialize=optimization_config.minimum_ac_consumption_grid_kw,
                                       doc='Minimum AC power consumption from grid.')
        model.pmax_veh_dc = pyo.Param(initialize=vehicle.maximum_dc_charging_power_kw,
                                      doc='EVs maximum DC charging power.')

        model.eta_dc = pyo.Param(initialize=vehicle.efficiency_dc_charging, doc='EVs DC charging efficiency.')

        #Set Decision Variables
        model.power_vehicle_ac_charging = pyo.Var(model.T, bounds=(0, model.pmax_veh_ac),
                                     doc='EVs AC charging power in kw, connected to wallbox, during timeslot t.')

        #lock vehicle ac charging values:
        # Intraday 不允许对 Day-Ahead 已经决定的“强制充电”时间段进行修改！
        locked_charging_event_ids = _get_vehicle_charging_events(day_ahead_optimization_result)
        if len(locked_charging_event_ids) != 0:
            print('locking charging event')
            for idx in locked_charging_event_ids:
                model.power_vehicle_ac_charging[idx].fix(day_ahead_optimization_result.loc[idx, 'ac_vehicle_charging_kw'].item())

        model.power_vehicle_dc_charging = pyo.Var(model.T, bounds=(0, model.pmax_veh_dc),
                                                  doc='EV DC charging power from grid in kw during timeslot t.')
        model.power_grid_ac_consumption = pyo.Var(model.T, bounds=(0, model.pmax_grid_ac),
                                      doc='Power consumption from ac grid.')
        model.charging_idle_losses = pyo.Var(model.T, bounds=(0, model.pmax_veh_ac),
                                             doc='Power losses AC charging and vehicle idle')
        model.bc = pyo.Var(model.T, bounds=(0, model.bc_max), doc='EVs battery capacity in kwh during timeslot t.')

        model.terminal_soc_deviation = pyo.Var(bounds=(0, None))

        model.comfort_binary = pyo.Var(model.T, domain=pyo.Binary, initialize=0,
                                       doc='Comfort binary value becomes 1 once SoC falls below soc_min to increment'
                                        'penalty.')
        model.dc_charging_binary = pyo.Var(model.T, domain=pyo.Binary, initialize=0, doc='Binary value becomes 1 if'
                                                                                         'vehicle is in DC charging'
                                                                                         ' mode, else 0')


        #Binary for intervall [0, epsilon], epsilon=1e-3
        model.ac_charging_threshold_binary_1 = pyo.Var(model.T, domain=pyo.Binary, initialize=0)
        #Binary for intervall [epsilon, 2.0], epsilon=1e-3
        model.ac_charging_threshold_binary_2 = pyo.Var(model.T, domain=pyo.Binary, initialize=0)
        #Binary for intervall [2.0, pmax]
        model.ac_charging_threshold_binary_3 = pyo.Var(model.T, domain=pyo.Binary, initialize=0)

        self._model = model

        print('Intraday Auction Model Compiled!')
    
    def optimize(self) -> pyo.ConcreteModel:

        """Optimize .
        
        Returns
        -------
        pyo.ConcreteModel
            Result computed as ``model``."""
        model = self._model
        optimization_config = self._optimization_config

        #CONSTRAINTS

        #General update of battery capacity.
        model.battery_capacity_update = pyo.Constraint(model.T, rule=self._battery_capacity_update_constraint)
        #Lagrange multiplier constraint for minimum SOC constraint.
        model.minimum_soc_constraint = pyo.Constraint(model.T, rule=self._minimum_soc_constraint)

        #AC charging binary constraint
        model.ac_charging_threshold_binary_1_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_1_constraint)
        model.ac_charging_threshold_binary_2_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_2_constraint)
        model.ac_charging_threshold_binary_3_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_3_constraint)
        model.ac_charging_threshold_binary_4_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_4_constraint)
        model.ac_charging_threshold_binary_5_constraint = pyo.Constraint(model.T, rule=self._ac_charging_threshold_binary_5_constraint)
        model.ac_charging_total_threshold_constraint = pyo.Constraint(model.T, rule=self._ac_charging_total_threshold_constraint)

        #Terminal SoC constraint
        model.terminal_soc_deviation_constraint = pyo.Constraint(model.T, rule=self._terminal_soc_deviation_constraint)

        #DC charging binary constraint
        model.dc_charging_binary_constraint = pyo.Constraint(model.T, rule=self._dc_charging_binary_constraint)

        #Grid balancing constraint
        model.grid_balancing_constraint = pyo.Constraint(model.T, rule=self._grid_balancing_constraint)
        #AC charging and idle losses constraint
        model.charging_idle_losses_constraint = pyo.Constraint(model.T, rule=self._charging_idle_losses_constraint)

        #Charging Power constraints for AC and DC-Charging.
        model.max_ac_charging_power_constraint = pyo.Constraint(model.T, rule=self._max_ac_charging_power_constraint)
        model.max_dc_charging_power_constraint = pyo.Constraint(model.T, rule=self._max_dc_charging_power_constraint)

        #set model objective
        model.objective = pyo.Objective(expr=self._get_model_objective(model), sense=pyo.minimize)

        if optimization_config.solver == 'HiGHS':
            #solver = HiGHS(time_limit=10, mip_heuristic_effort=0.2, mip_detect_symmetry='on', output_flag=False)
            solver = pyo.SolverFactory('appsi_highs')
            solver.options['time_limit'] = 60
            solver.options['mip_heuristic_effort'] = 0.2
            solver.options['mip_detect_symmetry'] = 'on'
            solver.options['output_flag'] = False

        elif optimization_config.solver == 'GLPK':
            solver = pyo.SolverFactory('glpk')

        elif optimization_config.solver == 'SCIP':
            solver = pyo.SolverFactory('scip', solver_io='nl')

        results = solver.solve(model)

        if hasattr(results, 'timing_info') and results.timing_info.wall_time < 0:
            results.timing_info.wall_time = 0.0

        if optimization_config.logging_enabled:
            log_infeasible_constraints(model, log_expression=True, log_variables=True)
            logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

        print('Optimization Finished!')

        return model

    def _get_model_objective(self, model):

        """Return model objective.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        
        Returns
        -------
        object
            Result computed as ``objective_function``."""
        def _cost_energy_exchange_ac(model):
            # 交流电充电成本
            # AC charging losses are already included in power_grid_ac_consumption term.
            """Cost energy exchange ac.
            
            Parameters
            ----------
            model : object
                Model used by this operation.
            
            Returns
            -------
            object
                Result computed as ``sum((model.power_grid_ac_consumption[t] * 0.25 * model.price_cons_ac[t] for t in model.T))``."""
            return sum(model.power_grid_ac_consumption[t] * 0.25 * model.price_cons_ac[t] for t in model.T)

        def _cost_charging_dc(model):
            # 直流电充电成本
            # 通常直流快充单价 price_cons_dc 会比交流电贵，所以优化器会尽量少用它。
            """Cost charging dc.
            
            Parameters
            ----------
            model : object
                Model used by this operation.
            
            Returns
            -------
            object
                Result computed as ``sum((model.power_vehicle_dc_charging[t] * 0.25 * model.price_cons_dc[t] for t in model.T))``."""
            return sum(model.power_vehicle_dc_charging[t] * 0.25 * model.price_cons_dc[t] for t in model.T)

        # 软约束

        def _penalty_comfort_minimum_soc(model):
            # 比最低电量低的处罚 J1
            """Penalty comfort minimum soc.
            
            Parameters
            ----------
            model : object
                Model used by this operation.
            
            Returns
            -------
            object
                Result computed as ``sum((model.comfort_binary[t] for t in model.T))``."""
            return sum(model.comfort_binary[t] for t in model.T)

        def _penalty_terminal_soc(model):
            # 最终电量不达标惩罚, 如果优化结束时 (t=N) 电量没满，差了多少度电，terminal_soc_deviation 就是多少。
            """Penalty terminal soc.
            
            Parameters
            ----------
            model : object
                Model used by this operation.
            
            Returns
            -------
            object
                Result computed as ``model.terminal_soc_deviation``."""
            return (model.terminal_soc_deviation)

        def _penalty_dc_charging_duration(model):
            # 直流充电时长惩罚, 
            # 论文提到要让直流充电“尽快完成” (completed as quickly as possible) 
            # 只要在进行直流充电，dc_charging_binary 就是 1。累加这个值，就是让充电持续的时间越短越好
            """Penalty dc charging duration.
            
            Parameters
            ----------
            model : object
                Model used by this operation.
            
            Returns
            -------
            object
                Result computed as ``sum((model.dc_charging_binary[t] for t in model.T))``."""
            return sum(model.dc_charging_binary[t] for t in model.T)

        """
        Model objective is to minimize charging costs over optimization horizon, fulfilling the constraints.
        """

        optimization_config = self._optimization_config

        #define lagrangians
        lambda_minimum_soc = optimization_config.lambda_minimum_soc
        lambda_terminal_soc = optimization_config.lambda_terminal_soc
        lambda_dc_charging_duration = optimization_config.lambda_dc_charging_duration

        energy_exchange_cost = _cost_energy_exchange_ac(model) + _cost_charging_dc(model)
        penalty_deviation_minimum_soc = _penalty_comfort_minimum_soc(model)
        penalty_deviation_terminal_soc = _penalty_terminal_soc(model)
        penalty_dc_charging_duration = _penalty_dc_charging_duration(model)
        penalty_deviation_terminal_soc = _penalty_terminal_soc(model)
        objective_function = (energy_exchange_cost +
                              lambda_minimum_soc * penalty_deviation_minimum_soc +
                              lambda_terminal_soc * penalty_deviation_terminal_soc +
                              lambda_dc_charging_duration * penalty_dc_charging_duration)

        return objective_function

    def _battery_capacity_update_constraint(self, model, t):
        """
        Battery capacity update
        """
        # 18 in the papaer
        # AC exchange
        # P^EV_n - P^EV,L_n (从电网端抽取的功率 — 减去损耗。这是关键点！因为经过车载充电机（OBC）会有发热损耗，真正进入电池的能量要少于墙上抽取的能量。)
        ac_energy_exchange = (model.power_vehicle_ac_charging[t] - model.charging_idle_losses[t]) * 0.25

        # Energy exchange DC
        # 直流充电桩输出的功率 乘以 效率
        dc_energy_exchange = model.power_vehicle_dc_charging[t] * 0.25 * model.eta_dc

        # Model driving event
        # 开车消耗的能量
        driving_energy_consumption = model.driving_consumption[t]

        # 建立时间连续性
        # 17 in the paper
        if t == model.T.first():
            # 初始电量百分比 × 电池总容量 = 初始能量 (kWh) (soc) 然后加上充的，减去跑的
            charging_energy_exchange = model.soc_ini * model.bc_max + ac_energy_exchange + dc_energy_exchange
            return model.bc[t] == (charging_energy_exchange - driving_energy_consumption)
        else:
            # .bc[t-1] (soc[t-1])
            charging_energy_exchange = model.bc[t - 1] + ac_energy_exchange + dc_energy_exchange
            return model.bc[t] == (charging_energy_exchange - driving_energy_consumption)

    def _grid_balancing_constraint(self, model, t):
        # AC 电网侧的总消耗 = 汽车充电 + 家庭基础负荷 - 充电损耗
        # 如果没有这个约束，优化器可能会为了“省钱”，把 power_grid_ac_consumption 设为 0，假装不用从电网买电。这行代码强制规定：“想用电，就得如实记录在账上。”
        # (8) in the paper
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
            Computed result for the selected input and execution branch."""
        return (model.power_grid_ac_consumption[t] == model.power_vehicle_ac_charging[t]
                + model.hh_cons_kw[t] + model.charging_idle_losses[t])

    def _minimum_soc_constraint(self, model, t):
        """Minimum soc constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.bc[t] / model.bc_max - model.soc_min >= -bigM * model.comfort_binary[t]``."""
        optimization_config = self._optimization_config
        """
        Minimum SoC constraint as binary
        """
        # 11 and 12 in the paper
        # 这是一个**“即使死也要死个明白”**的机制。它防止模型直接崩溃，而是通过高额罚款告诉我们：“这里电量不够了，但我尽力了。
        # 原则上不允许电量低于设定的最小值（比如 60%），但如果情况万分紧急（无解），我允许你打破这个规则，不过我要重重地罚你。
        bigM = optimization_config.constraint_bigM
        return (model.bc[t] / model.bc_max - model.soc_min) >= -bigM * model.comfort_binary[t]

    def _terminal_soc_deviation_constraint(self, model, t):
        # 离满电还差多少
        # 它的核心目的是防止优化器为了省钱，在仿真结束的前一刻把电池里的电全卖光，导致用户第二天早上出门时车是空的。
        # 这对应论文中的 公式 (13) 以及相关的拉格朗日惩罚项 J2​
        # deviation = needed - actual
        """Terminal soc deviation constraint.
        
        Parameters
        ----------
        model : object
            Model used by this operation.
        t : object
            T used by this operation.
        
        Returns
        -------
        object
            Result computed as ``model.terminal_soc_deviation >= model.soc_term * model.bc_max - model.bc[model.T.last()]``."""
        return model.terminal_soc_deviation >= (model.soc_term * model.bc_max - model.bc[model.T.last()])

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
            Computed result for the selected input and execution branch."""
        optimization_config = self._optimization_config
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

        """
        in section H
        状态,功率范围,Binary 状态,损耗公式,结果
        关机,0,z1​=1 (其他为0),0+0+α1​⋅0,0 (无损耗)
        非法/过渡,ϵ∼2.0,z2​=1,Penalty +0+α1​⋅P,极大值 (禁止)
        正常充电,2.0∼Pmax​,z3​=1,0+ α0​ +α1​⋅P,线性损耗 (拟合直线)
        """

        alpha_0 = optimization_config.loss_linearization_alpha_0
        alpha_1 = optimization_config.loss_linearization_alpha_1
        penalty = optimization_config.loss_linearization_forbidden_penalty_factor

        return model.charging_idle_losses[t] == (penalty * model.ac_charging_threshold_binary_2[t] +
                                                 alpha_0 * model.ac_charging_threshold_binary_3[t] +
                                                 alpha_1 * model.power_vehicle_ac_charging[t])

    # 分成3个phase
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
        optimization_config = self._optimization_config
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac > epsilon --> binary == 0
        # if ac <= epsilon --> binary 0 or 1
        return (model.power_vehicle_ac_charging[t] - epsilon) <= bigM * (1-model.ac_charging_threshold_binary_1[t])

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
        optimization_config = self._optimization_config
        minimum_power_ac_charger_kw = optimization_config.minimum_power_ac_charger_kw
        bigM = optimization_config.constraint_bigM
        # if ac > 2.0 --> binary == 0
        # if ac <= 2.0 --> binary 0 or 1
        return (model.power_vehicle_ac_charging[t] - minimum_power_ac_charger_kw) <= bigM * (1-model.ac_charging_threshold_binary_2[t])

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
        optimization_config = self._optimization_config
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac < epsilon --> binary == 0
        # if ac >= epsilon --> binary 0 or 1
        return (epsilon - model.power_vehicle_ac_charging[t]) <= bigM * (1-model.ac_charging_threshold_binary_2[t])

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
        optimization_config = self._optimization_config
        minimum_power_ac_charger_kw = optimization_config.minimum_power_ac_charger_kw
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac < 2.0 --> binary == 0
        # if ac >= 2.0 --> binary 0 or 1
        return (minimum_power_ac_charger_kw - model.power_vehicle_ac_charging[t]) <= bigM * (1-model.ac_charging_threshold_binary_3[t])

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
        optimization_config = self._optimization_config
        minimum_power_ac_charger_kw = optimization_config.minimum_power_ac_charger_kw
        bigM = optimization_config.constraint_bigM
        epsilon = optimization_config.constraint_epsilon
        # if ac >= 2.0 --> binary == 1
        return (model.power_vehicle_ac_charging[t] - minimum_power_ac_charger_kw) <= bigM * model.ac_charging_threshold_binary_3[t]

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

    def _dc_charging_binary_constraint(self, model, t):
        """
        Binary becomes 1, if vehicle is in AC charging mode. Else it is 0.
        """
        #if vehicle dc charging is > 0, binary is 1
        #if vehicle dc charging is 0, binary is 1 or 0 //. Whenever possible, 1 will be avoided due to loss modeling
        #via cost function.
        # 15 in the paper
        optimization_config = self._optimization_config
        bigM = optimization_config.constraint_bigM
        return bigM * (model.dc_charging_binary[t]) >= (model.power_vehicle_dc_charging[t])

    def _max_ac_charging_power_constraint(self, model, t):
        """
        AC Charging power shall be <= than maximum ac charging power from wallbox at any timeslot t.
        """
        # 3 in the paper
        return model.power_vehicle_ac_charging[t] <= model.max_charging_ac[t]

    def _max_dc_charging_power_constraint(self, model, t):
        """
        DC Charging power shall be <= than maximum dc charging power from grid at timeslot t.
        """
        # 3 in the paper
        return model.power_vehicle_dc_charging[t] <= model.max_charging_dc[t]

    def _infeasible_constraint_logging(self, set):
        # 当模型“无解”（Infeasible）时，它像法医一样去检查“尸体”，告诉你是哪一条具体的规则导致了无解
        """Infeasible constraint logging.
        
        Parameters
        ----------
        set : object
            Set used by this operation."""
        if set == 'off':
            model = self._model
            logging_logger = logging.getLogger()
            logging_logger.setLevel(logging.DEBUG)
            #Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            #Add handler to the logger
            logging_logger.addHandler(ch)
            #Log infeasible constraints of pyomo object
            print('Displaying Infeasible Constraints:')
            log_infeasible_constraints(model, log_expression=True, log_variables=True, logger=logging_logger)
            logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    def model_to_df(self, timeslots_to_extract) -> pd.DataFrame:
        """Model to df.
        
        Parameters
        ----------
        timeslots_to_extract : object
            Timeslots to extract used by this operation.
        
        Returns
        -------
        pd.DataFrame
            Result computed as ``df.iloc[0:timeslots_to_extract]``."""
        model = self._model

        timeslot = list(range(model.T[1], len(model.T)))

        calender_week = model.calender_week
        driving_distance_km = model.driving_distance
        driving_consumption_kwh = model.driving_consumption
        power_vehicle_ac_charging = [model.power_vehicle_ac_charging[i].value for i in timeslot]
        power_vehicle_ac_feedin = list(np.zeros(len(calender_week)))
        power_vehicle_dc_charging = [model.power_vehicle_dc_charging[i].value for i in timeslot]
        household_consumption_kw = model.hh_cons_kw
        photovoltaic_generation_kw = list(np.zeros(len(calender_week)))
        power_grid_ac_consumption_kw = [model.power_grid_ac_consumption[i].value for i in timeslot]
        power_grid_ac_feedin_kw = list(np.zeros(len(calender_week)))
        charging_idle_losses = [model.charging_idle_losses[i].value for i in timeslot]
        feedin_idle_losses = list(np.zeros(len(calender_week)))
        bc = [model.bc[i].value for i in timeslot]
        max_charging_ac = model.max_charging_ac
        max_charging_dc = model.max_charging_dc
        price_cons_ac = model.price_cons_ac
        price_ac_solar_feedin = list(np.zeros(len(calender_week)))
        price_cons_dc = model.price_cons_dc
        ambient_temperature = [model.ambient_temp[i] for i in timeslot]
        comfort_penalty_true = [model.comfort_binary[i].value for i in timeslot]
        dc_charging_binary = [model.dc_charging_binary[i].value for i in timeslot]

        ac_charging_threshold_binary_1 = [model.ac_charging_threshold_binary_1[i].value for i in timeslot]
        ac_charging_threshold_binary_2 = [model.ac_charging_threshold_binary_2[i].value for i in timeslot]
        ac_charging_threshold_binary_3 = [model.ac_charging_threshold_binary_3[i].value for i in timeslot]

        df_dict = {'calender_week': calender_week,
                   'driving_distance_km': driving_distance_km,
                   'driving_consumption_kwh': driving_consumption_kwh,
                   'ac_vehicle_charging_kw': power_vehicle_ac_charging,
                   'ac_vehicle_feedin_kw': power_vehicle_ac_feedin,
                   'dc_vehicle_charging_kw': power_vehicle_dc_charging,
                   'household_consumption_kw': household_consumption_kw,
                   'photovoltaic_generation_kw': photovoltaic_generation_kw,
                   'ac_grid_consumption_kw': power_grid_ac_consumption_kw,
                   'ac_grid_feedin_kw': power_grid_ac_feedin_kw,
                   'charging_idle_losses_kw': charging_idle_losses,
                   'feedin_idle_losses_kw': feedin_idle_losses,
                   'battery_capacity_kwh': bc,
                   'max_ac_charging_kw': max_charging_ac,
                   'max_dc_charging_kw': max_charging_dc,
                   'price_consumption_ac_euro_kwh': price_cons_ac,
                   'price_feedin_ac_euro_kwh': price_ac_solar_feedin,
                   'price_consumption_dc_euro_kwh': price_cons_dc,
                   'ambient_temperature': ambient_temperature,
                   'comfort_penalty_true': comfort_penalty_true,
                   'dc_charging_binary': dc_charging_binary,
                   'ac_charging_threshold_binary_1': ac_charging_threshold_binary_1,
                   'ac_charging_threshold_binary_2': ac_charging_threshold_binary_2,
                   'ac_charging_threshold_binary_3': ac_charging_threshold_binary_3}

        df = pd.DataFrame(df_dict)

        return df.iloc[0:timeslots_to_extract]