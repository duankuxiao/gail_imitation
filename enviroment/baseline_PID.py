import os
import pandas as pd
import numpy as np
from utils.tools import solar_heat_gain_cal, create_directory
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from utils.tools import get_random_error
from enviroment.base_env import BaseEnv


class PID_control():
    def __init__(self, max=5200, min=0, Kp=666, Ki=64, Kd=24):
        self.max = max
        self.min = min
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 微分增益
        self.Kd = Kd  # 积分增益
        self.intergral = 0  # 直到上一次的误差值
        self.pre_error = 0  # 上一次的误差值
        self.error = 0
        self.dt = 1

    def Load_cal(self, setPoint, Ti):
        error = setPoint - Ti
        Pout = self.Kp * error

        self.intergral += error
        Iout = self.Ki * self.intergral

        derivative = error - self.pre_error
        Dout = self.Kd * derivative

        output = Pout + Iout + Dout
        if (output > self.max):
            output = self.max
        elif (output < self.min):
            output = 0
        self.pre_error = error
        return round(output)


class baseline_model_pid(BaseEnv):
    def __init__(self, config, pid_controller):
        super(baseline_model_pid, self).__init__(config, flag='test')
        rc_config = config['rc_config']
        rl_config = config['rl_config']
        self.Ci = rc_config.Ci
        self.Ce = rc_config.Ce
        self.Ro = rc_config.Ro
        self.Ri = rc_config.Ri
        self.Rg = rc_config.Rg
        self.Rn = rc_config.Rn
        self.Awindow = rc_config.Awindow
        self.Ai = rc_config.Ai
        self.Awall = rc_config.Awall
        self.Av = rc_config.Av
        self.battery_capacity = config['rc_config'].battery_capacity
        self.charge_capacity = config['rc_config'].charge_capacity
        self.discharge_capacity = config['rc_config'].discharge_capacity

        self.path = rl_config.test_data_path

        self.pid_controller = pid_controller
        self.T_range = rl_config.T_range
        self.Ti_init = rl_config.Ti_init
        self.price = rl_config.price
        self.dt = 300

        self.folder_path = r'..\results\baseline\{}_{}_Kp{}_Ki{}_Kd{}'.format(rl_config.price,self.ac_control,pid_controller.Kp, pid_controller.Ki, pid_controller.Kd)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.battery_power_init = 0
        self.switch_previous = 1

    def _rc_state_init(self) -> dict:
        if self.price == 'dynamic':
            price = self.inputs.loc[:, 'price']
        elif self.price == 'fix':
            price = self.inputs.loc[:, 'price_fix']
        elif self.price == 'normal':
            price = self.inputs.loc[:, 'price_normal']
        variables = ['hour','temp_target', 'To', 'phi_i', 'phi_wall_s', 'phi_window_s', 'home', 'Ec_other', 'Ec_pv']

        for var in variables:
            globals()[var] = self.inputs.loc[:, var]

        variables = ['Ti', 'Te', 'L', 'L_ratio', 'cost', 'Ec_ac', 'Ec_demand', 'Ec_sell', 'Ec_buy', 'Ec_true', 'battery', 'Ec_charge', 'CO2']

        for var in variables:
            globals()[var] = np.zeros(len(self.inputs))

        Ti[0] = self.Ti_init  # T_set[0]
        Te[0] = (self.Ri * To[0] + self.Ro * Ti[0]) / (self.Ri + self.Ro)
        Ec_demand[0] = Ec_other[0]
        CO2[0] = 0.000457 * 1000000 * Ec_demand[0] / 1000
        return dict(hour=hour, To=To, Ti=Ti, T_target=temp_target, Te=Te, L=L, L_ratio=L_ratio, phi_i=phi_i,
                    phi_wall_s=phi_wall_s, phi_window_s=phi_window_s, home=home, Ec_pv=Ec_pv, Ec_charge=Ec_charge, battery=battery,
                    Ec_other=Ec_other, Ec_ac=Ec_ac, Ec_demand=Ec_demand, Ec_sell=Ec_sell, Ec_buy=Ec_buy,
                    Ec_true=Ec_true, price=price, cost=cost, CO2=CO2)

    def baseline_cal(self) -> dict:
        _rc_state_dict = self._rc_state_init()
        for t in range(1, len(self.inputs)):
            t = t - 1

            error_L = 1 + get_random_error(t, max_error=0.05)
            error_T = 1 + get_random_error(t, max_error=0.05)

            switch_previous = _rc_state_dict['home'][t-1]
            switch_current = _rc_state_dict['home'][t]

            if switch_previous == 0 and switch_current == 1:
                _rc_state_dict['L'][t], _rc_state_dict['Ec_ac'][t] = 3000, 1000
                _rc_state_dict['L'][t] = np.clip(_rc_state_dict['L'][t] * error_L, 0, 5200)
            elif switch_current == 0:
                _rc_state_dict['L'][t], _rc_state_dict['Ec_ac'][t] = 0, 0
            else:
                L_pid = self.pid_controller.Load_cal(_rc_state_dict['T_target'][t], round(_rc_state_dict['Ti'][t], 1))
                _rc_state_dict['L'][t] = np.clip(L_pid * error_L, 0, 5200) if L_pid >= 700 else 0
                if self.ac_control == 'pid':
                    if _rc_state_dict['Ti'][t] >= _rc_state_dict['T_target'][t] + 1:
                        _rc_state_dict['L'][t] = 0
                    else:
                        # delta_T = _rc_state_dict['temp_target'][t] - _rc_state_dict['Ti'][t]
                        # _rc_state_dict['L'][t] = np.clip(-100.95 * delta_T ** 2 + 1422.9 * delta_T + 600.07, 0, 5200) if delta_T >= 0 else 0
                        L_pid = self.pid_controller.Load_cal(_rc_state_dict['T_target'][t], round(_rc_state_dict['Ti'][t], 1))
                        _rc_state_dict['L'][t] = np.clip(L_pid * error_L, 0, 5200) if L_pid >= 700 else 0
                elif self.ac_control == 'Tset':
                    current_state = {'Ti':_rc_state_dict['Ti'][t],'phi_i':_rc_state_dict['phi_i'][t],'To':_rc_state_dict['To'][t],'phi_window_s':_rc_state_dict['phi_window_s'][t],'Te':_rc_state_dict['Te'][t],}
                    _rc_state_dict['L'][t] = self._load_from_RC(_rc_state_dict['T_target'][t+1],current_state)
                L_ = np.clip(self.w_ec_ac * _rc_state_dict['L'][t], 0, 5200) if _rc_state_dict['L'][t] >= 700 else 0
                _rc_state_dict['Ec_ac'][t] = self.w_ec_ac * ((-5.3319 * 1e-3 * L_ - 3.4284) * _rc_state_dict['To'][t] + 3.5117 * 1e-5 * L_ ** 2 + 1.07457 * 1e-1 * L_ + 96.152) if _rc_state_dict['L'][t] >= 700 else 30

            _rc_state_dict['Ti'][t + 1] = _rc_state_dict['Ti'][t] + self.dt / self.Ci * ((_rc_state_dict['Te'][t] - _rc_state_dict['Ti'][t]) / self.Ri + (
                    18 - _rc_state_dict['Ti'][t]) / self.Rg + (22 - _rc_state_dict['Ti'][t]) / self.Rn + self.Awindow * _rc_state_dict['phi_window_s'][t] + self.Ai * (
                                                                                                 _rc_state_dict['phi_i'][t] + _rc_state_dict['L'][t]) + self.Av * (
                                                                                                     _rc_state_dict['To'][t] - _rc_state_dict['Ti'][t]))  # switch[t-1]

            _rc_state_dict['Te'][t + 1] = _rc_state_dict['Te'][t] + self.dt / self.Ce * ((_rc_state_dict['Ti'][t] - _rc_state_dict['Te'][t]) / self.Ri + (
                    _rc_state_dict['To'][t] - _rc_state_dict['Te'][t]) / self.Ro + self.Awall * _rc_state_dict['phi_wall_s'][t])

            _rc_state_dict['Ti'][t + 1] = _rc_state_dict['Ti'][t + 1] * error_T
            _rc_state_dict['Te'][t + 1] = _rc_state_dict['Te'][t + 1] * error_T

            _rc_state_dict['Ec_demand'][t] = _rc_state_dict['Ec_other'][t] + _rc_state_dict['Ec_ac'][t]
            _rc_state_dict['battery'][t], _rc_state_dict['Ec_charge'][t] = self._battery_action(pv=_rc_state_dict['Ec_pv'][t], ec_total=_rc_state_dict['Ec_demand'][t],
                                                                                             flag='baseline')
            _rc_state_dict['Ec_true'][t] = _rc_state_dict['Ec_demand'][t] - (_rc_state_dict['Ec_pv'][t] - _rc_state_dict['Ec_charge'][t])
            if _rc_state_dict['Ec_true'][t] >= 0:
                _rc_state_dict['Ec_sell'][t] = 0
                _rc_state_dict['Ec_buy'][t] = _rc_state_dict['Ec_true'][t]
                _rc_state_dict['cost'][t] = (self.dt / 3600) * _rc_state_dict['price'][t] * _rc_state_dict['Ec_true'][t] / 1000

            else:
                _rc_state_dict['Ec_sell'][t] = -_rc_state_dict['Ec_true'][t]
                _rc_state_dict['Ec_buy'][t] = 0
                _rc_state_dict['cost'][t] = (self.dt / 3600) * 15 * _rc_state_dict['Ec_true'][t] / 1000
            _rc_state_dict['CO2'][t] = 0.000457 * 1000000 * _rc_state_dict['Ec_buy'][t] / 1000 + 38 * _rc_state_dict['Ec_pv'][t] / 1000
        return _rc_state_dict


if __name__ == '__main__':
    from args import get_config

    config = get_config()
    rl_config = config['rl_config']
    rl_config.test_data_path = r'D:\gail_github\data\simulation_data_2018_2019.csv'
    rl_config.T_range = 1
    rl_config.price = 'normal'  # dynamic  fix  normal
    rl_config.ac_control = 'pid'  # none pid Tset
    # pid_controller = PID_control(Kp=2132, Ki=515, Kd=53)
    # pid_controller = PID_control(Kp=677, Ki=166, Kd=63)
    pid_controller = PID_control(Kp=520, Ki=165, Kd=60)

    # pid_controller = PID_control()

    bc1 = baseline_model_pid(config=config, pid_controller=pid_controller)
    res = bc1.baseline_cal()

    res, metrics = bc1.evaluation(res, output=True)
    bc1._figure(res)
