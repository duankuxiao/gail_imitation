import os
import pandas as pd
import numpy as np
from utils.tools import solar_heat_gain_cal, create_directory
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
from utils.tools import get_random_error
from enviroment.base_env import BaseEnv


class PID_control():
    def __init__(self, max=5000, min=0, Kp=666, Ki=64, Kd=24):
        self.max = max
        self.min = min
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 微分增益
        self.Kd = Kd  # 积分增益
        self.intergral = 0  # 直到上一次的误差值
        self.pre_error = 0  # 上一次的误差值
        self.error = 0

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

        self.folder_path = r'..\results\baseline\Kp{}_Ki{}_Kd{}'.format(pid_controller.Kp, pid_controller.Ki, pid_controller.Kd)
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

        variables = ['Ti', 'Te', 'L', 'L_ratio', 'cost', 'Ec_ac', 'Ec_demand', 'Ec_sell', 'Ec_buy', 'Ec_true', 'battery', 'charge', 'CO2']

        for var in variables:
            globals()[var] = np.zeros(len(self.inputs))
        Ti[0] = self.Ti_init  # T_set[0]
        Te[0] = (self.Ri * To[0] + self.Ro * Ti[0]) / (self.Ri + self.Ro)
        Ec_demand[0] = Ec_other[0]
        CO2[0] = 0.000457 * 1000000 * Ec_demand[0] / 1000
        return dict(hour=hour, To=To, Ti=Ti, temp_target=temp_target, Te=Te, L=L, L_ratio=L_ratio, phi_i=phi_i,
                    phi_wall_s=phi_wall_s, phi_window_s=phi_window_s, home=home, Ec_pv=Ec_pv, charge=charge, battery=battery,
                    Ec_other=Ec_other, Ec_ac=Ec_ac, Ec_demand=Ec_demand, Ec_sell=Ec_sell, Ec_buy=Ec_buy,
                    Ec_true=Ec_true, price=price, cost=cost, CO2=CO2)

    def baseline_cal(self) -> dict:
        _rc_state_dict = self._rc_state_init()
        switch_previous = 1
        for t in range(1, len(self.inputs)):
            t = t - 1

            error_L = 1 + get_random_error(t, max_error=0.05)
            error_T = 1 + get_random_error(t, max_error=0.05)

            # error_L, error_T = 1, 1

            switch_current = _rc_state_dict['home'][t]

            L_pid = self.pid_controller.Load_cal(_rc_state_dict['temp_target'][t], round(_rc_state_dict['Ti'][t], 1))
            _rc_state_dict['L'][t] = np.clip(L_pid * error_L, 0, 5000) if L_pid >= 600 else 0

            switch_current, Tset, _rc_state_dict['L'][t], _rc_state_dict['Ec_ac'][t] = self._ac_action(switch_current, _rc_state_dict['L'][t], _rc_state_dict['Ti'][t],
                                                                                                       _rc_state_dict['To'][t])

            _rc_state_dict['Ti'][t + 1] = _rc_state_dict['Ti'][t] + self.dt / self.Ci * ((_rc_state_dict['Te'][t] - _rc_state_dict['Ti'][t]) / self.Ri + (
                    18 - _rc_state_dict['Ti'][t]) / self.Rg + (22 - _rc_state_dict['Ti'][t]) / self.Rn + self.Awindow * _rc_state_dict['phi_window_s'][t] + self.Ai * (
                                                                                                 _rc_state_dict['phi_i'][t] + _rc_state_dict['L'][t]) + self.Av * (
                                                                                                     _rc_state_dict['To'][t] - _rc_state_dict['Ti'][t]))  # switch[t-1]

            _rc_state_dict['Te'][t + 1] = _rc_state_dict['Te'][t] + self.dt / self.Ce * ((_rc_state_dict['Ti'][t] - _rc_state_dict['Te'][t]) / self.Ri + (
                    _rc_state_dict['To'][t] - _rc_state_dict['Te'][t]) / self.Ro + self.Awall * _rc_state_dict['phi_wall_s'][t])

            _rc_state_dict['Ti'][t + 1] = _rc_state_dict['Ti'][t + 1] * error_T
            _rc_state_dict['Te'][t + 1] = _rc_state_dict['Te'][t + 1] * error_T

            _rc_state_dict['Ec_demand'][t] = _rc_state_dict['Ec_other'][t] + _rc_state_dict['Ec_ac'][t]
            _rc_state_dict['battery'][t], _rc_state_dict['charge'][t] = self._battery_action(pv=_rc_state_dict['Ec_pv'][t], ec_total=_rc_state_dict['Ec_demand'][t],
                                                                                             flag='baseline')
            _rc_state_dict['Ec_true'][t] = _rc_state_dict['Ec_demand'][t] - (_rc_state_dict['Ec_pv'][t] - _rc_state_dict['charge'][t])
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

    def _figure(self, res):
        res = res.iloc[15553:15553 + 24 * 7 * 12, :]
        plt.figure(1, figsize=(20, 10))
        plt.plot(res['temp_target'], color="deepskyblue", ms=5, label='Target')
        plt.plot(res['Ti'], color="darkorange", marker='o', ms=5, label='Ti')
        plt.title('Simulation Res', fontsize=20)
        plt.xlabel('Time[5min]', fontsize=20)
        plt.ylabel('Temp[℃]', fontsize=20)
        text = "Kp{} Ki{} Kd{}".format(self.pid_controller.Kp, self.pid_controller.Ki, self.pid_controller.Kd)
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                 fontsize=20, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.ylim((20,24))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        # plt.figure(2, figsize=(20, 10))
        # plt.plot(res['Ec_ac'].iloc[15553:15553+24*7*12], color="deepskyblue", label='Ec_ac')
        # plt.plot(res['Ec_true'].iloc[15553:15553+24*7*12], color="darkorange", label='Ec_true')
        # plt.plot(res['Ec_total'].iloc[15553:15553+24*7*12], color="chartreuse", label='Ec_total')
        # plt.xlabel('time[{}min]'.format(5), fontsize=20)
        # plt.ylabel('Electricity consumption [W]', fontsize=20)
        # plt.xlim(-0.5)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.legend(fontsize=20)
        # plt.figure(3, figsize=(20, 10))
        # plt.plot(res['battery'].iloc[15553:15553+24*7*12], color="deepskyblue", label='battery')
        # plt.plot(res['charge'].iloc[15553:15553+24*7*12], color="darkorange", label='charge')
        # plt.plot(res['discharge'].iloc[15553:15553+24*7*12], color="chartreuse", label='discharge')
        # plt.xlabel('time[{}min]'.format(5), fontsize=20)
        # plt.ylabel('Battery', fontsize=20)
        # plt.xlim(-0.5)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.legend(fontsize=20)
        list = ['L', 'cost']
        list = ['L']
        for col in list:
            plt.figure(list.index(col) + 4, figsize=(20, 10))
            plt.plot(res[col], color="deepskyblue", ms=2, label=col)
            plt.xlabel('time[{}min]'.format(5), fontsize=20)
            plt.ylabel('{}'.format(col), fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
        plt.show()


if __name__ == '__main__':
    from args import get_config

    config = get_config()
    rl_config = config['rl_config']
    rl_config.test_data_path = r'D:\gail_github\data\simulation_data_2018_2019.csv'
    rl_config.T_range = 1
    rl_config.price = 'fix'  # dynamic  fix  normal
    # pid_controller = PID_control(Kp=2132, Ki=515, Kd=53)
    # pid_controller = PID_control(Kp=670, Ki=166, Kd=60)
    pid_controller = PID_control(Kp=1000, Ki=0.01, Kd=0)

    # pid_controller = PID_control()

    bc1 = baseline_model_pid(config=config, pid_controller=pid_controller)
    res = bc1.baseline_cal()

    res, metrics = bc1.evaluation(res, output=True)
    bc1._figure(res)
