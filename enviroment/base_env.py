import random
import numpy as np
import pandas as pd
import os
# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import torch
from utils.rl_tools import closest_number
from utils.tools import get_random_error
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


class PID_control():
    def __init__(self, max=5000, min=0, Kp=520, Ki=165, Kd=60):
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


class BaseEnv(gym.Env):
    def __init__(self, config, flag):
        self.flag = flag

        rl_config = config['rl_config']
        self._set_up(rl_config)
        self.inputs = self._read_inputs(flag)
        self._rc_parameters(config['rc_config'])

        self.t = 1
        self._reward_parameter(rl_config)
        self._init_action_space()
        self._init_observation_space()

    def _set_up(self, rl_config):
        self.train_data_path = rl_config.train_data_path
        self.test_data_path = rl_config.test_data_path

        self.use_pv_forecast = rl_config.use_pv_forecast
        self.use_next_state = rl_config.use_next_state

        self.ac_control = rl_config.ac_control
        self.obs_space = rl_config.obs
        self.price = rl_config.price
        self.Ti_init = rl_config.Ti_init
        self.timesteps = rl_config.timesteps
        self.total_timesteps = rl_config.timesteps

        self.T_range = rl_config.T_range
        self.T_delta = rl_config.T_delta
        self.battery_power_init = 0
        self.switch_previous = 1
        self.action_init = [1, 0.2, 0]
        if self.ac_control == 'pid':
            self.pid_controller = PID_control()

    def _reward_parameter(self, rl_config):
        self.wEc = rl_config.wEc
        self.wT = rl_config.wT
        self.c = rl_config.c
        self.wSell = rl_config.wSell
        self.wPV = rl_config.wPV
        self.wCO = rl_config.wCO
        self.wBuy = rl_config.wBuy
        self.wSSR = rl_config.wSSR
        self.w_ec_ac = rl_config.w_ec_ac
        self.w1 = rl_config.w1
        self.w2 = rl_config.w2
        self.w3 = rl_config.w3
        self.w4 = rl_config.w4
        self.w5 = rl_config.w5



    def _rc_parameters(self, rc_config):
        self.dt = rc_config.dt
        self.Ci = rc_config.Ci
        self.Ce = rc_config.Ce
        self.Ro = rc_config.Ro
        self.Ri = rc_config.Ri
        self.Rn = rc_config.Rn
        self.Rg = rc_config.Rg
        self.Awindow = rc_config.Awindow
        self.Ai = rc_config.Ai
        self.Awall = rc_config.Awall
        self.Av = rc_config.Av
        self.battery_capacity = rc_config.battery_capacity
        self.charge_capacity = rc_config.charge_capacity
        self.discharge_capacity = rc_config.discharge_capacity
        self.Ti_next, self.Te_next = 22, 22

    def _init_action_space(self):
        self.act_low = np.array([-1, 0, -self.discharge_capacity])
        self.act_high = np.array([1, 5000, self.charge_capacity])
        if self.ac_control == 'pid' or self.ac_control == 'Tset':
            self.act_low = np.array([-1, 16, -self.discharge_capacity])
            self.act_high = np.array([1, 28, self.charge_capacity])
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]).astype(np.float64), high=np.array([1, 1, 1]).astype(np.float64), dtype=np.float64)

    def _init_observation_space(self):
        observation_space = spaces.Dict(
            {
                'hour': spaces.Box(low=0, high=23, dtype=np.float64),
                'battery': spaces.Box(low=0, high=self.battery_capacity, dtype=np.float64),
                'switch': spaces.Box(low=-1, high=1, dtype=np.float64),
                'To': spaces.Box(low=-10, high=50, dtype=np.float64),
                'price': spaces.Box(low=0, high=50, dtype=np.float64),
                'Ec_pv': spaces.Box(low=0, high=3900, dtype=np.float64),
                'Ec_demand': spaces.Box(low=0, high=3000, dtype=np.float64),
                'Ti': spaces.Box(low=10, high=40, dtype=np.float64),

                # 'workday': spaces.Box(low=0, high=1, dtype=np.float64),
                # 'T_target': spaces.Box(low=16, high=28, dtype=np.float64),
                # 'home': spaces.Box(low=0, high=1, dtype=np.float64),
                # 'Ec_sell': spaces.Box(low=0, high=3000, dtype=np.float64),
                # 'Ec_buy': spaces.Box(low=0, high=3000, dtype=np.float64),
                # 'Tset': spaces.Box(low=16, high=28, dtype=np.float64),
            }
        )
        self.current_observation_space = observation_space
        if self.use_next_state:
            self.next_observation_space = spaces.Dict(
                {
                    # 'hour_next': spaces.Box(low=0, high=23, dtype=np.float64),
                    # 'To_next': spaces.Box(low=-10, high=50, dtype=np.float64),
                    # 'price_next': spaces.Box(low=0, high=50, dtype=np.float64),
                    # 'Ec_pv_next': spaces.Box(low=0, high=3900, dtype=np.float64),
                    # 'workday_next': spaces.Box(low=0, high=1, dtype=np.float64),
                    # 'Ti_next': spaces.Box(low=10, high=40, dtype=np.float64),
                    # 'T_target_next': spaces.Box(low=16, high=28, dtype=np.float64),
                    # 'home_next': spaces.Box(low=0, high=1, dtype=np.float64),

                    # 'price_next': spaces.Box(low=0, high=50, dtype=np.float64),
                    # 'hour': spaces.Box(low=0, high=23, dtype=np.float64),
                    # 'To_next': spaces.Box(low=-10, high=50, dtype=np.float64),
                    # 'battery_next': spaces.Box(low=0, high=self.battery_capacity, dtype=np.float64),

                }
            )
            observation_space = spaces.Dict({**observation_space.spaces, **self.next_observation_space.spaces})
        if self.use_pv_forecast:
            self.pv_forecast_observation_space = spaces.Dict()
            for i in range(12):
                self.pv_forecast_observation_space[f'PV_future_{i + 1}'] = spaces.Box(low=0, high=3900, dtype=np.float64)
            observation_space = spaces.Dict({**observation_space.spaces, **self.pv_forecast_observation_space.spaces})

        if self.obs_space == 'box':
            obs_low = []
            obs_high = []

            # 遍历 _observation_space_dict 定义的每个空间，提取其上下限
            for key, space in observation_space.items():
                if isinstance(space, spaces.Box):  # 确保是 Box 类型
                    obs_low.append(space.low[0])  # 取低限值
                    obs_high.append(space.high[0])  # 取高限值

            # 将低限值和高限值转化为 numpy 数组，构建 Box 类型空间
            self.obs_low = np.array(obs_low, dtype=np.float64)
            self.obs_high = np.array(obs_high, dtype=np.float64)
            observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.observation_space = observation_space

    def test(self, flag: str) -> dict:
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError()

    def step(self, action: np.ndarray):
        raise NotImplementedError()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int = 9743):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.

    def _read_inputs(self, flag) -> pd.DataFrame:
        path = self.train_data_path if flag == 'train' else self.test_data_path
        inputs = pd.read_csv(path, index_col=0).iloc[:46081, :]
        inputs.index = pd.to_datetime(inputs.index)
        inputs['Ec_pv'] = inputs['PV']
        inputs['Ec_other'] = inputs['Ec_demand'] * 0.5
        inputs['phi_i'] = inputs['phi_i'] + 0.5 * inputs['phi_i'] * (inputs['home'] == 1)
        return inputs

    def _denormalize_act(self, action: np.ndarray) -> np.ndarray:
        return action * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2

    def _normalize_act(self, action: np.ndarray) -> np.ndarray:
        return (action - ((self.act_high + self.act_low) / 2)) / ((self.act_high - self.act_low) / 2)

    def _denormalize_obs_dict(self, observation: dict) -> dict:
        denorm_observation = {}
        for key in observation:
            if isinstance(self.observation_space.spaces[key], gym.spaces.Box):
                low = self.observation_space.spaces[key].low
                high = self.observation_space.spaces[key].high
                # 执行反归一化操作
                denorm_observation[key] = observation[key] * (high - low) / 2 + (high + low) / 2
        return denorm_observation

    def _normalize_obs_dict(self, observation: dict) -> dict:
        norm_observation = {}
        for key in observation:
            if isinstance(self.observation_space.spaces[key], gym.spaces.Box):
                low = self.observation_space.spaces[key].low
                high = self.observation_space.spaces[key].high
                norm_observation[key] = (observation[key] - ((high + low) / 2)) / ((high - low) / 2)
        return norm_observation

    def _denormalize_obs(self, observation: np.ndarray) -> np.ndarray:
        return observation * ((self.obs_high - self.obs_low) / 2) + ((self.obs_high + self.obs_low) / 2)

    def _normalize_obs(self, observation: np.ndarray) -> np.ndarray:
        return (observation - ((self.obs_high + self.obs_low) / 2)) / ((self.obs_high - self.obs_low) / 2)

    def _load_from_RC(self,Tset,current_state):
        L = ((Tset-current_state['Ti'])*self.Ci /self.dt - (self.Av * (current_state['To'] - current_state['Ti']) + (current_state['Te'] - current_state['Ti']) / self.Ri + (18 - current_state['Ti']) / self.Rg + (22 - current_state['Ti']) / self.Rn + self.Awindow * current_state[
            'phi_window_s']))/self.Ai - current_state['phi_i']
        return L

    def _RC(self, current_state):
        Ti_next = current_state['Ti'] + self.dt / self.Ci * (
                (current_state['Te'] - current_state['Ti']) / self.Ri + (18 - current_state['Ti']) / self.Rg + (22 - current_state['Ti']) / self.Rn + self.Awindow * current_state[
            'phi_window_s'] + self.Ai * (current_state['phi_i'] + current_state['L']) + self.Av * (current_state['To'] - current_state['Ti']))
        Te_next = current_state['Te'] + self.dt / self.Ce * (
                (current_state['Ti'] - current_state['Te']) / self.Ri + (current_state['To'] - current_state['Te']) / self.Ro + self.Awall * current_state['phi_wall_s'])
        return Ti_next,Te_next

    def _update_state_RC(self, t: int, action: np.ndarray) -> (dict, dict, dict):
        obs_variables = {}
        current_state = self._rc_state_init(t)
        next_state = self._rc_state_init(t + 1)

        error_L = 1 + get_random_error(t, max_error=0.03)
        error_T = 1 + get_random_error(t, max_error=0.03)

        current_state['t'] = t
        current_state['Ti'], current_state['Te'] = self.Ti_next, self.Te_next

        current_state['switch'], current_state['Tset'], current_state['L'], current_state['Ec_ac'] = self._ac_action(action[0], action[1], current_state['Ti'], current_state['To'],current_state)
        current_state['L'] = np.clip(current_state['L'] * error_L, 0, 5200)
        Ti_,Te_ = self._RC(current_state)

        self.Ti_next = Ti_ * error_T
        self.Te_next = Te_ * error_T

        next_state['Ti'] = Ti_
        next_state['Te'] = Te_

        current_state['battery'], current_state['Ec_charge'] = self._battery_action(action[2], self.Ec_pv, self.Ec_demand)
        current_state['Ec_true'] = current_state['Ec_demand'] - (current_state['Ec_pv'] - current_state['Ec_charge'])

        current_state['Ec_demand'] = current_state['Ec_ac'] + current_state['Ec_other']
        self.Ec_demand = current_state['Ec_demand']
        self.Ec_pv = current_state['Ec_pv']


        if current_state['Ec_true'] >= 0:
            current_state['Ec_sell'] = 0
            current_state['Ec_buy'] = current_state['Ec_true']
            current_state['cost'] = (self.dt / 3600) * current_state['price'] * current_state['Ec_true'] / 1000

        else:
            current_state['Ec_sell'] = -current_state['Ec_true']
            current_state['Ec_buy'] = 0
            current_state['cost'] = (self.dt / 3600) * 15 * current_state['Ec_true'] / 1000

        current_state['CO2'] = 0.000457 * 1000000 * current_state['Ec_buy'] / 1000 + 38 * current_state['Ec_pv'] / 1000

        if self.obs_space == 'dict':
            for key in self.current_observation_space:
                obs_variables[key] = current_state[key]
            if self.use_next_state:
                for key in self.next_observation_space:
                    obs_variables[key] = next_state[key[:-5]]
            if self.use_pv_forecast:
                for key in self.pv_forecast_observation_space:
                    step = int(key.split('_')[-1])  # 提取未来步数
                    future_idx = (t + step) % len(self.inputs)  # 计算未来索引
                    if future_idx >= len(self.inputs):  # 超出数据长度时，设置 PV 为 0
                        obs_variables[key] = 0.0
        else:
            obs_list = []
            for key in self.current_observation_space:
                obs_list.append(current_state[key])
            if self.use_next_state:
                for key in self.next_observation_space:
                    obs_list.append(next_state[key[:-5]])

            if self.use_pv_forecast:
                for step in range(1, 13):
                    future_idx = t + step
                    if future_idx >= len(self.inputs):  # 超出数据长度时，设置 PV 为 0
                        obs_list.append(0.0)
                    else:
                        obs_list.append(self.inputs['PV'].iloc[future_idx])
            obs_variables = np.array(obs_list, dtype=np.float64)
        reward_states = {'cost': current_state['cost'], 'battery': current_state['battery'],'Ec_pv': current_state['Ec_pv'],'Ec_demand': current_state['Ec_demand'],
                         'Ec_charge': current_state['Ec_charge'],'Ec_sell': current_state['Ec_sell'],'Ec_buy': current_state['Ec_buy'],'CO2': current_state['CO2'],
                         'home':next_state['home'],'Ti': self.Ti_next,'T_target':next_state['T_target'],'switch':current_state['switch']}
        return obs_variables, current_state, reward_states

    def _battery_action(self, Ec_charge: np.float32 = 0.0, pv: float = 0.0, ec_total: float or int = 0.0, flag: str = 'rl') -> (float, float, float):  # 0 <= self.battery_power <= 20
        if flag == 'rl':
            # if pv > 0:
            #     Ec_charge = np.clip(Ec_charge, 0, min((self.battery_capacity - self.battery_power_init) * 12, self.charge_capacity))
            # else:
            #     Ec_charge = np.clip(Ec_charge,-min(float(self.battery_power_init) * 12, self.discharge_capacity),0)

            Ec_charge = np.clip(Ec_charge, -min(float(self.battery_power_init) * 12, self.discharge_capacity),
                             min((self.battery_capacity - self.battery_power_init) * 12, self.charge_capacity))

        else:
            if pv > 0:
                Ec_charge = np.clip(pv - ec_total, 0, min((self.battery_capacity - self.battery_power_init) * 12, pv, self.charge_capacity))
            else:
                Ec_charge = -min(float(self.battery_power_init) * 12, ec_total, self.discharge_capacity)
        Ec_charge *= 0.97
        battery_power = np.clip(self.battery_power_init + Ec_charge * (self.dt / 3600), 0, self.battery_capacity)
        self.battery_power_init = round(battery_power)
        return round(battery_power), round(Ec_charge)

    def _ac_action(self, switch: int = -1, action: float = 0.0, Ti: float = 22, To: float = 16.0,current_state:dict=None) -> (int, int, float, float):
        switch_current = 1 if switch > 0 else -1
        if self.switch_previous == -1 and switch_current == 1:
            L, Ec_ac = 3000, 1000
            delta_T = 0.1851 * np.exp(1.7194 * (L / 2500))
            value_list = [i * 0.5 for i in range(32, 57)]
            Tset = min(value_list, key=lambda x: abs(x - (delta_T + Ti)))
        elif switch_current == -1:
            L, Ec_ac, Tset = 0, 0, 0
        else:
            L = action
            delta_T = 0.1851 * np.exp(1.7194 * (L / 2500))
            value_list = [i * 0.5 for i in range(32, 57)]
            Tset = min(value_list, key=lambda x: abs(x - (delta_T + Ti)))
            L = np.clip(L, 0, 5000) if L >= 600 else 0
            if self.ac_control == 'pid':
                Tset = closest_number(action)
                L = self.pid_controller.Load_cal(float(Tset), float(Ti))
            elif self.ac_control == 'Tset':
                Tset = closest_number(action)
                L = self._load_from_RC(Tset,current_state)
            L = np.clip(L, 0, 5200) if L >= 700 else 0
            L_ = np.clip(self.w_ec_ac * L, 0, 5200) if L >= 700 else 0
            Ec_ac = np.clip(self.w_ec_ac*((-5.3319 * 1e-3 * L_ - 3.4284) * To + 3.5117 * 1e-5 * L_ ** 2 + 1.07457 * 1e-1 * L_ + 96.152), 0, 1500) if L >= 700 else 30
        self.switch_previous = switch_current
        return switch_current, Tset, L, Ec_ac

    def _rc_state_init(self, t: int) -> (dict, dict):
        # print('---------------------------',t,'-----------------------------')
        if t >= len(self.inputs):
            t = t - len(self.inputs)
        To = self.inputs['To'].iloc[t]
        Ti = self.inputs['Ti'].iloc[t]
        phi_window_s = self.inputs['phi_window_s'].iloc[t]
        phi_wall_s = self.inputs['phi_wall_s'].iloc[t]
        phi_i = self.inputs['phi_i'].iloc[t]
        price = self.inputs['price_' + self.price].iloc[t]
        price = price if price <= 50 else 50
        Ec_other = self.inputs['Ec_other'].iloc[t]
        Ec_pv = self.inputs['Ec_pv'].iloc[t]

        workday = self.inputs['workday'].iloc[t]
        T_target = self.inputs['temp_target'].iloc[t]
        home = self.inputs['home'].iloc[t]
        dayofyear = self.inputs['dayofyear'].iloc[t]
        hour = self.inputs['hour'].iloc[t]
        state = dict(t=t, To=To, phi_i=phi_i, phi_wall_s=phi_wall_s, phi_window_s=phi_window_s, Ec_other=Ec_other, Ec_pv=Ec_pv, Ti=Ti,
                         price=price, T_target=T_target, home=home, dayofyear=dayofyear, hour=hour, workday=workday)
        return state

    def evaluation(self, res, output=False):
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)
        res['L'] = res['L'] * 0.8
        if self.price == 'fix':
            sr_baseline, cost_baseline = 0.902, 15167  # 0.7773 25303
        elif self.price == 'normal':
            sr_baseline, cost_baseline = 0.902, 10177  # 0.7773 8772
        else:
            sr_baseline, cost_baseline = 0.902, 24377  # 0.7773 -2890
        selected_rows = res[res['home'] == 1]
        rate = round((1-res['Ec_sell'].sum() / res['Ec_pv'].sum()) * 100,2)
        mape = mean_absolute_percentage_error(selected_rows['T_target'], selected_rows['Ti'])
        mse = mean_squared_error(selected_rows['T_target'], selected_rows['Ti'])
        Ec_demand = ((res['Ec_demand'].sum())/12/1000)
        Ec_true = ((res['Ec_true'].sum())/12/1000)
        Ec_max = res['Ec_demand'].max()
        CO2 = round(((res['CO2'].sum())/12)/1000)
        cost = res['cost'].sum()
        # print(res)
        filtered_df = res[res['home'] == 1]
        count_within_range = ((filtered_df['Ti'] - filtered_df['T_target']).abs() <= self.T_range).sum()
        sr = count_within_range / len(filtered_df) if len(filtered_df) > 0 else 0

        sr_eval = (sr - sr_baseline) / sr_baseline * 100
        cost_eval = (cost_baseline - cost) / cost_baseline * 100

            # print('Test_reward = %.1f' % (test_reward))
        metrics = {'mape': mape, 'mse': mape, 'sr': sr, 'Ec_demand': Ec_demand,'Ec_true':Ec_true, 'Ec_max':Ec_max, 'cost': cost, 'sr_improve': sr_eval, 'cost_improve': cost_eval,'rate':rate,'CO2': CO2 }
        if output:
            res.to_csv(os.path.join(self.folder_path,'sr_eval{}_cost_eval{}_mape{}_sr{}_Ecd{}_cost{}_rate{}_CO{}.csv'.format(round(sr_eval, 2), round(cost_eval, 2), round(mape * 100, 2),
                                                                                                     round(sr * 100, 2), round(Ec_demand, 2), round(cost, 2),rate,CO2)))
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).T
            metrics_df.to_csv(os.path.join(self.folder_path, 'metrics_mape{}_sr{}_Ecd{}_cost{}_rate{}_CO{}.csv'.format( round(mape * 100, 2),
                                                                                                     round(sr * 100, 2), round(Ec_demand, 2), round(cost, 2),rate,CO2)))
        blue_start = "\033[94m"
        blue_end = "\033[0m"
        print(blue_start + 'sr_eval{}_cost_eval{}_mape{}_sr{}_Ecd{}_cost{}_rate{}_CO{}kg.csv'.format(round(sr_eval, 2), round(cost_eval, 2), round(mape * 100, 2),
                                                                                                     round(sr * 100, 2), round(Ec_demand, 2), round(cost, 2),round(rate,2),CO2) + blue_end)
        self._figure(res)
        return res,metrics

    def res_figure(self, res):
        plt.figure(0, figsize=(20, 10))
        plt.plot(res['To'], color="deepskyblue", marker='*', ms=5, label='To')
        plt.plot(res['Ti'], color="darkorange", marker='o', ms=5, label='Ti')
        plt.plot(res['T_target'], color="chartreuse", label='Target')
        plt.xlabel('time[{}min]'.format(5), fontsize=20)
        plt.ylabel('Temp', fontsize=20)
        plt.xlim(-0.5)
        plt.title('{}'.format(self.setting))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        list = ['L', 'Ec_true']
        for col in list:
            plt.figure(list.index(col) + 1, figsize=(20, 10))
            plt.plot(res[col], color="deepskyblue", marker='*', ms=2, label=col)
            plt.xlabel('time[{}min]'.format(5), fontsize=20)
            plt.ylabel('{}'.format(col), fontsize=20)
            plt.xlim(-0.5)
            plt.title('{}'.format(self.setting))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
        plt.show()

    def _figure(self, res):
        res = res.iloc[15553:15553 + 24 * 7 * 12, :]
        plt.figure(1, figsize=(20, 10))
        plt.plot(res['T_target'], color="deepskyblue", ms=5, label='Target')
        plt.plot(res['Ti'], color="darkorange", marker='o', ms=5, label='Ti')
        plt.title('Simulation Res', fontsize=20)
        plt.xlabel('Time[5min]', fontsize=20)
        plt.ylabel('Temp[℃]', fontsize=20)
        # text = "Kp{} Ki{} Kd{}".format(self.pid_controller.Kp, self.pid_controller.Ki, self.pid_controller.Kd)
        # plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
        #          fontsize=20, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.ylim((20,24))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
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