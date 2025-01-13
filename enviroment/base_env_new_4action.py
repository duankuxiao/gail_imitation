import random
import numpy as np
import pandas as pd
# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import torch
from enviroment.baseline_PID import PID_control
from utils.rl_tools import closest_number
from utils.tools import get_random_error


class BaseEnv(gym.Env):
    def __init__(self, config,flag):
        self.flag = flag

        rl_config = config['rl_config']
        self._set_up(rl_config)
        self.inputs = self._read_inputs(flag)

        self._rc_parameters(config['rc_config'])
        self.t = 1
        self._reward_parameter(rl_config)
        self._action_space()
        self._observation_space() if self.obs_space == 'box' else self._observation_space_dict()

    def _set_up(self,rl_config):
        self.train_data_path = rl_config.train_data_path
        self.test_data_path = rl_config.test_data_path

        self.pid = rl_config.pid
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
        self.current_fea = ['L', 'battery', 'charge','Ec_ac','Ec_pv','Ec_sell','Ec_buy','Ec_demand','Ec_true','cost','switch','Tset']
        self.pred_fea = ['PV','To','home']
        if self.pid:
            self.pid_controller = PID_control()

    def _reward_parameter(self,rl_config):
        self.wEc = rl_config.wEc
        self.wT = rl_config.wT
        self.c = rl_config.c
        self.wS = rl_config.wS
        self.cB = rl_config.cB
        self.wPV = rl_config.wPV
        self.wC = rl_config.wC

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


    def _action_space(self):
        self.act_low = np.array([-1, 0, -self.discharge_capacity])
        self.act_high = np.array([1, 5000, self.charge_capacity])
        if self.pid:
            self.act_low = np.array([-1, 16, -self.discharge_capacity])
            self.act_high = np.array([1, 28, self.charge_capacity])
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]).astype(np.float64), high=np.array([1, 1, 1]).astype(np.float64), dtype=np.float64)

    def _observation_space_dict(self):
        self.observation_space = spaces.Dict(
            {'hour': spaces.Box(low=0, high=23, dtype=np.float64),
             'workday': spaces.Box(low=0, high=1, dtype=np.float64),
             'home': spaces.Box(low=0, high=1, dtype=np.float64),
             'battery': spaces.Box(low=0, high=self.battery_capacity, dtype=np.float64),
             'switch': spaces.Box(low=-1, high=1, dtype=np.float64),

             'To': spaces.Box(low=-10, high=50, dtype=np.float64),
              # 'charge': spaces.Box(low=-self.discharge_capacity, high=self.charge_capacity, dtype=np.float64),
              'price': spaces.Box(low=0, high=50, dtype=np.float64),
              # 'L': spaces.Box(low=0, high=5000, dtype=np.float64),
             'Ec_pv': spaces.Box(low=0, high=3900, dtype=np.float64),
             'Ec_sell': spaces.Box(low=0, high=10000, dtype=np.float64),
             'Ec_buy': spaces.Box(low=0, high=10000, dtype=np.float64),
             'Ec_demand': spaces.Box(low=0, high=10000, dtype=np.float64),
             })
        if self.T_delta:
            self.observation_space['T_delta'] = spaces.Box(low=-20, high=20, dtype=np.float64)
        else:
            self.observation_space['Ti'] = spaces.Box(low=-20, high=50, dtype=np.float64)
            self.observation_space['temp_target'] = spaces.Box(low=16, high=30, dtype=np.float64)

    def _observation_space(self):
        self.obs_low = np.array([0,    0, 0,                     0, -1,-10,    0,      0,    0,    0,0,-20,16]).astype(np.float64)
        self.obs_high = np.array([23,  1, 1, self.battery_capacity,  1, 50,   50,   3900,10000,10000,10000, 50,30]).astype(np.float64)
        '''
        [rc_state['hour'] 0-23,  rc_state['workday'] 0-1, rc_state['home'] 0-1, 
         rc_state['battery'] 0-battery_capacity, rc_state['switch'] -1-1, rc_state['To'] 0-25,rc_state['PV'] 0-3900,rc_state['Ec_total'] 0-4000, 
         rc_state['price'] 0-50,rc_state['Ti'] -10-50, rc_state['temp_target'] 16-28]
        '''

        self.observation_space = spaces.Box(low=-np.ones(len(self.obs_low)), high=np.ones(len(self.obs_high)), dtype=np.float64)
        # self.observation_space = Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)

    def test(self, flag: str) -> dict:
        raise NotImplementedError

    def reset(self,**kwargs):
        raise NotImplementedError()

    def step(self, action: np.ndarray):
        raise NotImplementedError()

    def reward_function(self, fea_variables):
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

    def _read_inputs(self,flag) -> pd.DataFrame:
        path = self.train_data_path if flag == 'train' else self.test_data_path
        inputs = pd.read_csv(path, index_col=0).iloc[:46081, :]
        inputs.index = pd.to_datetime(inputs.index)
        inputs['PV'] = inputs['PV'] * 1.5
        inputs['Ec_demand'] = inputs['Ec_demand'] * 0.6
        inputs['phi_i'] = inputs['phi_i'] * 0.6
        return inputs

    def _denormalize_act(self, action: np.ndarray) -> np.ndarray:
        return action * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2

    def _normalize_act(self, action: np.ndarray) -> np.ndarray:
        return (action - ((self.act_high + self.act_low) / 2)) / ((self.act_high - self.act_low) / 2)

    def _denormalize_obs_dict(self, observation: dict) -> dict:
        denorm_observation = {}
        # 反归一化观察值
        for key in observation:
            if isinstance(self.observation_space.spaces[key], gym.spaces.Box):
                low = self.observation_space.spaces[key].low
                high = self.observation_space.spaces[key].high
                # 执行反归一化操作
                denorm_observation[key] = observation[key] * (high - low) / 2 + (high + low) / 2
        return denorm_observation

    def _normalize_obs_dict(self, observation: dict) -> dict:
        norm_observation = {}
        # 对每个 Box 空间进行归一化
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

    def _update_state_RC(self, t: int, action: np.ndarray) -> (dict, dict, dict):
        obs_variables = {}
        current_state = self._rc_state_init(t)
        next_state = self._rc_state_init(t + 1)

        error_L = 1 + get_random_error(t, max_error=0.05)
        error_T = 1 + get_random_error(t, max_error=0.05)

        current_state['t'] = t
        current_state['Ti'], current_state['Te'] = self.Ti_next, self.Te_next

        current_state['switch'], current_state['Tset'], current_state['L'], current_state['Ec_ac'] = self._ac_action(action, current_state['Ti'], current_state['To'])
        L = np.clip(current_state['L'] * error_L, 0, 5000)
        current_state['L'] = L
        self.Ti_next = current_state['Ti'] + self.dt / self.Ci * (
                (current_state['Te'] - current_state['Ti']) / self.Ri + (18 - current_state['Ti']) / self.Rg + (22 - current_state['Ti']) / self.Rn + self.Awindow * current_state['phi_window_s'] +
                self.Ai * (current_state['phi_i'] + L) + self.Av * (current_state['To'] - current_state['Ti']))
        self.Te_next = current_state['Te'] + self.dt / self.Ce * (
                (current_state['Ti'] - current_state['Te']) / self.Ri + (current_state['To'] - current_state['Te']) / self.Ro + self.Awall * current_state['phi_wall_s'])
        next_state['Ti'] = self.Ti_next * error_T
        next_state['Te'] = self.Te_next * error_T


        current_state['Ec_demand'] = current_state['Ec_ac'] + current_state['Ec_other']
        current_state['battery'], current_state['charge'] = self._battery_action(action[2], current_state['Ec_pv'], current_state['Ec_demand'])
        current_state['Ec_true'] = current_state['Ec_demand'] - (current_state['Ec_pv'] - current_state['charge'])

        current_state['cost'] = (self.dt / 3600) * current_state['price'] * current_state['Ec_true'] / 1000 if current_state['Ec_true'] >= 0 else (self.dt / 3600) * 11.5 * current_state['Ec_true'] / 1000
        if current_state['Ec_true'] >= 0:
            current_state['Ec_sell'] = 0
            current_state['Ec_buy'] = current_state['Ec_true']
        else:
            current_state['Ec_sell'] = -current_state['Ec_true']
            current_state['Ec_buy'] = 0
        fea_variables = {'cost': current_state['cost'], 'Ti': next_state['Ti'], 'home': next_state['home'], 'temp_target': next_state['temp_target'], 'Ec_sell': current_state['Ec_sell'],
                         'battery': current_state['battery'], 'Ec_pv': current_state['Ec_pv'], 'Ec_demand': current_state['Ec_demand'],'Ec_buy': current_state['Ec_buy'], 'charge':current_state['charge']}

        for key in self.current_fea:
            next_state[key] = current_state[key]
        if self.obs_space == 'dict':
            for key in self.observation_space.keys():
                obs_variables[key] = next_state[key]
        else:
            obs_list = [next_state['hour'], next_state['workday'], next_state['home'], next_state['battery'], next_state['switch'], next_state['To'], next_state['price'],
                        next_state['Ec_pv'], next_state['Ec_sell'],next_state['Ec_buy'], next_state['Ec_demand'],next_state['Ti'], next_state['temp_target']]
            obs_variables = np.array(obs_list, dtype=np.float64)
        info_variables = next_state
        return obs_variables, info_variables, fea_variables

    def _battery_action(self, charge: np.float32, pv: float, ec_total: float or int) -> (float, float, float):  # 0 <= self.battery_power <= 20
        # TODO pv
        charge = np.clip(charge, -min(float(self.battery_power_init) * 12, self.discharge_capacity), min((self.battery_capacity - self.battery_power_init) * 12, self.charge_capacity))
        charge *= 0.95
        battery_power = np.clip(self.battery_power_init + charge * (self.dt / 3600), 0, self.battery_capacity)
        self.battery_power_init = battery_power
        return battery_power, charge


    def _ac_action(self, action: np.array, Ti: float, To: float) -> (int, int, float, float):
        error_L = 1
        switch_current = 1 if action[0] >= 0 else -1
        if self.switch_previous == -1 and switch_current == 1:
            L, Ec_ac = 3000, 1000
            L = np.clip(L * error_L, 0, 5000)
            delta_T = 0.1851 * np.exp(1.7194 * (L / 2500))
            value_list = [i * 0.5 for i in range(32, 57)]
            Tset = min(value_list, key=lambda x: abs(x - (delta_T + Ti)))
        elif switch_current == -1:
            L, Ec_ac, Tset = 0, 0, 0
        else:
            L = np.clip(action[1], 0, 5000)
            delta_T = 0.1851 * np.exp(1.7194 * (L / 2500))
            value_list = [i * 0.5 for i in range(32, 57)]
            Tset = min(value_list, key=lambda x: abs(x - (delta_T + Ti)))
            L = np.clip(L * error_L, 0, 5000) if L >= 600 else 0
            if self.pid:
                Tset = closest_number(action[1])
                L = self.pid_controller.Load_cal(float(Tset), float(Ti))
                L = np.clip(L * error_L, 0, 5000) if L >= 600 else 0
            Ec_ac = np.clip((-5.3319 * 1e-3 * L - 3.4284) * To + 3.5117 * 1e-5 * L ** 2 + 1.07457 * 1e-1 * L + 96.152, 0, 1500) if L >= 500 else 30
        self.switch_previous = switch_current
        return switch_current, Tset, L, Ec_ac

    def _rc_state_init(self, t: int) -> (dict, dict):
        # print('---------------------------',t,'-----------------------------')
        if t >= len(self.inputs):
            t = t - len(self.inputs)
        To = self.inputs['To'].iloc[t]
        phi_window_s = self.inputs['phi_window_s'].iloc[t]
        phi_wall_s = self.inputs['phi_wall_s'].iloc[t]
        phi_i = self.inputs['phi_i'].iloc[t]
        price = self.inputs['price_'+self.price].iloc[t]
        price = price if price <= 50 else 50
        Ec_other = self.inputs['Ec_demand'].iloc[t]
        Ec_pv = self.inputs['PV'].iloc[t]

        workday = self.inputs['workday'].iloc[t]
        temp_target = self.inputs['temp_target'].iloc[t]
        home = self.inputs['home'].iloc[t]
        dayofyear = self.inputs['dayofyear'].iloc[t]
        hour = self.inputs['hour'].iloc[t]
        mpc_state = dict(t=t, To=To, phi_i=phi_i, phi_wall_s=phi_wall_s, phi_window_s=phi_window_s, Ec_other=Ec_other, Ec_pv=Ec_pv,
                         price=price, temp_target=temp_target, home=home, dayofyear=dayofyear, hour=hour, workday=workday)
        return mpc_state

if __name__ == '__main__':
    from arg import get_config

    config = get_config()
    env1 = BaseEnv(config)
    print(env1.action_space)
