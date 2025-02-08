import os

import numpy as np
from enviroment.base_env import BaseEnv
import random


class CustomEnv(BaseEnv):
    def __init__(self, config: dict, reward_fun=None, setting=None, flag='train'):
        super(CustomEnv, self).__init__(config, flag)
        self.reward = config['rl_config'].reward
        self.reward_fun = reward_fun
        self.episode_rewards, self.ep_rew_mean = [], -np.inf
        # obs_space_len = self.observation_space.shape[0] if self.obs_space == 'box' else len(self.observation_space)
        # self.setting = 'obs{}_act{}_pred{}_'.format(obs_space_len, self.action_space.shape[0], len(self.pred_fea)) + setting
        self.folder_path = './results/' + setting + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.random_init = config['rl_config'].random_init if flag == 'train' else False

    def _init_state(self):
        self.action_init = [self.switch_previous, -0.2, 0]
        self.battery_power_init = 0
        if self.ac_control == 'pid' or self.ac_control == 'Tset':
            self.action_init = [self.switch_previous, 0, 0]

        if self.flag == 'test':
            self.t_init = 0
            self.battery_power_init = 2500
            self.switch_previous = -1
            self.total_timesteps = len(self.inputs)

        elif self.flag == 'eval':
            self.total_timesteps = 7 * 24 * 12
            self.battery_power_init = np.random.randint(0, self.battery_capacity)
            self.t_init = random.randrange(0, len(self.inputs) - self.total_timesteps - 1, 24 * 12)
            self.switch_previous = random.choice([-1, 1])

        else:
            self.total_timesteps = self.timesteps
            self.t_init = random.randrange(0, len(self.inputs) - self.total_timesteps - 1, 24 * 12)
            self.switch_previous = 1

            if self.random_init:
                self.t_init = random.randrange(0, len(self.inputs) - self.timesteps - 2)  # TODO -1
                self.battery_power_init = np.random.randint(0, self.battery_capacity)
                self.switch_previous = random.choice([-1, 1])
                self.action_init = np.random.uniform(low=-1, high=1.0, size=3)
        self.Ec_pv = self.inputs['Ec_pv'].iloc[self.t_init]
        self.Ec_demand = self.inputs['Ec_other'].iloc[self.t_init]

    def reset(self, **kwargs) -> np.ndarray:
        # 检查并移除不兼容的参数
        if 'options' in kwargs:
            del kwargs['options']
        if 'seed' in kwargs:
            SEED = kwargs['seed']
        self.episode_rewards, self.ep_rew_mean = [], -np.inf

        self._init_state()

        self.t = self.t_init + 1
        states_variables, _, _ = self._update_state_RC(self.t_init, self._denormalize_act(self.action_init))
        norm_obs = self._normalize_obs(states_variables) if self.obs_space == 'box' else self._normalize_obs_dict(states_variables)
        return norm_obs, {'obs': norm_obs}

    def reward_function(self, reward_variables: dict) -> float:
        cost = reward_variables['cost']
        battery = reward_variables['battery']
        Ec_pv = reward_variables['Ec_pv']
        Ec_demand = reward_variables['Ec_demand']
        Ec_charge = reward_variables['Ec_charge']
        Ec_sell = reward_variables['Ec_sell']
        Ec_buy = reward_variables['Ec_buy']
        home = reward_variables['home']
        CO2 = reward_variables['CO2']
        ssr = 0

        if self.reward == 'linear':

            T_error = abs(reward_variables['Ti'] - reward_variables['T_target']) * home
            if T_error <= self.T_range:
                RT = self.wT
                if T_error <= 0.5:
                    RT += 0.1 * self.wT
            else:
                RT = -self.wT * min(2 ** T_error, 10)

            # 1
            if battery <= self.battery_capacity * 0.99:
                R_battery = -self.wPV * abs(Ec_pv - (Ec_demand + Ec_charge))
            else:
                R_battery = -self.wPV * abs(Ec_pv - (Ec_demand + Ec_charge + Ec_sell))

            reward = RT - self.wSell * Ec_sell - self.wCO * CO2 - self.wBuy * Ec_buy + self.wSSR * ssr + R_battery

            # 2
            # R1 = self.w1 * min(Ec_pv, Ec_demand) / Ec_demand
            # R2 = self.w2 * min(Ec_pv - Ec_demand, Ec_charge) / self.charge_capacity
            # R3 = self.w3 * (battery - self.battery_capacity) * Ec_sell
            # R4 = (self.w4 * min(-Ec_charge, Ec_demand) / Ec_demand) if Ec_pv < 10 else 0
            # R5 = -self.w5 * Ec_buy
            #
            # switch_current = reward_variables['switch']
            # switch_change_penalty = 0
            # if switch_current == 1 and self.switch_previous == 0:
            #     switch_change_penalty = 1
            # self.w_switch_penalty = 0.5
            # R_switch = -self.w_switch_penalty * switch_change_penalty
            # R_cost = -self.wEc*cost
            # reward = R1 + R2 + R3 + R4 + R5 + RT + R_switch + R_cost
            # print('R1: {}, R2: {}, R3: {}, R4: {} R5:{} RT:{} R_cost:{}'.format(R1, R2, R3, R4,R5,RT, R_cost))

        elif self.reward == 'dl':
            reward = self.reward_fun.get_r(np.array(list(reward_variables.values())))
            reward = reward.squeeze().detach().cpu().numpy()
        else:
            raise 'reward function not defined'
        return reward / self.timesteps

    def step(self, action: np.array) -> (np.ndarray, float, bool, dict):
        denorm_action = self._denormalize_act(action)
        obs_variables, state_variables,reward_variables = self._update_state_RC(self.t, denorm_action)
        self.t += 1
        done = True if self.t == self.t_init + self.total_timesteps - 1 else False
        truncated = True if self.t == self.t_init + self.total_timesteps - 1 else False  # 在没有截断逻辑时，设置为 False
        norm_obs = self._normalize_obs(obs_variables) if self.obs_space == 'box' else self._normalize_obs_dict(obs_variables)
        reward = self.reward_function(reward_variables)
        self.episode_rewards.append(reward)
        self.ep_rew_mean = np.array(self.episode_rewards).mean() if len(self.episode_rewards) > 0 else np.nan
        info = dict(obs=norm_obs, rews=reward, output=state_variables)
        return norm_obs, reward, done, truncated, info


if __name__ == '__main__':
    value_len = [0.5 * i for i in range(32, 57)]
