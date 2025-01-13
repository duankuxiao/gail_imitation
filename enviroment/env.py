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
        self.folder_path = './results/baseline/' + setting + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.random_init = config['rl_config'].random_init if flag == 'train' else False

    def _init_state(self):
        self.action_init = [self.switch_previous, -0.2, 0]
        self.battery_power_init = 0
        if self.pid:
            self.action_init = [self.switch_previous, 0, 0]

        if self.flag == 'test':
            self.t_init = 0
            self.switch_previous = -1
            self.total_timesteps = len(self.inputs)

        elif self.flag == 'eval':
            self.total_timesteps = 7 * 24 * 12
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
        PV = reward_variables['Ec_pv']
        Ec_demand = reward_variables['Ec_demand']
        charge = reward_variables['charge']
        Ec_sell = reward_variables['Ec_sell']
        Ec_buy = reward_variables['Ec_buy']
        home = reward_variables['home']
        CO2 = reward_variables['CO2']
        if self.reward == 'linear':
            if self.T_delta:
                reward = self.wT * reward_variables['T_delta'] if reward_variables['T_delta'] >= 1 else self.c - self.wEc * cost
            else:
                T_delta = abs(reward_variables['Ti'] - reward_variables['temp_target'])
                reward = (self.c * home) if T_delta < self.T_range else (- self.wT * T_delta * home)

            reward += -self.wEc * cost
            reward += -self.wS * Ec_sell  # TODO

            if battery <= self.battery_capacity * 0.99:
                reward += -self.wPV * abs(PV - (Ec_demand + charge))
            else:
                reward += -self.wPV * abs(PV - (Ec_demand + charge + Ec_sell))
            reward += 0.0001 * CO2
        elif self.reward == 'dl':
            reward = self.reward_fun.get_r(np.array(list(reward_variables.values())))
            reward = reward.squeeze().detach().cpu().numpy()
        else:
            raise 'reward function not defined'
        return reward / self.timesteps

    def step(self, action: np.array) -> (np.ndarray, float, bool, dict):
        denorm_action = self._denormalize_act(action)
        # print('input t : {}'.format(self.t))
        obs_variables, info_variables,fea_variables = self._update_state_RC(self.t, denorm_action)
        self.t += 1
        # if None in obs_variables:
        # print('input t : {}  obs_variables : {}'.format(self.t, obs_variables))

        done = True if self.t == self.t_init + self.total_timesteps - 1 else False
        truncated = True if self.t == self.t_init + self.total_timesteps - 1 else False  # 在没有截断逻辑时，设置为 False
        norm_obs = self._normalize_obs(obs_variables) if self.obs_space == 'box' else self._normalize_obs_dict(obs_variables)
        reward = self.reward_function(fea_variables)
        self.episode_rewards.append(reward)
        self.ep_rew_mean = np.array(self.episode_rewards).mean() if len(self.episode_rewards) > 0 else np.nan
        info = dict(obs=norm_obs, rews=reward, output=info_variables)
        return norm_obs, reward, done, truncated, info


if __name__ == '__main__':
    value_len = [0.5 * i for i in range(32, 57)]
