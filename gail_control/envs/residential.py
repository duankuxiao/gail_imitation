"""Gymnasium environment used by the RL and imitation algorithms."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from gail_control.envs.base import BaseEnv


class CustomEnv(BaseEnv):
    """Residential AC and battery control environment."""

    def __init__(self, config: dict, reward_fun=None, setting: str = "", flag: str = "train"):
        super().__init__(config, flag)
        rl_config = config["rl_config"]

        self.reward = rl_config.reward
        self.reward_fun = reward_fun
        self.episode_rewards = []
        self.ep_rew_mean = np.nan

        obs_space_len = self.observation_space.shape[0] if self.obs_space == "box" else len(self.observation_space)
        setting_suffix = setting or "default"
        self.setting = f"obs{obs_space_len}_act{self.action_space.shape[0]}_{setting_suffix}"
        self.folder_path = str(Path(rl_config.results_dir) / self.setting)
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)

        self.random_init = rl_config.random_init if flag == "train" else False

    def _init_state(self) -> None:
        self.action_init = [self.switch_previous, -0.2, 0]
        self.battery_power_init = 0
        if self.ac_control in {"pid", "Tset"}:
            self.action_init = [self.switch_previous, 0, 0]

        if self.flag == "test":
            self.t_init = 0
            self.switch_previous = -1
            self.total_timesteps = len(self.inputs)
        elif self.flag == "eval":
            self.total_timesteps = 7 * 24 * 12
            max_start = max(1, len(self.inputs) - self.total_timesteps - 1)
            self.t_init = random.randrange(0, max_start, 24 * 12) if max_start > 24 * 12 else 0
            self.switch_previous = random.choice([-1, 1])
        else:
            self.total_timesteps = self.timesteps
            max_start = max(1, len(self.inputs) - self.total_timesteps - 1)
            self.t_init = random.randrange(0, max_start, 24 * 12) if max_start > 24 * 12 else 0
            self.switch_previous = 1

            if self.random_init:
                self.t_init = random.randrange(0, max_start)
                self.battery_power_init = np.random.randint(0, self.battery_capacity)
                self.switch_previous = random.choice([-1, 1])
                self.action_init = np.random.uniform(low=-1, high=1.0, size=3)

        self.Ec_pv = self.inputs["Ec_pv"].iloc[self.t_init]
        self.Ec_demand = self.inputs["Ec_other"].iloc[self.t_init]
        self.Ec_price = self.inputs[f"price_{self.price}"].iloc[self.t_init]
        self.Ti_next = self.inputs["Ti"].iloc[self.t_init]
        if "Te" in self.inputs.columns:
            self.Te_next = self.inputs["Te"].iloc[self.t_init]
        else:
            outdoor_temperature = self.inputs["To"].iloc[self.t_init]
            self.Te_next = (self.Ri * outdoor_temperature + self.Ro * self.Ti_next) / (self.Ri + self.Ro)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        seed = kwargs.pop("seed", None)
        kwargs.pop("options", None)
        if seed is not None:
            self.seed(seed)

        self.episode_rewards = []
        self.ep_rew_mean = np.nan
        self._init_state()

        self.t = self.t_init + 1
        states_variables, _, _ = self._update_state_RC(self.t_init, self._denormalize_act(self.action_init))
        norm_obs = self._normalize_obs(states_variables) if self.obs_space == "box" else self._normalize_obs_dict(states_variables)
        return norm_obs, {"obs": norm_obs}

    def reward_function(self, reward_variables: dict) -> float:
        if self.reward == "dl":
            if self.reward_fun is None:
                raise ValueError("reward_fun is required when reward='dl'")
            reward = self.reward_fun.get_r(np.array(list(reward_variables.values())))
            return float(reward.squeeze().detach().cpu().numpy()) / self.timesteps

        if self.reward != "linear":
            raise ValueError(f"Undefined reward function: {self.reward}")

        cost = reward_variables["cost"]
        battery = reward_variables["battery"]
        ec_pv = reward_variables["Ec_pv"]
        ec_demand = reward_variables["Ec_demand"]
        ec_charge = reward_variables["Ec_charge"]
        ec_sell = reward_variables["Ec_sell"]
        ec_buy = reward_variables["Ec_buy"]
        home = reward_variables["home"]
        co2 = reward_variables["CO2"]

        temperature_error = abs(reward_variables["Ti"] - reward_variables["T_target"])
        if temperature_error <= self.T_range:
            temperature_reward = self.wT * home
        else:
            temperature_reward = -self.wT * (temperature_error ** 2) * home

        if ec_pv > 0:
            pv_reward = -self.wPV * abs(ec_pv - (ec_demand + ec_charge + ec_sell))
        else:
            pv_reward = -self.wPV * abs(ec_demand + ec_charge)

        reward = (
            temperature_reward
            - self.wEc * cost
            - self.wSell * ec_sell
            - self.wCO * co2
            - self.wBuy * ec_buy
            + pv_reward
        )
        return float(reward) / self.timesteps

    def step(self, action: np.ndarray):
        denorm_action = self._denormalize_act(action)
        obs_variables, info_variables, reward_variables = self._update_state_RC(self.t, denorm_action)
        reward = self.reward_function(reward_variables)

        self.t += 1
        episode_end = self.t == self.t_init + self.total_timesteps - 1
        norm_obs = self._normalize_obs(obs_variables) if self.obs_space == "box" else self._normalize_obs_dict(obs_variables)

        self.episode_rewards.append(reward)
        self.ep_rew_mean = float(np.mean(self.episode_rewards)) if self.episode_rewards else np.nan
        info = {"obs": norm_obs, "rews": reward, "output": info_variables}
        return norm_obs, reward, episode_end, episode_end, info


if __name__ == "__main__":
    from gail_control.config import get_config

    config = get_config([])
    env = CustomEnv(config, setting="smoke")
    observation, _ = env.reset()
    print(env.observation_space)
    print(observation)
