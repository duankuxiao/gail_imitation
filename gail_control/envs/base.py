"""Base environment logic for the residential AC and battery simulator."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from gail_control.utils.rl_tools import closest_number
from gail_control.utils.tools import get_random_error


class PIDControl:
    """Simple PID controller used by the baseline and optional AC control."""

    def __init__(self, max_value: float = 5000, min_value: float = 0, Kp: float = 520, Ki: float = 165, Kd: float = 60):
        self.max = max_value
        self.min = min_value
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.pre_error = 0.0

    def calculate_load(self, set_point: float, indoor_temperature: float) -> int:
        error = set_point - indoor_temperature
        proportional = self.Kp * error
        self.integral += error
        integral = self.Ki * self.integral
        derivative = self.Kd * (error - self.pre_error)
        output = proportional + integral + derivative

        if output > self.max:
            output = self.max
        elif output < self.min:
            output = 0

        self.pre_error = error
        return round(output)

    def Load_cal(self, setPoint: float, Ti: float) -> int:
        """Backward-compatible alias for the original method name."""
        return self.calculate_load(setPoint, Ti)


PID_control = PIDControl


class BaseEnv(gym.Env):
    """Shared simulator mechanics for train/eval/test environments."""

    def __init__(self, config: dict, flag: str):
        self.flag = flag
        rl_config = config["rl_config"]

        self._set_up(rl_config)
        self.inputs = self._read_inputs(flag)
        self._rc_parameters(config["rc_config"])

        self.t = 1
        self._reward_parameter(rl_config)
        self._init_action_space()
        self._init_observation_space()

    def _set_up(self, rl_config) -> None:
        self.train_data_path = rl_config.train_data_path
        self.test_data_path = rl_config.test_data_path
        self.use_pv_forecast = rl_config.use_pv_forecast
        self.use_next_state = rl_config.use_next_state
        self.use_To_forecast = rl_config.use_To_forecast
        self.ac_control = rl_config.ac_control
        self.obs_space = rl_config.obs
        self.price = rl_config.price
        self.Ti_init = rl_config.Ti_init
        self.timesteps = rl_config.timesteps
        self.total_timesteps = rl_config.timesteps
        self.T_range = rl_config.T_range
        self.T_delta = rl_config.T_delta
        self.baseline_sr_reference = getattr(rl_config, "baseline_sr_reference", None)
        self.baseline_cost_reference = getattr(rl_config, "baseline_cost_reference", None)
        self.battery_power_init = 0
        self.switch_previous = 1
        self.action_init = [1, 0.2, 0]
        if self.ac_control == "pid":
            self.pid_controller = PIDControl()

    def _reward_parameter(self, rl_config) -> None:
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

    def _rc_parameters(self, rc_config) -> None:
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

    def _init_action_space(self) -> None:
        self.act_low = np.array([-1, 0, -self.discharge_capacity], dtype=np.float64)
        self.act_high = np.array([1, 5000, self.charge_capacity], dtype=np.float64)
        if self.ac_control in {"pid", "Tset"}:
            self.act_low = np.array([-1, 16, -self.discharge_capacity], dtype=np.float64)
            self.act_high = np.array([1, 28, self.charge_capacity], dtype=np.float64)

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def _init_observation_space(self) -> None:
        observation_space = spaces.Dict(
            {
                "hour": spaces.Box(low=0, high=23, dtype=np.float64),
                "Ti": spaces.Box(low=10, high=40, dtype=np.float64),
                "To": spaces.Box(low=-10, high=50, dtype=np.float64),
                "T_target": spaces.Box(low=16, high=28, dtype=np.float64),
                "home": spaces.Box(low=0, high=1, dtype=np.float64),
                "battery": spaces.Box(low=0, high=self.battery_capacity, dtype=np.float64),
                "Ec_pv": spaces.Box(low=0, high=3900, dtype=np.float64),
                "Ec_demand": spaces.Box(low=0, high=6000, dtype=np.float64),
                "price": spaces.Box(low=0, high=50, dtype=np.float64),
            }
        )
        self.current_observation_space = observation_space

        if self.use_next_state:
            self.next_observation_space = spaces.Dict(
                {
                    "To_next": spaces.Box(low=-10, high=50, dtype=np.float64),
                    "Ec_pv_next": spaces.Box(low=0, high=3900, dtype=np.float64),
                }
            )
            observation_space = spaces.Dict({**observation_space.spaces, **self.next_observation_space.spaces})

        if self.use_pv_forecast:
            self.pv_forecast_observation_space = spaces.Dict(
                {
                    f"PV_future_{step}": spaces.Box(low=0, high=3900, dtype=np.float64)
                    for step in range(1, 13)
                }
            )
            observation_space = spaces.Dict({**observation_space.spaces, **self.pv_forecast_observation_space.spaces})

        if self.use_To_forecast:
            self.To_forecast_observation_space = spaces.Dict(
                {
                    f"To_future_{step}": spaces.Box(low=-10, high=50, dtype=np.float64)
                    for step in range(1, 13)
                }
            )
            observation_space = spaces.Dict({**observation_space.spaces, **self.To_forecast_observation_space.spaces})

        self.raw_observation_space = observation_space

        if self.obs_space == "box":
            obs_low = []
            obs_high = []
            for space in observation_space.spaces.values():
                if isinstance(space, spaces.Box):
                    obs_low.append(float(np.asarray(space.low).reshape(-1)[0]))
                    obs_high.append(float(np.asarray(space.high).reshape(-1)[0]))

            self.obs_low = np.array(obs_low, dtype=np.float64)
            self.obs_high = np.array(obs_high, dtype=np.float64)
            observation_space = spaces.Box(
                low=-np.ones_like(self.obs_low),
                high=np.ones_like(self.obs_high),
                dtype=np.float64,
            )
        else:
            observation_space = spaces.Dict(
                {
                    key: spaces.Box(
                        low=-np.ones_like(space.low, dtype=np.float64),
                        high=np.ones_like(space.high, dtype=np.float64),
                        dtype=np.float64,
                    )
                    for key, space in observation_space.spaces.items()
                }
            )

        self.observation_space = observation_space

    def test(self, flag: str) -> dict:
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int = 9743) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _read_inputs(self, flag: str) -> pd.DataFrame:
        path = self.train_data_path if flag == "train" else self.test_data_path
        inputs = pd.read_csv(path, index_col=0).iloc[:46081, :]
        inputs.index = pd.to_datetime(inputs.index)
        inputs["Ec_pv"] = inputs["PV"] * 1.4
        inputs["Ec_other"] = inputs["Ec_demand"] * 0.5
        inputs["phi_i"] = inputs["phi_i"] + 0.5 * inputs["phi_i"] * (inputs["home"] == 1)
        return inputs

    def _denormalize_act(self, action: np.ndarray) -> np.ndarray:
        return action * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2

    def _normalize_act(self, action: np.ndarray) -> np.ndarray:
        return (action - ((self.act_high + self.act_low) / 2)) / ((self.act_high - self.act_low) / 2)

    def _denormalize_obs_dict(self, observation: dict) -> dict:
        denorm_observation = {}
        for key, value in observation.items():
            if isinstance(self.raw_observation_space.spaces[key], gym.spaces.Box):
                low = self.raw_observation_space.spaces[key].low
                high = self.raw_observation_space.spaces[key].high
                denorm_observation[key] = value * (high - low) / 2 + (high + low) / 2
        return denorm_observation

    def _normalize_obs_dict(self, observation: dict) -> dict:
        norm_observation = {}
        for key, value in observation.items():
            if isinstance(self.raw_observation_space.spaces[key], gym.spaces.Box):
                low = self.raw_observation_space.spaces[key].low
                high = self.raw_observation_space.spaces[key].high
                norm_observation[key] = (value - ((high + low) / 2)) / ((high - low) / 2)
        return norm_observation

    def _denormalize_obs(self, observation: np.ndarray) -> np.ndarray:
        return observation * ((self.obs_high - self.obs_low) / 2) + ((self.obs_high + self.obs_low) / 2)

    def _normalize_obs(self, observation: np.ndarray) -> np.ndarray:
        return (observation - ((self.obs_high + self.obs_low) / 2)) / ((self.obs_high - self.obs_low) / 2)

    def _load_from_RC(self, Tset: float, current_state: dict) -> float:
        load = (
            (
                (Tset - current_state["Ti"]) * self.Ci / self.dt
                - (
                    self.Av * (current_state["To"] - current_state["Ti"])
                    + (current_state["Te"] - current_state["Ti"]) / self.Ri
                    + (18 - current_state["Ti"]) / self.Rg
                    + (22 - current_state["Ti"]) / self.Rn
                    + self.Awindow * current_state["phi_window_s"]
                )
            )
            / self.Ai
            - current_state["phi_i"]
        )
        return load

    def _RC(self, current_state: dict) -> Tuple[float, float]:
        Ti_next = current_state["Ti"] + self.dt / self.Ci * (
            (current_state["Te"] - current_state["Ti"]) / self.Ri
            + (18 - current_state["Ti"]) / self.Rg
            + (22 - current_state["Ti"]) / self.Rn
            + self.Awindow * current_state["phi_window_s"]
            + self.Ai * (current_state["phi_i"] + current_state["L"])
            + self.Av * (current_state["To"] - current_state["Ti"])
        )
        Te_next = current_state["Te"] + self.dt / self.Ce * (
            (current_state["Ti"] - current_state["Te"]) / self.Ri
            + (current_state["To"] - current_state["Te"]) / self.Ro
            + self.Awall * current_state["phi_wall_s"]
        )
        return Ti_next, Te_next

    def _future_value(self, column: str, t: int, step: int, default: float = 0.0) -> float:
        future_idx = t + step
        if future_idx >= len(self.inputs):
            return default
        return self.inputs[column].iloc[future_idx]

    def _build_observation(self, t: int, current_state: dict, next_state: dict):
        if self.obs_space == "dict":
            obs_variables = {key: current_state[key] for key in self.current_observation_space}
            if self.use_next_state:
                for key in self.next_observation_space:
                    obs_variables[key] = next_state[key[:-5]]
            if self.use_pv_forecast:
                for key in self.pv_forecast_observation_space:
                    step = int(key.split("_")[-1])
                    obs_variables[key] = self._future_value("Ec_pv", t, step)
            if self.use_To_forecast:
                for key in self.To_forecast_observation_space:
                    step = int(key.split("_")[-1])
                    obs_variables[key] = self._future_value("To", t, step)
            return obs_variables

        obs_list = [current_state[key] for key in self.current_observation_space]
        if self.use_next_state:
            for key in self.next_observation_space:
                obs_list.append(next_state[key[:-5]])
        if self.use_pv_forecast:
            obs_list.extend(self._future_value("Ec_pv", t, step) for step in range(1, 13))
        if self.use_To_forecast:
            obs_list.extend(self._future_value("To", t, step) for step in range(1, 13))
        return np.array(obs_list, dtype=np.float64)

    def _update_state_RC(self, t: int, action: np.ndarray) -> Tuple[dict, dict, dict]:
        current_state = self._rc_state_init(t)
        next_state = self._rc_state_init(t + 1)

        error_L = 1 + get_random_error(t, max_error=0.03)
        error_T = 1 + get_random_error(t, max_error=0.03)

        current_state["t"] = t
        current_state["Ti"], current_state["Te"] = self.Ti_next, self.Te_next
        (
            current_state["switch"],
            current_state["Tset"],
            current_state["L"],
            current_state["Ec_ac"],
        ) = self._ac_action(action[0], action[1], current_state["Ti"], current_state["To"], current_state)
        current_state["L"] = np.clip(current_state["L"] * error_L, 0, 5200)

        current_state["Ec_demand"] = current_state["Ec_ac"] + current_state["Ec_other"]
        current_state["battery"], current_state["Ec_charge"] = self._battery_action(
            action[2],
            current_state["Ec_pv"],
            current_state["Ec_demand"],
            flag="rl",
        )
        current_state["Ec_true"] = current_state["Ec_demand"] - (
            current_state["Ec_pv"] - current_state["Ec_charge"]
        )

        if current_state["Ec_true"] >= 0:
            current_state["Ec_sell"] = 0
            current_state["Ec_buy"] = current_state["Ec_true"]
            current_state["cost"] = (self.dt / 3600) * current_state["price"] * current_state["Ec_true"] / 1000
        else:
            current_state["Ec_sell"] = -current_state["Ec_true"]
            current_state["Ec_buy"] = 0
            current_state["cost"] = (self.dt / 3600) * 10 * current_state["Ec_true"] / 1000

        self.Ec_demand = current_state["Ec_demand"]
        self.Ec_pv = current_state["Ec_pv"]
        self.Ec_price = current_state["price"]

        current_state["CO2"] = (
            0.000457 * 1000000 * current_state["Ec_buy"] / 1000
            + 38 * current_state["Ec_pv"] / 1000
        )

        ti_next, te_next = self._RC(current_state)
        self.Ti_next = ti_next * error_T
        self.Te_next = te_next * error_T
        next_state["Ti"] = ti_next
        next_state["Te"] = te_next

        obs_variables = self._build_observation(t, current_state, next_state)
        reward_states = {
            "cost": current_state["cost"],
            "battery": current_state["battery"],
            "Ec_pv": current_state["Ec_pv"],
            "Ec_demand": current_state["Ec_demand"],
            "Ec_charge": current_state["Ec_charge"],
            "Ec_sell": current_state["Ec_sell"],
            "Ec_buy": current_state["Ec_buy"],
            "CO2": current_state["CO2"],
            "home": next_state["home"],
            "Ti": self.Ti_next,
            "T_target": next_state["T_target"],
            "switch": current_state["switch"],
        }
        return obs_variables, current_state, reward_states

    def _rl_battery_bounds(self, Ec_pv: float, Ec_total: float) -> Tuple[float, float]:
        """Constrain RL battery actions to self-consumption behavior.

        The paper's battery logic prioritizes PV self-use: charge only from
        surplus PV and discharge only to cover local net demand. This prevents
        the agent from learning lossy grid arbitrage such as selling stored
        energy at the low feed-in tariff and buying it back later.
        """
        available_discharge = min(float(self.battery_power_init) * 12, self.discharge_capacity)
        available_charge = min((self.battery_capacity - self.battery_power_init) * 12, self.charge_capacity)
        local_deficit = max(Ec_total - Ec_pv, 0)
        pv_surplus = max(Ec_pv - Ec_total, 0)
        lower_bound = -min(available_discharge, local_deficit)
        upper_bound = min(available_charge, pv_surplus)
        return lower_bound, upper_bound

    def _battery_action(
        self,
        Ec_charge: np.float32 = 0.0,
        Ec_pv: float = 0.0,
        Ec_total: float = 0.0,
        flag: str = "rl",
    ) -> Tuple[float, float]:
        if flag == "rl":
            lower_bound, upper_bound = self._rl_battery_bounds(Ec_pv, Ec_total)
            Ec_charge = np.clip(Ec_charge, lower_bound, upper_bound)
        elif flag == "none":
            if Ec_pv > 0:
                Ec_charge = np.clip(
                    Ec_pv - Ec_total,
                    0,
                    min((self.battery_capacity - self.battery_power_init) * 12, Ec_pv, self.charge_capacity),
                )
            else:
                Ec_charge = -min(float(self.battery_power_init) * 12, Ec_total, self.discharge_capacity)
        elif flag == "base":
            if Ec_pv > 0:
                if self.battery_power_init == self.battery_capacity:
                    Ec_charge = -min(Ec_total - Ec_pv, self.discharge_capacity)
                else:
                    Ec_charge = np.clip(
                        Ec_pv - Ec_total,
                        0,
                        min((self.battery_capacity - self.battery_power_init) * 12, Ec_pv, self.charge_capacity),
                    )
            else:
                Ec_charge = -min(float(self.battery_power_init) * 12, Ec_total, self.discharge_capacity)
        else:
            raise ValueError(f"Unknown battery action flag: {flag}")

        if Ec_charge > 0:
            Ec_charge *= 0.97
            battery_power = np.clip(self.battery_power_init + Ec_charge * (self.dt / 3600), 0, self.battery_capacity)
        else:
            battery_power = np.clip(self.battery_power_init + Ec_charge * (self.dt / 3600), 0, self.battery_capacity)
            Ec_charge *= 0.97

        self.battery_power_init = round(battery_power)
        return round(battery_power), round(Ec_charge)

    def _ac_action(
        self,
        switch: int = -1,
        action: float = 0.0,
        Ti: float = 22,
        To: float = 16.0,
        current_state: dict = None,
    ) -> Tuple[int, float, float, float]:
        switch_current = 1 if switch > 0 else -1
        if self.switch_previous == -1 and switch_current == 1:
            load, Ec_ac = 3000, 1000
            delta_T = 0.1851 * np.exp(1.7194 * (load / 2500))
            value_list = [i * 0.5 for i in range(32, 57)]
            Tset = min(value_list, key=lambda x: abs(x - (delta_T + Ti)))
        elif switch_current == -1:
            load, Ec_ac, Tset = 0, 0, 0
        else:
            load = action
            delta_T = 0.1851 * np.exp(1.7194 * (load / 2500))
            value_list = [i * 0.5 for i in range(32, 57)]
            Tset = min(value_list, key=lambda x: abs(x - (delta_T + Ti)))
            load = np.clip(load, 0, 5000) if load >= 600 else 0

            if self.ac_control == "pid":
                Tset = closest_number(action)
                load = self.pid_controller.Load_cal(float(Tset), float(Ti))
            elif self.ac_control == "Tset":
                Tset = closest_number(action)
                load = self._load_from_RC(Tset, current_state)

            load = np.clip(load, 0, 5200) if load >= 700 else 0
            load_for_energy = np.clip(self.w_ec_ac * load, 0, 5200) if load >= 700 else 0
            Ec_ac = (
                np.clip(
                    self.w_ec_ac
                    * (
                        (-5.3319e-3 * load_for_energy - 3.4284) * To
                        + 3.5117e-5 * load_for_energy**2
                        + 1.07457e-1 * load_for_energy
                        + 96.152
                    ),
                    0,
                    1500,
                )
                if load >= 700
                else 30
            )

        self.switch_previous = switch_current
        return switch_current, Tset, load, Ec_ac

    def _rc_state_init(self, t: int) -> dict:
        if t >= len(self.inputs):
            t = t - len(self.inputs)

        price = self.inputs[f"price_{self.price}"].iloc[t]
        price = price if price <= 50 else 50
        return {
            "t": t,
            "To": self.inputs["To"].iloc[t],
            "phi_i": self.inputs["phi_i"].iloc[t],
            "phi_wall_s": self.inputs["phi_wall_s"].iloc[t],
            "phi_window_s": self.inputs["phi_window_s"].iloc[t],
            "Ec_other": self.inputs["Ec_other"].iloc[t],
            "Ec_pv": self.inputs["Ec_pv"].iloc[t],
            "Ti": self.inputs["Ti"].iloc[t],
            "price": price,
            "T_target": self.inputs["temp_target"].iloc[t],
            "home": self.inputs["home"].iloc[t],
            "dayofyear": self.inputs["dayofyear"].iloc[t],
            "hour": self.inputs["hour"].iloc[t],
            "workday": self.inputs["workday"].iloc[t],
        }

    def evaluation(self, res, output: bool = False, plot: bool = False):
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)
        res = res.copy()
        res["L"] = res["L"] * 0.8

        sr_reference = self.baseline_sr_reference
        cost_reference = self.baseline_cost_reference

        if res.empty:
            mape = np.nan
            mse = np.nan
        else:
            mape = mean_absolute_percentage_error(res["T_target"], res["Ti"])
            mse = mean_squared_error(res["T_target"], res["Ti"])

        pv_sum = res["Ec_pv"].sum()
        rate = round((1 - res["Ec_sell"].sum() / pv_sum) * 100, 2) if pv_sum else np.nan
        Ec_demand = res["Ec_demand"].sum() / 12 / 1000
        Ec_true = res["Ec_true"].sum() / 12 / 1000
        Ec_max = res["Ec_demand"].max()
        CO2 = round((res["CO2"].sum() / 12) / 1000)
        cost = res["cost"].sum()
        count_within_range = ((res["Ti"] - res["T_target"]).abs() <= self.T_range).sum()
        sr = count_within_range / len(res) if len(res) > 0 else 0

        sr_baseline = sr if sr_reference is None else sr_reference
        cost_baseline = cost if cost_reference is None else cost_reference
        sr_eval = (sr - sr_baseline) / sr_baseline * 100 if sr_baseline else np.nan
        cost_eval = (cost_baseline - cost) / cost_baseline * 100 if cost_baseline else np.nan
        metrics = {
            "mape": mape,
            "mse": mse,
            "sr": sr,
            "Ec_demand": Ec_demand,
            "Ec_true": Ec_true,
            "Ec_max": Ec_max,
            "cost": cost,
            "sr_improve": sr_eval,
            "cost_improve": cost_eval,
            "rate": rate,
            "CO2": CO2,
        }

        if output:
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
            result_name = (
                f"sr_eval{round(sr_eval, 2)}_cost_eval{round(cost_eval, 2)}_"
                f"mape{round(mape * 100, 2)}_sr{round(sr * 100, 2)}_"
                f"Ecd{round(Ec_demand, 2)}_cost{round(cost, 2)}_rate{rate}_CO{CO2}.csv"
            )
            res.to_csv(os.path.join(self.folder_path, result_name))
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).T
            metrics_name = (
                f"metrics_mape{round(mape * 100, 2)}_sr{round(sr * 100, 2)}_"
                f"Ecd{round(Ec_demand, 2)}_cost{round(cost, 2)}_rate{rate}_CO{CO2}.csv"
            )
            metrics_df.to_csv(os.path.join(self.folder_path, metrics_name))

        print(
            "\033[94m"
            + (
                f"sr_eval{round(sr_eval, 2)}_cost_eval{round(cost_eval, 2)}_"
                f"mape{round(mape * 100, 2)}_sr{round(sr * 100, 2)}_"
                f"Ecd{round(Ec_demand, 2)}_cost{round(cost, 2)}_rate{rate}_CO{CO2}kg.csv"
            )
            + "\033[0m"
        )
        if plot:
            self._figure(res)
        return res, metrics

    def _figure(self, res: pd.DataFrame) -> None:
        window = res.iloc[15553 : 15553 + 24 * 7 * 12, :]
        plt.figure(1, figsize=(20, 10))
        plt.plot(window["T_target"], color="deepskyblue", ms=5, label="Target")
        plt.plot(window["Ti"], color="darkorange", marker="o", ms=5, label="Ti")
        plt.title("Simulation Result", fontsize=20)
        plt.xlabel("Time [5 min]", fontsize=20)
        plt.ylabel("Temperature [C]", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)

        for index, col in enumerate(["L"], start=4):
            plt.figure(index, figsize=(20, 10))
            plt.plot(window[col], color="deepskyblue", ms=2, label=col)
            plt.xlabel("Time [5 min]", fontsize=20)
            plt.ylabel(col, fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
        plt.show()
