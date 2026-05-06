"""Load MPC/expert demonstrations for GASAC/GAIL training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from imitation.data import types


def _round_half_degree(value: float) -> float:
    return float(np.clip(round(value * 2) / 2, 16, 28))


def _first_existing(row: pd.Series, names: Iterable[str]):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"None of these columns exist in expert data: {list(names)}")


def _env_input_value(env, column: str, row_index: int):
    if hasattr(env, "inputs") and column in env.inputs.columns and row_index < len(env.inputs):
        return env.inputs[column].iloc[row_index]
    return None


def _future_env_input_value(env, column: str, row_index: int, step: int, default: float = 0.0) -> float:
    value = _env_input_value(env, column, row_index + step)
    return default if value is None else value


def _price_value(row: pd.Series, env, row_index: int) -> float:
    price_column = f"price_{env.price}"
    env_price = _env_input_value(env, price_column, row_index)
    if env_price is not None:
        return float(min(env_price, 50))
    return float(_first_existing(row, [price_column, "price"]))


def _state_value(key: str, row: pd.Series, env, row_index: int) -> float:
    if key == "T_target":
        env_target = _env_input_value(env, "temp_target", row_index)
        return float(env_target if env_target is not None else _first_existing(row, ["T_target", "temp_target", "target_temp_normal"]))
    if key == "Ec_pv":
        env_pv = _env_input_value(env, "Ec_pv", row_index)
        return float(env_pv if env_pv is not None else _first_existing(row, ["Ec_pv", "PV"]))
    if key == "Ec_demand":
        return float(_first_existing(row, ["Ec_total", "Power", "Ec_demand"]))
    if key == "price":
        return _price_value(row, env, row_index)
    if key in {"To", "hour", "home", "workday"}:
        env_value = _env_input_value(env, key, row_index)
        return float(env_value if env_value is not None else row[key])
    if key.startswith("PV_future_"):
        step = int(key.split("_")[-1])
        return float(_future_env_input_value(env, "Ec_pv", row_index, step))
    if key.startswith("To_future_"):
        step = int(key.split("_")[-1])
        return float(_future_env_input_value(env, "To", row_index, step))
    if key.endswith("_next"):
        base_key = key[:-5]
        return _state_value(base_key, row, env, row_index + 1)
    return float(row[key])


def _observation_keys(env) -> List[str]:
    if hasattr(env, "raw_observation_space"):
        return list(env.raw_observation_space.spaces.keys())
    return list(env.current_observation_space)


def _state_row(row: pd.Series, env, row_index: int) -> np.ndarray:
    values = []
    for key in _observation_keys(env):
        values.append(_state_value(key, row, env, row_index))
    return np.array(values, dtype=np.float64)


def _infer_setpoint_from_load(row: pd.Series, env) -> float:
    if "Tset" in row:
        return _round_half_degree(row["Tset"])
    if "L" not in row:
        return _round_half_degree(_first_existing(row, ["T_target", "temp_target", "target_temp_normal"]))

    load = row["L"]
    if load <= 0:
        return _round_half_degree(_first_existing(row, ["T_target", "temp_target", "target_temp_normal"]))

    ti = row["Ti"]
    te = row["Te"] if "Te" in row else ti
    to = row["To"]
    phi_i = row["phi_i"]
    phi_window_s = row["phi_window_s"]
    thermal_terms = (
        env.Av * (to - ti)
        + (te - ti) / env.Ri
        + (18 - ti) / env.Rg
        + (22 - ti) / env.Rn
        + env.Awindow * phi_window_s
    )
    setpoint = (((load + phi_i) * env.Ai + thermal_terms) * env.dt / env.Ci) + ti
    return _round_half_degree(setpoint)


def _action_row(row: pd.Series, env) -> np.ndarray:
    switch = 1.0 if row["switch"] > 0 else -1.0
    setpoint = _infer_setpoint_from_load(row, env)

    if "Ec_charge" in row:
        charge_rate = row["Ec_charge"]
    elif "charge" in row and "discharge" in row:
        charge_rate = row["charge"] - row["discharge"]
    else:
        charge_rate = _first_existing(row, ["C", "charge"])

    physical_action = np.array([switch, setpoint, charge_rate], dtype=np.float64)
    physical_action = np.clip(physical_action, env.act_low, env.act_high)
    return env._normalize_act(physical_action).astype(np.float32)


def heuristic_action(env, obs: np.ndarray) -> np.ndarray:
    """Return a normalized action from an environment-consistent expert.

    This controller is deliberately simple and uses only the current
    observation variables: a one-step RC comfort target for AC and PV-first
    self-consumption for the battery.
    """
    state = dict(zip(_observation_keys(env), env._denormalize_obs(obs)))
    target = float(state["T_target"])
    indoor_temperature = float(state["Ti"])
    setpoint = target + 0.25 * np.clip(target - indoor_temperature, -0.5, 0.8)
    setpoint = _round_half_degree(setpoint)

    pv = float(state["Ec_pv"])
    demand = float(state["Ec_demand"])
    charge_rate = np.clip(pv - demand, -env.discharge_capacity, env.charge_capacity)

    physical_action = np.array([1.0, setpoint, charge_rate], dtype=np.float64)
    physical_action = np.clip(physical_action, env.act_low, env.act_high)
    return env._normalize_act(physical_action).astype(np.float32)


def generate_heuristic_transitions(env, min_timesteps: int) -> types.Transitions:
    """Roll out the heuristic expert in the current environment."""
    if min_timesteps <= 0:
        raise ValueError("min_timesteps must be positive for heuristic demonstrations")

    obs_items = []
    acts = []
    next_obs_items = []
    dones = []
    infos = []

    while len(acts) < min_timesteps:
        obs, _ = env.reset()
        done = False
        while not done and len(acts) < min_timesteps:
            action = heuristic_action(env, obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            obs_items.append(obs)
            acts.append(action)
            next_obs_items.append(next_obs)
            dones.append(done)
            infos.append({})

            obs = next_obs

    return types.Transitions(
        obs=np.asarray(obs_items, dtype=np.float64),
        acts=np.asarray(acts, dtype=np.float32),
        infos=np.asarray(infos, dtype=object),
        next_obs=np.asarray(next_obs_items, dtype=np.float64),
        dones=np.asarray(dones, dtype=bool),
    )


def load_expert_transitions(csv_path: str, env, max_rows: int = 0) -> types.Transitions:
    """Load MPC expert CSV data as normalized state-action transitions.

    The paper defines the GASAC expert data as state-action pairs from MPC.
    This function maps the saved MPC CSV columns to the environment's normalized
    observation and action spaces.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Expert data not found: {path}")

    data = pd.read_csv(path)
    if max_rows and max_rows > 0:
        data = data.iloc[:max_rows].copy()
    if len(data) < 2:
        raise ValueError("Expert data must contain at least two rows")

    raw_obs = np.vstack([_state_row(row, env, row_index) for row_index, (_, row) in enumerate(data.iloc[:-1].iterrows())])
    raw_next_obs = np.vstack(
        [_state_row(row, env, row_index + 1) for row_index, (_, row) in enumerate(data.iloc[1:].iterrows())]
    )
    acts = np.vstack([_action_row(row, env) for _, row in data.iloc[:-1].iterrows()])

    obs = env._normalize_obs(raw_obs).astype(np.float64)
    next_obs = env._normalize_obs(raw_next_obs).astype(np.float64)
    dones = np.zeros(len(acts), dtype=bool)
    dones[-1] = True
    infos = np.array([{} for _ in range(len(acts))], dtype=object)

    return types.Transitions(obs=obs, acts=acts, infos=infos, next_obs=next_obs, dones=dones)
