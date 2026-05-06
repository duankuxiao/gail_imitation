"""Utilities for recalculating rewards from saved simulation CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from args import get_config
from utils.tools import load_config


def shift_column_up(dataframe: pd.DataFrame, column_list) -> pd.DataFrame:
    data = dataframe.copy()
    missing_columns = [column for column in column_list if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    for column in column_list:
        data[column] = data[column].shift(-1)
        data.loc[data.index[-1], column] = np.nan
    return data


def random_consecutive_sum_average(
    dataframe: pd.DataFrame,
    column_name: str,
    num_values: int,
    num_samples: int,
) -> float:
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")
    if len(dataframe[column_name]) < num_values:
        raise ValueError(f"Not enough data in column '{column_name}' to extract {num_values} consecutive values")

    sums = []
    for _ in range(num_samples):
        start_index = np.random.randint(0, len(dataframe[column_name]) - num_values + 1)
        consecutive_values = dataframe[column_name].iloc[start_index : start_index + num_values]
        sums.append(consecutive_values.sum())
    return float(np.mean(sums))


def _column(data: pd.DataFrame, *candidates: str):
    for candidate in candidates:
        if candidate in data.columns:
            return data[candidate]
    raise ValueError(f"None of these columns exist: {candidates}")


def reward_function(rl_config, rc_config, data: pd.DataFrame) -> pd.Series:
    cost = _column(data, "cost")
    battery = _column(data, "battery")
    pv = _column(data, "Ec_pv", "PV")
    ec_total = _column(data, "Ec_demand", "Ec_total")
    charge = _column(data, "Ec_charge", "charge")
    sell = _column(data, "Ec_sell", "sell")
    home = _column(data, "home")
    target = _column(data, "T_target", "temp_target")

    temperature_delta = abs(data["Ti"] - target)
    within_target_range = temperature_delta < rl_config.T_range

    reward = pd.Series(0.0, index=data.index)
    reward += rl_config.wT * home * within_target_range
    reward += -rl_config.wT * (temperature_delta ** 2) * home * (~within_target_range)
    reward += -rl_config.wEc * cost

    pv_condition = pv > 0
    reward += -rl_config.wPV * abs(pv - (ec_total + charge + sell)) * pv_condition
    reward += -rl_config.wPV * abs(ec_total + charge) * (~pv_condition)
    return reward / rl_config.timesteps


def calculate_reward_average(filepath, filename, config) -> float:
    file = Path(filepath) / filename
    data = pd.read_csv(file, index_col=0)
    if "sell" not in data.columns:
        data["sell"] = np.maximum(-data["Ec_true"], 0)

    data = shift_column_up(data, ["Ti", "home", "temp_target" if "temp_target" in data.columns else "T_target"])
    data["reward"] = reward_function(config["rl_config"], config["rc_config"], data.iloc[:46080, :])
    average_sum = random_consecutive_sum_average(data, "reward", config["rl_config"].timesteps, 512)
    print(average_sum)
    return average_sum


if __name__ == "__main__":
    config = get_config([])
    print("Use calculate_reward_average(filepath, filename, config) with a saved result CSV.")
