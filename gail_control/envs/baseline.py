"""PID baseline controller and evaluation model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from gail_control.envs.base import BaseEnv, PIDControl
from gail_control.utils.tools import get_random_error


class BaselineModelPID(BaseEnv):
    """Rule-based PID baseline for AC, PV, and battery comparisons."""

    def __init__(self, config, pid_controller: PIDControl, ac: bool = True, pv: bool = True, battery: bool = True):
        super().__init__(config, flag="test")
        rc_config = config["rc_config"]
        rl_config = config["rl_config"]

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
        self.battery_capacity = rc_config.battery_capacity
        self.charge_capacity = rc_config.charge_capacity
        self.discharge_capacity = rc_config.discharge_capacity

        self.path = rl_config.test_data_path
        self.seed_value = rl_config.seed
        self.pid_controller = pid_controller
        self.T_range = rl_config.T_range
        self.Ti_init = rl_config.Ti_init
        self.price = rl_config.price
        self.dt = rc_config.dt
        self.ac = ac
        self.pv = pv
        self.battery_enabled = battery

        folder_name = (
            f"{rl_config.model_id}_{rl_config.price}_{self.ac_control}_"
            f"Kp{pid_controller.Kp}_Ki{pid_controller.Ki}_Kd{pid_controller.Kd}_seed{rl_config.seed}"
        )
        if not self.ac:
            folder_name += "_noac"
        if not self.battery_enabled:
            folder_name += "_nobattery"
        if not self.pv:
            folder_name += "_nopv"

        self.folder_path = str(Path(rl_config.results_dir) / "baseline" / folder_name)
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.battery_power_init = 0
        self.switch_previous = 1

    def _price_series(self):
        if self.price == "dynamic":
            return self.inputs.loc[:, "price"]
        return self.inputs.loc[:, f"price_{self.price}"]

    def _rc_state_init(self) -> Dict[str, np.ndarray]:
        n_rows = len(self.inputs)
        state = {
            "hour": self.inputs["hour"].to_numpy(),
            "To": self.inputs["To"].to_numpy(),
            "T_target": self.inputs["temp_target"].to_numpy(),
            "phi_i": self.inputs["phi_i"].to_numpy(),
            "phi_wall_s": self.inputs["phi_wall_s"].to_numpy(),
            "phi_window_s": self.inputs["phi_window_s"].to_numpy(),
            "home": self.inputs["home"].to_numpy(),
            "Ec_other": self.inputs["Ec_other"].to_numpy(),
            "Ec_pv": self.inputs["Ec_pv"].to_numpy(),
            "price": self._price_series().to_numpy(),
            "Ti": np.zeros(n_rows),
            "Te": np.zeros(n_rows),
            "L": np.zeros(n_rows),
            "L_ratio": np.zeros(n_rows),
            "cost": np.zeros(n_rows),
            "Ec_ac": np.zeros(n_rows),
            "Ec_demand": np.zeros(n_rows),
            "Ec_sell": np.zeros(n_rows),
            "Ec_buy": np.zeros(n_rows),
            "Ec_true": np.zeros(n_rows),
            "battery": np.zeros(n_rows),
            "Ec_charge": np.zeros(n_rows),
            "CO2": np.zeros(n_rows),
        }

        state["Ti"][0] = self.Ti_init
        state["Te"][0] = (self.Ri * state["To"][0] + self.Ro * state["Ti"][0]) / (self.Ri + self.Ro)
        state["Ec_demand"][0] = state["Ec_other"][0]
        state["CO2"][0] = 0.000457 * 1000000 * state["Ec_demand"][0] / 1000
        return state

    def baseline_cal(self) -> Dict[str, np.ndarray]:
        state = self._rc_state_init()
        for t in range(len(self.inputs) - 1):
            error_L = 1 + get_random_error(t + 100, seed=self.seed_value, max_error=0.05)
            error_T = 1 + get_random_error(t + 300, seed=self.seed_value, max_error=0.05)

            switch_previous = state["home"][t - 1] if t > 0 else 0
            switch_current = state["home"][t]

            if switch_previous == 0 and switch_current == 1:
                state["L"][t], state["Ec_ac"][t] = 3000, 1000
                state["L"][t] = np.clip(state["L"][t] * error_L, 0, 5200)
            elif switch_current == 0:
                state["L"][t], state["Ec_ac"][t] = 0, 0
            else:
                pid_load = self.pid_controller.Load_cal(state["T_target"][t], round(state["Ti"][t], 1))
                state["L"][t] = np.clip(pid_load * error_L, 0, 5200) if pid_load >= 700 else 0

                if self.ac_control == "pid":
                    if state["Ti"][t] >= state["T_target"][t] + 1:
                        state["L"][t] = 0
                    else:
                        pid_load = self.pid_controller.Load_cal(state["T_target"][t], round(state["Ti"][t], 1))
                        state["L"][t] = np.clip(pid_load * error_L, 0, 5200) if pid_load >= 700 else 0
                elif self.ac_control == "Tset":
                    current_state = {
                        "Ti": state["Ti"][t],
                        "phi_i": state["phi_i"][t],
                        "To": state["To"][t],
                        "phi_window_s": state["phi_window_s"][t],
                        "Te": state["Te"][t],
                    }
                    load = self._load_from_RC(state["T_target"][t + 1], current_state)
                    state["L"][t] = np.clip(load, 0, 5200) if pid_load >= 700 else 0

                load_for_energy = np.clip(self.w_ec_ac * state["L"][t], 0, 5200) if state["L"][t] >= 700 else 0
                state["Ec_ac"][t] = (
                    self.w_ec_ac
                    * (
                        (-5.3319e-3 * load_for_energy - 3.4284) * state["To"][t]
                        + 3.5117e-5 * load_for_energy**2
                        + 1.07457e-1 * load_for_energy
                        + 96.152
                    )
                    if state["L"][t] >= 700
                    else 30
                )

            if not self.ac:
                state["L"][t], state["Ec_ac"][t] = 0, 0

            state["Ti"][t + 1] = state["Ti"][t] + self.dt / self.Ci * (
                (state["Te"][t] - state["Ti"][t]) / self.Ri
                + (18 - state["Ti"][t]) / self.Rg
                + (22 - state["Ti"][t]) / self.Rn
                + self.Awindow * state["phi_window_s"][t]
                + self.Ai * (state["phi_i"][t] + state["L"][t])
                + self.Av * (state["To"][t] - state["Ti"][t])
            )
            state["Te"][t + 1] = state["Te"][t] + self.dt / self.Ce * (
                (state["Ti"][t] - state["Te"][t]) / self.Ri
                + (state["To"][t] - state["Te"][t]) / self.Ro
                + self.Awall * state["phi_wall_s"][t]
            )
            state["Ti"][t + 1] *= error_T
            state["Te"][t + 1] *= error_T

            state["Ec_demand"][t] = state["Ec_other"][t] + state["Ec_ac"][t]
            state["battery"][t], state["Ec_charge"][t] = self._battery_action(
                Ec_pv=state["Ec_pv"][t],
                Ec_total=state["Ec_demand"][t],
                flag="base",
            )

            if not self.battery_enabled:
                state["Ec_charge"][t], state["battery"][t] = 0, 0

            if not self.pv:
                state["Ec_pv"][t] = 0
                state["Ec_charge"][t], state["battery"][t] = 0, 0

            state["Ec_true"][t] = state["Ec_demand"][t] - (state["Ec_pv"][t] - state["Ec_charge"][t])
            if state["Ec_true"][t] >= 0:
                state["Ec_sell"][t] = 0
                state["Ec_buy"][t] = state["Ec_true"][t]
                state["cost"][t] = (self.dt / 3600) * state["price"][t] * state["Ec_true"][t] / 1000
            else:
                state["Ec_sell"][t] = -state["Ec_true"][t]
                state["Ec_buy"][t] = 0
                state["cost"][t] = (self.dt / 3600) * 10 * state["Ec_true"][t] / 1000
            state["CO2"][t] = 0.000457 * 1000000 * state["Ec_buy"][t] / 1000 + 38 * state["Ec_pv"][t] / 1000
        return state


PID_control = PIDControl
baseline_model_pid = BaselineModelPID


if __name__ == "__main__":
    from gail_control.config import get_config

    config = get_config([])
    rl_config = config["rl_config"]
    rl_config.T_range = 1
    rl_config.model_id = "test"
    rl_config.price = "normal"
    rl_config.seed = 4396
    rl_config.ac_control = "pid"

    pid = PIDControl(Kp=1732, Ki=215, Kd=53)
    baseline = BaselineModelPID(config=config, pid_controller=pid, ac=True, pv=False, battery=False)
    result = baseline.baseline_cal()
    baseline.evaluation(result, output=True, plot=False)
