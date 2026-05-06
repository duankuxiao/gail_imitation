"""Run a grid of RL experiments and collect metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gail_control.config import get_config
from scripts.train import build_setting, main_rl, normalize_policy_kwargs


def configure_grid_defaults(rl_config) -> None:
    rl_config.timesteps = 12 if rl_config.smoke_test else 24 * 12
    rl_config.policy = "sac"
    rl_config.episodes = 1 if rl_config.smoke_test else 300000
    rl_config.epoches = 1 if rl_config.smoke_test else 5000
    rl_config.test_freq = 1 if rl_config.smoke_test else 1000
    rl_config.save_freq = 1 if rl_config.smoke_test else 10000
    rl_config.check_freq = 5000
    rl_config.use_rms_prop = True
    rl_config.normalize = True
    rl_config.price = "normal"
    rl_config.use_action = False
    rl_config.use_next_state = False
    rl_config.use_pv_forecast = False
    rl_config.ac_control = "Tset"
    rl_config.il_policy = "gail"
    rl_config.random_init = True
    rl_config.verbose = 0
    rl_config.is_training = True
    rl_config.model_id = "base"
    normalize_policy_kwargs(rl_config)


def run_grid(config: dict) -> pd.DataFrame:
    rl_config = config["rl_config"]
    configure_grid_defaults(rl_config)

    all_metrics = []
    run_count = 0
    for wEc in [0, 0.5, 0.1, 0.05, 0.01, 0.005]:
        for wPV in [0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
            if rl_config.grid_max_runs and run_count >= rl_config.grid_max_runs:
                return pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

            rl_config.wT = 1
            rl_config.wEc = wEc
            rl_config.wPV = wPV
            rl_config.wCO = 0
            rl_config.wSell = 0
            rl_config.wBuy = 0
            rl_config.wSSR = 0
            rl_config.w1 = 0
            rl_config.w2 = 0
            rl_config.w3 = 0
            rl_config.w4 = 0
            rl_config.w5 = 0

            setting = build_setting(rl_config)
            _, metrics = main_rl(config, setting)
            metrics_row = pd.DataFrame([{**{"wEc": wEc, "wPV": wPV}, **metrics}])
            all_metrics.append(metrics_row)

            final_metrics = pd.concat(all_metrics, ignore_index=True)
            output_path = Path(rl_config.results_dir) / "res_metrics.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_metrics.to_csv(output_path, index=False)
            run_count += 1

    return pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()


if __name__ == "__main__":
    run_grid(get_config())
