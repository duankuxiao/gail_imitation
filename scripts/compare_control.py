"""Run the PID, SAC, and GAIL control comparison experiment.

This script compares:
- M1: rule-based/PID baseline
- M3: direct SAC reinforcement learning
- M4: BC-regularized GASAC/GAIL with configurable expert demonstrations

The default training budgets are intentionally configurable because a
paper-scale CPU run can be very long when CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gail_control.config import get_config
from gail_control.envs.baseline import BaselineModelPID, PIDControl
from gail_control.training.experiment import Exp
from scripts.train import build_setting, normalize_policy_kwargs


def cuda_compute_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        return float((x + 1).cpu()[0]) == 2.0
    except Exception:
        return False


def configure_common(config: dict, device: str, seed: int, price: str) -> dict:
    config = copy.deepcopy(config)
    rl_config = config["rl_config"]
    rl_config.device = device
    rl_config.seed = seed
    rl_config.price = price
    rl_config.policy = "sac"
    rl_config.il_policy = "gail"
    rl_config.input_policy = "MlpPolicy"
    rl_config.ac_control = "Tset"
    rl_config.timesteps = 864
    rl_config.learning_rate = 0.0003
    rl_config.batch_size = 256
    rl_config.gamma = 0.99
    rl_config.random_init = True
    rl_config.use_action = True
    rl_config.use_next_state = False
    rl_config.use_pv_forecast = False
    rl_config.use_To_forecast = False
    rl_config.wT = 1
    rl_config.wEc = 0.001
    rl_config.wPV = 0.001
    rl_config.wCO = 0
    rl_config.wSell = 0
    rl_config.wBuy = 0
    rl_config.wSSR = 0
    rl_config.verbose = 0
    rl_config.results_dir = "results"
    normalize_policy_kwargs(rl_config)
    return config


def energy_diagnostics(result: pd.DataFrame) -> dict:
    """Return cost and battery/PV diagnostics from a saved episode."""
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result)
    if result.empty:
        return {
            "buy_cost": float("nan"),
            "sell_credit": float("nan"),
            "buy_kwh": float("nan"),
            "sell_kwh": float("nan"),
            "charge_kwh": float("nan"),
            "discharge_kwh": float("nan"),
            "pv_self_use_rate": float("nan"),
        }

    scale = 1 / 12 / 1000
    buy_rows = result[result["Ec_true"] >= 0]
    sell_rows = result[result["Ec_true"] < 0]
    pv_sum = result["Ec_pv"].sum()
    return {
        "buy_cost": buy_rows["cost"].sum(),
        "sell_credit": sell_rows["cost"].sum(),
        "buy_kwh": result["Ec_buy"].sum() * scale,
        "sell_kwh": result["Ec_sell"].sum() * scale,
        "charge_kwh": result.loc[result["Ec_charge"] > 0, "Ec_charge"].sum() * scale,
        "discharge_kwh": -result.loc[result["Ec_charge"] < 0, "Ec_charge"].sum() * scale,
        "pv_self_use_rate": 1 - result["Ec_sell"].sum() / pv_sum if pv_sum else float("nan"),
    }


def run_baseline(config: dict) -> dict:
    rl_config = config["rl_config"]
    rl_config.model_id = "paper_m1_baseline"
    rl_config.ac_control = "Tset"

    start = time.perf_counter()
    baseline = BaselineModelPID(
        config=config,
        pid_controller=PIDControl(Kp=1732, Ki=215, Kd=53),
        ac=True,
        pv=True,
        battery=True,
    )
    result = baseline.baseline_cal()
    _, metrics = baseline.evaluation(result, output=True, plot=False)
    elapsed = time.perf_counter() - start
    return {
        "model": "M1_baseline",
        "training_seconds": 0.0,
        "total_seconds": elapsed,
        **metrics,
        **energy_diagnostics(result),
    }


def run_gasac(
    config: dict,
    gasac_steps: int,
    demo_batch_size: int,
    demo_min_timesteps: int,
    expert_source: str,
    bc_pretrain_epochs: int,
    bc_posttrain_epochs: int,
    bc_charge_loss_weight: float,
) -> dict:
    config = copy.deepcopy(config)
    rl_config = config["rl_config"]
    rl_config.mode = "il"
    rl_config.model_id = f"paper_m4_gasac_{gasac_steps}"
    rl_config.epoches = gasac_steps
    rl_config.test_freq = gasac_steps
    rl_config.expert_source = expert_source
    rl_config.demo_min_episodes = 1
    rl_config.demo_min_timesteps = demo_min_timesteps
    rl_config.demo_batch_size = demo_batch_size
    rl_config.gen_replay_buffer_capacity = 1_000_000
    rl_config.disc_updates_per_round = 1
    rl_config.bc_pretrain_epochs = bc_pretrain_epochs
    rl_config.bc_posttrain_epochs = bc_posttrain_epochs
    rl_config.bc_pretrain_batch_size = demo_batch_size
    rl_config.bc_charge_loss_weight = bc_charge_loss_weight
    normalize_policy_kwargs(rl_config)

    setting = build_setting(rl_config)
    exp = Exp(config, setting)
    train_env = exp.build_env(config)
    test_env = exp.build_env(config, flag="test")
    learner = exp.build_agent(rl_config, train_env)
    trainer = exp.build_il_trainer(learner, expert=None, env=train_env)

    start_train = time.perf_counter()
    learner = exp.train_il(trainer, epoch=gasac_steps)
    training_seconds = time.perf_counter() - start_train

    result, metrics = exp.test_episode(learner, test_env, output=True, plot=False)
    return {"model": "M4_GASAC", "training_seconds": training_seconds, **metrics, **energy_diagnostics(result)}


def run_sac(config: dict, sac_steps: int) -> dict:
    config = copy.deepcopy(config)
    rl_config = config["rl_config"]
    rl_config.mode = "rl"
    rl_config.model_id = f"paper_m3_sac_{sac_steps}"
    rl_config.episodes = sac_steps
    rl_config.epoches = sac_steps
    rl_config.save_freq = sac_steps
    rl_config.test_freq = sac_steps
    rl_config.is_training = True
    normalize_policy_kwargs(rl_config)

    setting = build_setting(rl_config)
    exp = Exp(config, setting)
    train_env = exp.build_env(config)
    eval_env = exp.build_env(config, flag="eval")
    test_env = exp.build_env(config, flag="test")
    agent = exp.build_agent(rl_config, train_env)
    callback = exp.callback_fuc(rl_config, eval_env=eval_env)

    start_train = time.perf_counter()
    agent = exp.train(agent, train_env, callback)
    training_seconds = time.perf_counter() - start_train

    result, metrics = exp.test_episode(agent, test_env, output=True, plot=False)
    return {"model": "M3_SAC", "training_seconds": training_seconds, **metrics, **energy_diagnostics(result)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gasac_steps", type=int, default=150_000)
    parser.add_argument("--sac_steps", type=int, default=300_000)
    parser.add_argument("--demo_batch_size", type=int, default=256)
    parser.add_argument("--demo_min_timesteps", type=int, default=46_080)
    parser.add_argument("--expert_source", type=str, default="heuristic", choices=["csv", "heuristic", "policy"])
    parser.add_argument("--bc_pretrain_epochs", type=int, default=5)
    parser.add_argument("--bc_posttrain_epochs", type=int, default=5)
    parser.add_argument("--bc_charge_loss_weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=9743)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--price", type=str, default="normal", choices=["fix", "normal", "dynamic"])
    parser.add_argument("--output_dir", type=str, default="results/control_experiment")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not cuda_compute_available():
        print("CUDA requested but unavailable for compute; falling back to CPU.")
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = configure_common(get_config([]), device=device, seed=args.seed, price=args.price)

    results = []
    print("Running M1 baseline...")
    baseline_result = run_baseline(copy.deepcopy(base_config))
    results.append(baseline_result)
    base_config["rl_config"].baseline_sr_reference = baseline_result["sr"]
    base_config["rl_config"].baseline_cost_reference = baseline_result["cost"]

    print(f"Running M4 GASAC for {args.gasac_steps} timesteps...")
    results.append(
        run_gasac(
            copy.deepcopy(base_config),
            args.gasac_steps,
            args.demo_batch_size,
            args.demo_min_timesteps,
            args.expert_source,
            args.bc_pretrain_epochs,
            args.bc_posttrain_epochs,
            args.bc_charge_loss_weight,
        )
    )

    print(f"Running M3 SAC for {args.sac_steps} timesteps...")
    results.append(run_sac(copy.deepcopy(base_config), args.sac_steps))

    summary = pd.DataFrame(results)
    baseline = summary.loc[summary["model"] == "M1_baseline"].iloc[0]
    for idx, row in summary.iterrows():
        summary.loc[idx, "sr_improve"] = (row["sr"] - baseline["sr"]) / baseline["sr"] * 100
        summary.loc[idx, "cost_improve"] = (baseline["cost"] - row["cost"]) / baseline["cost"] * 100
        summary.loc[idx, "sr_vs_baseline_pct"] = summary.loc[idx, "sr_improve"]
        summary.loc[idx, "cost_vs_baseline_pct"] = summary.loc[idx, "cost_improve"]

    sac_time = float(summary.loc[summary["model"] == "M3_SAC", "training_seconds"].iloc[0])
    gasac_time = float(summary.loc[summary["model"] == "M4_GASAC", "training_seconds"].iloc[0])
    time_saving_pct = (sac_time - gasac_time) / sac_time * 100 if sac_time > 0 else float("nan")

    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "device": device,
                "requested_device": args.device,
                "gasac_steps": args.gasac_steps,
                "sac_steps": args.sac_steps,
                "seed": args.seed,
                "price": args.price,
                "expert_source": args.expert_source,
                "demo_min_timesteps": args.demo_min_timesteps,
                "bc_pretrain_epochs": args.bc_pretrain_epochs,
                "bc_posttrain_epochs": args.bc_posttrain_epochs,
                "bc_charge_loss_weight": args.bc_charge_loss_weight,
                "time_saving_pct": time_saving_pct,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"training_time_saving_pct={time_saving_pct:.2f}")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
