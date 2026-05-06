"""Command line configuration for training and evaluation.

The project historically passed around argparse namespaces.  The public
``get_config`` API keeps that shape for compatibility while centralising all
defaults in one parser.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional


def str_to_bool(value):
    """Parse common command line boolean spellings."""
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        name,
        type=str_to_bool,
        nargs="?",
        const=True,
        default=default,
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GAIL control for residential AC and battery systems")

    rc_group = parser.add_argument_group("RC model")
    rc_group.add_argument("--dt", type=int, default=300, help="Simulation step in seconds")
    rc_group.add_argument("--Ri", type=float, default=1.083801e-04, help="Indoor-envelope thermal resistance")
    rc_group.add_argument("--Ro", type=float, default=1.259751e03, help="Outdoor-envelope thermal resistance")
    rc_group.add_argument("--Rg", type=float, default=8.305594e03, help="Ground thermal resistance")
    rc_group.add_argument("--Rn", type=float, default=7.160465e-04, help="Neighbor thermal resistance")
    rc_group.add_argument("--Ci", type=float, default=3.087783e07, help="Indoor thermal capacitance")
    rc_group.add_argument("--Ce", type=float, default=5.212502e08, help="Envelope thermal capacitance")
    rc_group.add_argument("--Ai", type=float, default=7.589300e01, help="Internal gain coefficient")
    rc_group.add_argument("--Awindow", type=float, default=1.244492e02, help="Window area coefficient")
    rc_group.add_argument("--Awall", type=float, default=2.770246e01, help="Wall area coefficient")
    rc_group.add_argument("--Av", type=float, default=1.110868e04, help="Ventilation coefficient")
    rc_group.add_argument("--battery_capacity", type=float, default=30000 / 6, help="Battery capacity")
    rc_group.add_argument("--charge_capacity", type=float, default=4500, help="Battery charge rate in W")
    rc_group.add_argument("--discharge_capacity", type=float, default=4500, help="Battery discharge rate in W")

    data_group = parser.add_argument_group("Data and outputs")
    data_group.add_argument("--expert_data_path", type=str, default="data/normal_expert_train.csv")
    data_group.add_argument("--train_data_path", type=str, default="data/simulation_data_2017_2018.csv")
    data_group.add_argument("--test_data_path", type=str, default="data/simulation_data_2018_2019.csv")
    data_group.add_argument("--results_dir", type=str, default="results")
    data_group.add_argument("--model_path", type=str, default=None, help="Optional model path for evaluation")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epoches", type=int, default=10000, help="Imitation learning iterations")
    train_group.add_argument("--episodes", type=int, default=500000, help="RL total timesteps")
    train_group.add_argument("--timesteps", type=int, default=864, help="Episode length")
    train_group.add_argument("--learning_rate", type=float, default=0.0003)
    train_group.add_argument("--batch_size", type=int, default=256)
    train_group.add_argument("--save_freq", type=int, default=10000)
    train_group.add_argument("--test_freq", type=int, default=10000)
    train_group.add_argument("--check_freq", type=int, default=5000)
    train_group.add_argument("--verbose", type=int, default=0)
    train_group.add_argument("--seed", type=int, default=9743)
    train_group.add_argument("--is_training", type=str_to_bool, nargs="?", const=True, default=True)
    train_group.add_argument("--mode", type=str, default="rl", choices=["rl", "il"], help="Run RL or imitation learning")
    train_group.add_argument("--device", type=str, default="cpu", help="Torch device used by SB3/imitation")
    train_group.add_argument("--smoke_test", type=str_to_bool, nargs="?", const=True, default=False)
    train_group.add_argument("--grid_max_runs", type=int, default=0, help="Limit run_main grid executions; 0 means unlimited")

    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--reward", type=str, default="linear", choices=["linear", "dl"])
    env_group.add_argument("--activation", type=str, default="relu", choices=["tanh", "relu", "sigmoid"])
    env_group.add_argument("--obs", type=str, default="box", choices=["box", "dict"])
    env_group.add_argument("--Ti_init", type=float, default=22)
    env_group.add_argument("--T_range", type=float, default=1)
    env_group.add_argument("--price", type=str, default="normal", choices=["fix", "normal", "dynamic"])
    env_group.add_argument("--target", type=str, default="fix", choices=["fix", "dynamic", "normal"])
    env_group.add_argument("--ac_control", type=str, default="base", choices=["base", "pid", "Tset", "none"])
    _add_bool_argument(env_group, "--T_delta", False, "Use temperature delta in observations")
    _add_bool_argument(env_group, "--random_init", False, "Randomise training initial state")
    _add_bool_argument(env_group, "--use_action", True, "Include action in imitation reward net")
    _add_bool_argument(env_group, "--use_next_state", False, "Include next-state features")
    _add_bool_argument(env_group, "--use_pv_forecast", False, "Include PV forecast features")
    _add_bool_argument(env_group, "--use_To_forecast", False, "Include outdoor temperature forecast features")

    reward_group = parser.add_argument_group("Reward weights")
    reward_group.add_argument("--wT", type=float, default=1)
    reward_group.add_argument("--wEc", type=float, default=0.1)
    reward_group.add_argument("--c", type=float, default=0)
    reward_group.add_argument("--wSell", type=float, default=0)
    reward_group.add_argument("--wSSR", type=float, default=0)
    reward_group.add_argument("--wPV", type=float, default=0.001)
    reward_group.add_argument("--wCO", type=float, default=0)
    reward_group.add_argument("--wBuy", type=float, default=0)
    reward_group.add_argument("--w_ec_ac", type=float, default=0.9)
    reward_group.add_argument("--w1", type=float, default=0.5)
    reward_group.add_argument("--w2", type=float, default=0.25)
    reward_group.add_argument("--w3", type=float, default=0.25)
    reward_group.add_argument("--w4", type=float, default=0.4)
    reward_group.add_argument("--w5", type=float, default=0.5)

    algo_group = parser.add_argument_group("Algorithms")
    algo_group.add_argument("--policy", type=str, default="sac", choices=["dqn", "ppo", "sac", "a2c", "ddpg", "td3"])
    algo_group.add_argument("--il_policy", type=str, default="gail", choices=["bc", "gail", "airl", "sqil", "dagger"])
    algo_group.add_argument("--input_policy", type=str, default="MlpPolicy")
    _add_bool_argument(algo_group, "--normalize", False, "Normalize advantages where supported")
    _add_bool_argument(algo_group, "--use_rms_prop", False, "Use RMSProp where supported")
    algo_group.add_argument("--gamma", type=float, default=0.99)
    algo_group.add_argument("--e_greed", type=float, default=0.05)
    algo_group.add_argument("--ent_coef", type=float, default=0.05)
    algo_group.add_argument("--policy_kwargs", type=str, nargs="+", default=None)
    algo_group.add_argument("--model_id", type=str, default="base")
    algo_group.add_argument("--demo_min_episodes", type=int, default=400)
    algo_group.add_argument("--demo_min_timesteps", type=int, default=864)
    algo_group.add_argument("--demo_batch_size", type=int, default=864)
    algo_group.add_argument("--gen_replay_buffer_capacity", type=int, default=1_000_000)
    algo_group.add_argument("--disc_updates_per_round", type=int, default=1)
    algo_group.add_argument("--bc_pretrain_epochs", type=int, default=0)
    algo_group.add_argument("--bc_posttrain_epochs", type=int, default=0)
    algo_group.add_argument("--bc_pretrain_batch_size", type=int, default=256)
    algo_group.add_argument("--bc_charge_loss_weight", type=float, default=1.0)
    algo_group.add_argument(
        "--expert_source",
        type=str,
        default="csv",
        choices=["csv", "heuristic", "policy"],
        help="Use CSV, environment-consistent heuristic, or policy rollouts as demonstrations",
    )

    return parser


def _split_namespaces(namespace: argparse.Namespace) -> dict:
    rc_fields = {
        "dt",
        "Ri",
        "Ro",
        "Rg",
        "Rn",
        "Ci",
        "Ce",
        "Ai",
        "Awindow",
        "Awall",
        "Av",
        "battery_capacity",
        "charge_capacity",
        "discharge_capacity",
    }

    rc_config = argparse.Namespace(**{name: getattr(namespace, name) for name in rc_fields})
    rl_config = argparse.Namespace(
        **{name: value for name, value in vars(namespace).items() if name not in rc_fields}
    )
    return {"rc_config": rc_config, "rl_config": rl_config}


def get_config(argv: Optional[Iterable[str]] = None) -> dict:
    """Return config namespaces used by the rest of the project."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    return _split_namespaces(namespace)
