"""Project entry points for RL and imitation learning experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gail_control.training.experiment import Exp


def default_policy_kwargs(rl_config=None):
    hidden_size = [128, 128] if rl_config is not None and rl_config.mode == "il" else [256, 256]
    return {
        "activation_fn": torch.nn.ReLU,
        "net_arch": {"qf": hidden_size, "pi": hidden_size},
    }


def normalize_policy_kwargs(rl_config) -> None:
    """Keep CLI strings from being passed directly into stable-baselines3."""
    if rl_config.policy_kwargs is None:
        rl_config.policy_kwargs = default_policy_kwargs(rl_config)
    elif not isinstance(rl_config.policy_kwargs, dict):
        rl_config.policy_kwargs = None


def build_setting(rl_config, model_id: Optional[str] = None) -> str:
    model_id = model_id or rl_config.model_id
    setting = (
        f"{model_id}_{rl_config.policy}_{rl_config.il_policy}_{rl_config.price}_{rl_config.ac_control}_"
        f"ts{rl_config.timesteps}_ep{rl_config.epoches}_wT{rl_config.wT}_wEc{rl_config.wEc}_"
        f"wPV{rl_config.wPV}_wCO{rl_config.wCO}_wS{rl_config.wSell}_wB{rl_config.wBuy}_"
        f"wSSR{rl_config.wSSR}_seed{rl_config.seed}"
    )
    if rl_config.use_action:
        setting += "_act"
    if rl_config.use_next_state:
        setting += "_nextobs"
    if rl_config.use_pv_forecast:
        setting += "_pvforecast"
    if rl_config.use_To_forecast:
        setting += "_toforecast"
    return setting


def _load_agent(agent, rl_config, setting: str, env):
    model_path = rl_config.model_path
    if model_path is None:
        model_path = Path(rl_config.results_dir) / setting / "best_model"
    print(f"load agent from {model_path} ...")
    return agent.load(str(model_path), env=env, device=rl_config.device)


def main(config: dict, setting: Optional[str] = None, flag: str = "train"):
    """Train/evaluate expert RL, then train/evaluate imitation learner."""
    rl_config = config["rl_config"]
    normalize_policy_kwargs(rl_config)
    setting = setting or build_setting(rl_config)

    exp = Exp(config, setting)
    env = exp.build_env(config=config)
    eval_env = exp.build_env(config=config, flag="eval")
    test_env = exp.build_env(config=config, flag="test")

    expert = None
    if rl_config.expert_source == "policy":
        expert = exp.build_agent(rl_config, env)
        callback = exp.callback_fuc(rl_config, eval_env=eval_env)

        if rl_config.is_training:
            print("train new expert agent ...")
            expert = exp.train(expert, env, callback)
            exp.test_episode(expert, test_env, output=True)
        else:
            expert = _load_agent(expert, rl_config, setting, env)
            exp.test_episode(expert, test_env, output=True)
    else:
        print(f"prepare {rl_config.expert_source} expert demonstrations ...")

    learner = exp.build_agent(rl_config, env)
    if rl_config.expert_source == "policy":
        exp.test_episode(learner, test_env, output=False)
    il_trainer = exp.build_il_trainer(learner, expert, env)

    print("train imitation learner ...")
    rounds = max(1, int(rl_config.epoches / rl_config.test_freq))
    for _ in range(rounds):
        learner = exp.train_il(il_trainer, epoch=int(rl_config.test_freq))

    result = exp.test_episode(learner, test_env, output=True)
    torch.cuda.empty_cache()
    return result


def main_rl(config: dict, setting: Optional[str] = None) -> Tuple[object, dict]:
    """Train or evaluate the RL expert only."""
    rl_config = config["rl_config"]
    normalize_policy_kwargs(rl_config)
    setting = setting or build_setting(rl_config)

    exp = Exp(config, setting)
    env = exp.build_env(config=config)
    eval_env = exp.build_env(config=config, flag="eval")
    test_env = exp.build_env(config=config, flag="test")

    expert = exp.build_agent(rl_config, env)
    callback = exp.callback_fuc(rl_config, eval_env=eval_env)

    if rl_config.is_training:
        print("train new expert agent ...")
        expert = exp.train(expert, env, callback)
    else:
        expert = _load_agent(expert, rl_config, setting, env)

    return exp.test_episode(expert, test_env, output=True)


def run_from_cli(argv=None):
    from gail_control.config import get_config

    config = get_config(argv)
    rl_config = config["rl_config"]
    setting = build_setting(rl_config)
    if rl_config.mode == "il":
        return main(config, setting)
    return main_rl(config, setting)


if __name__ == "__main__":
    run_from_cli()
