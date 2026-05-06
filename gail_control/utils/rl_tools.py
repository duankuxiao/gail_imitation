"""RL helper utilities."""

from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


def closest_number(num: float) -> float:
    possible_values = [i * 0.5 for i in range(int(16 * 2), int(28 * 2) + 1)]
    return min(possible_values, key=lambda x: abs(x - num))


def same_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def learning_rate_schedule(initial_value: float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        return initial_value / (1 + 0.001 * progress)

    return schedule


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Save the model when the current environment reward mean improves."""

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        episode_rewards = self.training_env.get_attr("episode_rewards")[0]
        if not episode_rewards:
            return True

        mean_reward = float(np.mean(episode_rewards))
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            save_path = os.path.join(self.log_dir, "best_model")
            self.model.save(save_path)
            if self.verbose > 0:
                print("Best model saved with mean reward:", self.best_mean_reward)
        return True
