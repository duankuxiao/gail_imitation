"""Run stable-baselines3 environment validation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gail_control.config import get_config
from gail_control.envs.residential import CustomEnv
from stable_baselines3.common.env_checker import check_env


def main() -> None:
    config = get_config([])
    env = CustomEnv(config, setting="check", flag="train")
    check_env(env)


if __name__ == "__main__":
    main()
