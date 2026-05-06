"""Evaluate a trained RL expert model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gail_control.config import get_config
from scripts.train import build_setting, main_rl


def evaluate(argv=None):
    config = get_config(argv)
    rl_config = config["rl_config"]
    rl_config.is_training = False
    setting = build_setting(rl_config)
    return main_rl(config, setting)


if __name__ == "__main__":
    evaluate()
