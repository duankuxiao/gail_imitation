from stable_baselines3.common.env_checker import check_env
from exp.exp import Exp

from args import get_config
from env.base_env import BaseEnv
from env.env import CustomEnv
config = get_config()

# exp = Exp(config)
# env = exp.build_env(config=config)
env = CustomEnv(config,setting='check')
check_env(env)