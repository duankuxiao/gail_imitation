"""Experiment orchestration for RL and imitation learning."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.algorithms.sqil import SQIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import logger as imit_logger
from imitation.util.networks import RunningNorm
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from gail_control.envs.residential import CustomEnv
from gail_control.imitation.expert_data import generate_heuristic_transitions, load_expert_transitions
from gail_control.utils.rl_tools import learning_rate_schedule
from gail_control.utils.tools import save_config


RL_ALGORITHMS = {
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "a2c": A2C,
    "ddpg": DDPG,
    "td3": TD3,
}

IL_ALGORITHMS = {
    "gail": GAIL,
    "bc": BC,
    "airl": AIRL,
    "sqil": SQIL,
    "dagger": SimpleDAggerTrainer,
}


class Exp:
    """Build environments, agents, trainers, callbacks, and evaluations."""

    def __init__(self, config: Dict[str, Any], setting: str = "gail_imitation"):
        rl_config = config["rl_config"]
        self.config = config
        self.rl_config = rl_config
        self.episodes = rl_config.episodes
        self.verbose = rl_config.verbose
        self.learning_rate = rl_config.learning_rate
        self.gamma = rl_config.gamma
        self.input_policy = rl_config.input_policy
        self.policy_name = rl_config.policy
        self.il_policy_name = rl_config.il_policy
        self.use_action = rl_config.use_action
        self.use_next_state = rl_config.use_next_state
        self.seed = rl_config.seed
        self.epoches = rl_config.epoches
        self.t_range = rl_config.T_range
        self.price = rl_config.price
        self.setting = setting

        self.folder_path = Path(rl_config.results_dir) / setting
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.best_agent_path = self.folder_path

        self.tensorboard_path = self.folder_path / "runs"
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        save_config(config, self.folder_path / "config.pkl")

        self.logger = SummaryWriter(str(self.tensorboard_path / "il_eval"))
        self.custom_logger = imit_logger.configure(
            folder=str(self.tensorboard_path / "il_train"),
            format_strs=["tensorboard", "stdout"],
        )
        print(f"tensorboard --logdir={self.tensorboard_path} --port=6006")

    def build_env(self, config: Dict[str, Any], flag: str = "train") -> gym.Env:
        """Build the train, eval, or test environment."""
        print(f"building {flag} env ...")
        if flag != "train":
            env = CustomEnv(config=config, setting=self.setting, flag=flag)
            env.seed(self.seed)
            return env

        env = DummyVecEnv(
            [lambda: RolloutInfoWrapper(CustomEnv(config=config, setting=self.setting, flag=flag))]
        )
        env.get_attr("unwrapped")[0].seed(self.seed)
        return env

    def build_il_trainer(self, learner, expert, env):
        """Build an imitation learning trainer."""
        if self.il_policy_name not in IL_ALGORITHMS:
            raise ValueError(f"Undefined imitation learning policy: {self.il_policy_name}")

        min_episodes = self.rl_config.demo_min_episodes
        min_timesteps = self.rl_config.demo_min_timesteps
        demo_batch_size = self.rl_config.demo_batch_size
        gen_replay_buffer_capacity = self.rl_config.gen_replay_buffer_capacity
        rng = np.random.default_rng(self.seed)

        print("building imitation learning trainer ...")
        if self.rl_config.expert_source == "csv":
            base_env = env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped
            demonstrations = load_expert_transitions(
                self.rl_config.expert_data_path,
                base_env,
                max_rows=max(min_timesteps + 1, demo_batch_size + 1) if self.rl_config.smoke_test else 0,
            )
        elif self.rl_config.expert_source == "heuristic":
            base_env = env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped
            demonstrations = generate_heuristic_transitions(
                base_env,
                min_timesteps=max(min_timesteps, demo_batch_size),
            )
        else:
            if expert is None:
                raise ValueError("expert policy is required when expert_source='policy'")
            demonstrations = rollout.rollout(
                expert,
                env,
                rollout.make_sample_until(min_timesteps=min_timesteps, min_episodes=min_episodes),
                rng=rng,
                verbose=self.verbose,
            )

        self._last_demonstrations = demonstrations
        self._pretrain_actor_from_demonstrations(
            learner,
            demonstrations,
            epochs=int(getattr(self.rl_config, "bc_pretrain_epochs", 0) or 0),
        )

        if self.il_policy_name in {"gail", "airl"}:
            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                use_action=self.use_action,
                use_next_state=self.use_next_state,
                normalize_input_layer=RunningNorm,
            )
            trainer_cls = IL_ALGORITHMS[self.il_policy_name]
            return trainer_cls(
                demonstrations=demonstrations,
                demo_batch_size=demo_batch_size,
                gen_replay_buffer_capacity=gen_replay_buffer_capacity,
                n_disc_updates_per_round=self.rl_config.disc_updates_per_round,
                venv=env,
                gen_algo=learner,
                reward_net=reward_net,
                log_dir=str(self.tensorboard_path),
                init_tensorboard=False,
                init_tensorboard_graph=False,
                custom_logger=self.custom_logger,
            )

        if self.il_policy_name == "bc":
            return BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=demonstrations,
                rng=rng,
                device=self.rl_config.device,
            )

        if self.il_policy_name == "sqil":
            return SQIL(venv=env, demonstrations=demonstrations, policy=self.input_policy)

        if self.rl_config.expert_source == "csv":
            raise ValueError("DAGGER requires an expert policy; use --expert_source policy")
        scratch_dir = Path(tempfile.mkdtemp(prefix="dagger_", dir=str(self.folder_path)))
        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
            device=self.rl_config.device,
        )
        return SimpleDAggerTrainer(
            venv=env,
            scratch_dir=str(scratch_dir),
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )

    def build_agent(self, rl_config, env: gym.Env):
        """Build a stable-baselines3 agent."""
        if self.policy_name not in RL_ALGORITHMS:
            raise ValueError(f"Undefined RL policy: {self.policy_name}")

        print(f"building agent: {self.policy_name}...")
        agent_cls = RL_ALGORITHMS[self.policy_name]
        common_kwargs = {
            "learning_rate": learning_rate_schedule(rl_config.learning_rate),
            "gamma": rl_config.gamma,
            "verbose": rl_config.verbose,
            "seed": rl_config.seed,
            "tensorboard_log": str(self.tensorboard_path / "rltrain"),
            "device": rl_config.device,
        }

        if self.policy_name == "a2c":
            return agent_cls(
                rl_config.input_policy,
                env=env,
                normalize_advantage=rl_config.normalize,
                use_rms_prop=rl_config.use_rms_prop,
                **common_kwargs,
            )

        if self.policy_name == "ppo":
            return agent_cls(
                rl_config.input_policy,
                env=env,
                batch_size=rl_config.batch_size,
                normalize_advantage=rl_config.normalize,
                policy_kwargs=rl_config.policy_kwargs,
                **common_kwargs,
            )

        if self.policy_name == "sac":
            return agent_cls(
                rl_config.input_policy,
                env=env,
                batch_size=rl_config.batch_size,
                policy_kwargs=rl_config.policy_kwargs,
                **common_kwargs,
            )

        return agent_cls(
            rl_config.input_policy,
            env=env,
            batch_size=rl_config.batch_size,
            **common_kwargs,
        )

    def build_callbacks(self, rl_config, eval_env: gym.Env = None):
        """Build callbacks used during RL training."""
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=3,
            best_model_save_path=str(self.best_agent_path),
            log_path=str(self.best_agent_path),
            eval_freq=rl_config.save_freq,
            deterministic=True,
            render=False,
        )
        return CallbackList([eval_callback])

    def callback_fuc(self, rl_config, eval_env: gym.Env = None):
        """Backward-compatible alias for the original misspelled method."""
        return self.build_callbacks(rl_config, eval_env)

    def train(self, agent, env, callback):
        print("start learning agent...")
        agent.learn(total_timesteps=self.episodes, callback=callback, tb_log_name=self.policy_name)
        best_agent_path = self.best_agent_path / "best_model"
        if best_agent_path.with_suffix(".zip").exists():
            return agent.load(str(best_agent_path), env=env, device=self.rl_config.device)
        return agent

    def train_il(self, il_trainer, epoch: int):
        if self.il_policy_name in {"gail", "airl"}:
            il_trainer.train(epoch)
            self._pretrain_actor_from_demonstrations(
                il_trainer.gen_algo,
                getattr(self, "_last_demonstrations", None),
                epochs=int(getattr(self.rl_config, "bc_posttrain_epochs", 0) or 0),
            )
            return il_trainer.gen_algo
        if self.il_policy_name == "bc":
            il_trainer.train(n_epochs=2000)
            return il_trainer.policy
        if self.il_policy_name == "sqil":
            il_trainer.train(total_timesteps=1000)
            return il_trainer.policy
        if self.il_policy_name == "dagger":
            il_trainer.train(8000)
            return il_trainer.policy
        raise ValueError(f"Undefined imitation learning policy: {self.il_policy_name}")

    def _pretrain_actor_from_demonstrations(self, learner, demonstrations, epochs: int) -> None:
        if epochs <= 0 or not hasattr(learner, "policy") or not hasattr(learner.policy, "actor"):
            return
        if demonstrations is None or not hasattr(demonstrations, "obs") or not hasattr(demonstrations, "acts"):
            return

        batch_size = int(getattr(self.rl_config, "bc_pretrain_batch_size", 256) or 256)
        obs = th.as_tensor(np.asarray(demonstrations.obs).copy(), dtype=th.float32, device=learner.device)
        acts = th.as_tensor(np.asarray(demonstrations.acts).copy(), dtype=th.float32, device=learner.device)
        action_weights = th.ones(acts.shape[-1], dtype=th.float32, device=learner.device)
        if acts.shape[-1] >= 3:
            action_weights[-1] = float(getattr(self.rl_config, "bc_charge_loss_weight", 1.0) or 1.0)
        actor = learner.policy.actor
        actor.train()

        n_samples = obs.shape[0]
        for epoch in range(epochs):
            permutation = th.randperm(n_samples, device=learner.device)
            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                pred_actions = actor(obs[indices], deterministic=True)
                loss = th.nn.functional.mse_loss(
                    pred_actions * action_weights,
                    acts[indices] * action_weights,
                )
                actor.optimizer.zero_grad()
                loss.backward()
                actor.optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            if self.verbose:
                print(f"bc_pretrain_epoch={epoch + 1} loss={np.mean(epoch_losses):.6f}")

    def test_episode(
        self,
        agent,
        env,
        num_episodes: int = 1,
        epoch: int = 0,
        flag: str = "test",
        output: bool = False,
        plot: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        print("start testing agent...")
        info_dict = {
            "To": [],
            "Ti": [],
            "T_target": [],
            "Te": [],
            "switch": [],
            "L": [],
            "Tset": [],
            "home": [],
            "Ec_charge": [],
            "battery": [],
            "Ec_ac": [],
            "Ec_pv": [],
            "Ec_demand": [],
            "Ec_true": [],
            "Ec_buy": [],
            "Ec_sell": [],
            "price": [],
            "cost": [],
            "CO2": [],
        }

        metrics = {}
        res = pd.DataFrame()
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                for key in info_dict:
                    info_dict[key].append(info["output"][key])

            if flag == "eval":
                self.logger.add_scalar("ep/ep_rew_mean", total_reward, epoch)

            res = pd.DataFrame(info_dict)
            res, metrics = env.evaluation(res, output=output, plot=plot)
        return res, metrics
