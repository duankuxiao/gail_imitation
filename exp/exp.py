import os
import tempfile
import gymnasium as gym
import pandas as pd

from stable_baselines3 import DQN, PPO, SAC, A2C, DDPG, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList, EvalCallback, StopTrainingOnMaxEpisodes

from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
import pathlib
from torch.utils.tensorboard import SummaryWriter

from utils.rl_tools import learning_rate_schedule,SaveOnBestTrainingRewardCallback
import numpy as np

from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.sqil import SQIL
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from enviroment.env import CustomEnv
import warnings
from utils.tools import save_config
from imitation.util import logger as imit_logger
from imitation.scripts.train_adversarial import save

class Exp():
    def __init__(self, config: dict, setting: str='gail_imitation'):
        rl_config = config['rl_config']
        self.episodes = rl_config.episodes
        self.verbose = rl_config.verbose
        self.learning_rate = rl_config.learning_rate
        self.gamma = rl_config.gamma
        self.input_policy = rl_config.input_policy
        self.policy_name = rl_config.policy
        self.il_policy_name = rl_config.il_policy
        self.use_action = rl_config.use_action
        self.use_next_state = rl_config.use_next_state
        self.best_reward = None
        self.seed = rl_config.seed
        self.epoches = rl_config.epoches
        self.T_range = rl_config.T_range
        self.price = rl_config.price
        # result save
        self.setting = setting
        self.folder_path = './results/' + setting + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.best_agent_path = self.folder_path
        self.tensorboard_path = self.folder_path + 'runs/'
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        save_config(config,os.path.join(self.folder_path,'config.pkl'))

        self.logger = SummaryWriter(self.tensorboard_path+'/il_eval')
        self.custom_logger = imit_logger.configure(
            folder=self.tensorboard_path+'/il_train/',
            format_strs=["tensorboard", "stdout"],
        )
        print('tensorboard --logdir={} --port=6006'.format(self.tensorboard_path))

    def build_env(self, config:dict,flag='train') -> gym.Env:
        print('building {} env ...'.format(flag))
        if flag != 'train':
            env = CustomEnv(config=config, setting=self.setting, flag=flag)
            env.seed(self.seed)
        else:
            env = DummyVecEnv([lambda: RolloutInfoWrapper(CustomEnv(config=config, setting=self.setting,flag=flag))])
            env.get_attr('unwrapped')[0].seed(self.seed)  # 设置种子
        # env = Monitor(env,reset_keywords={'seed':self.seed})
        return env

    def build_il_trainer(self,learner,expert,env) -> GAIL:
        min_episodes = 400
        min_timesteps = 14 * 24 * 12
        demo_batch_size = 512
        gen_replay_buffer_capacity = 1000000
        print('building imitation learning trainer ...')
        il_trainer_dict = {'gail': GAIL, 'bc': BC, 'airl': AIRL, 'sqil': SQIL, 'dagger':SimpleDAggerTrainer}
        il_trainer = il_trainer_dict[self.il_policy_name]
        if self.il_policy_name == 'gail' or self.il_policy_name == 'airl':
            rollouts = rollout.rollout(expert, env, rollout.make_sample_until(min_timesteps=min_timesteps, min_episodes=min_episodes), rng=np.random.default_rng(self.seed),
                                       verbose=self.verbose)
            reward_net = BasicRewardNet(observation_space=env.observation_space,action_space=env.action_space,use_action=self.use_action,use_next_state=self.use_next_state,
                                        normalize_input_layer=RunningNorm,)  # normalize_input_layer=RunningNorm,
            il_trainer = il_trainer(demonstrations=rollouts,demo_batch_size=demo_batch_size,gen_replay_buffer_capacity=gen_replay_buffer_capacity,n_disc_updates_per_round=5,
                                    venv=env,gen_algo=learner,reward_net=reward_net, log_dir=self.tensorboard_path, init_tensorboard=False, init_tensorboard_graph=False,
                                    custom_logger=self.custom_logger)
        elif self.il_policy_name == 'bc':
            rollouts = rollout.rollout(expert, env, rollout.make_sample_until(min_timesteps=min_timesteps, min_episodes=min_episodes), rng=np.random.default_rng(self.seed),
                                       verbose=self.verbose)
            transitions = rollout.flatten_trajectories(rollouts)
            il_trainer = BC(observation_space=env.observation_space,action_space=env.action_space,demonstrations=transitions,rng=np.random.default_rng(self.seed),device='cpu')
        elif self.il_policy_name == 'sqil':
            rollouts = rollout.rollout(expert, env, rollout.make_sample_until(min_timesteps=min_timesteps, min_episodes=min_episodes), rng=np.random.default_rng(self.seed), verbose=self.verbose)
            il_trainer = SQIL(venv=env,demonstrations=rollouts,policy=self.input_policy,)
        elif self.il_policy_name == 'dagger':
            bc_trainer = BC(observation_space=env.observation_space,action_space=env.action_space,rng=np.random.default_rng(self.seed),)
            with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
                print(tmpdir)
                il_trainer = SimpleDAggerTrainer(venv=env,scratch_dir=tmpdir,expert_policy=expert,bc_trainer=bc_trainer,rng=np.random.default_rng(self.seed),)
        else:
            raise ValueError('Undefined imitation learning policy input')
        return il_trainer

    def build_agent(self, rl_config, env: gym.Env):
        print('building agent: {}...'.format(self.policy_name))
        agent_dict = {'dqn': DQN, 'ppo': PPO, 'sac': SAC, 'a2c': A2C, 'ddpg': DDPG,'td3': TD3}
        agent = agent_dict[self.policy_name]
        if self.policy_name == 'a2c':
            agent = agent(rl_config.input_policy, env=env, batch_size=rl_config.batch_size, normalize_advantage=rl_config.normalize, use_rms_prop=rl_config.use_rms_prop,
                          learning_rate=learning_rate_schedule(rl_config.learning_rate), gamma=rl_config.gamma, verbose=rl_config.verbose,
                          seed=rl_config.seed, tensorboard_log=self.tensorboard_path+'/rltrain/')
        elif self.policy_name == 'ppo':
            agent = agent(rl_config.input_policy, env=env, batch_size=rl_config.batch_size, normalize_advantage=rl_config.normalize, policy_kwargs=rl_config.policy_kwargs,   # ent_coef=-0.001,
                          learning_rate=learning_rate_schedule(rl_config.learning_rate), gamma=rl_config.gamma, verbose=rl_config.verbose,
                          seed=rl_config.seed, tensorboard_log=self.tensorboard_path+'/rltrain/')
        elif self.policy_name == 'sac':
            agent = agent(rl_config.input_policy, env=env, batch_size=rl_config.batch_size, policy_kwargs=rl_config.policy_kwargs,
                          learning_rate=learning_rate_schedule(rl_config.learning_rate), gamma=rl_config.gamma, verbose=rl_config.verbose,
                          seed=rl_config.seed, tensorboard_log=self.tensorboard_path+'/rltrain/')
        elif self.policy_name == 'td3':
            agent = agent(rl_config.input_policy, env=env, batch_size=rl_config.batch_size,
                          learning_rate=learning_rate_schedule(rl_config.learning_rate), gamma=rl_config.gamma, verbose=rl_config.verbose,
                          seed=rl_config.seed, tensorboard_log=self.tensorboard_path+'/rltrain/')
        else:
            raise print('Undefined policy input')
        return agent

    def callback_fuc(self,rl_config,eval_env:gym.Env=None):
        savebest_callback = SaveOnBestTrainingRewardCallback(check_freq=rl_config.check_freq, log_dir=self.best_agent_path)

        checkpoint_callback = CheckpointCallback(save_freq=rl_config.save_freq, save_path=self.best_agent_path)
        eval_callback = EvalCallback(eval_env,n_eval_episodes=3, best_model_save_path=self.best_agent_path, log_path=self.best_agent_path, eval_freq=rl_config.save_freq,  # 每隔 10000 步进行一次评估
                                     deterministic=True, render=False)
        maxepisodes_callback = StopTrainingOnMaxEpisodes(max_episodes=rl_config.episodes, verbose=rl_config.verbose)
        callback = CallbackList([eval_callback])
        return callback

    def train(self, agent, env, callback):
        print('start learning agent...')
        agent.learn(total_timesteps=self.episodes, callback=callback, tb_log_name='{}'.format(self.policy_name),)
        best_agent_path = os.path.join(self.best_agent_path, 'best_model')
        # self.agent.save(best_agent_path)
        agent = agent.load(best_agent_path, env=env)
        return agent

    def train_il(self,il_trainer,epoch):
        if self.il_policy_name == 'gail' or self.il_policy_name == 'airl':
            il_trainer.train(epoch)
            return il_trainer.gen_algo
        elif self.il_policy_name == 'bc':
            il_trainer.train(n_epochs=2000)
            return il_trainer.policy
        elif self.il_policy_name == 'sqil':
            il_trainer.train(total_timesteps=1000)
            return il_trainer.policy
        elif self.il_policy_name == 'dagger':
            il_trainer.train(8000)
            return il_trainer.policy
        else:
            raise print('Undefined imitation learning policy input')
        # save(il_trainer.gen_algo,  pathlib.Path('checkpoints/il_model'))

    def test_episode(self, agent, env, num_episodes=1,epoch=0,flag='test',output=False):
        print('start testing agent...')
        info_dict = dict(To=[], Ti=[], T_target=[], Te=[], switch=[], L=[], Tset=[], home=[],Ec_charge=[], battery=[],Ec_ac=[], Ec_pv=[], Ec_demand=[],  Ec_true=[], Ec_buy=[], Ec_sell=[], price=[], cost=[],CO2=[])
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                for key in info_dict.keys():
                    info_dict[key].append(info['output'][key])
            if flag == 'eval':
                self.logger.add_scalar('ep/ep_rew_mean', total_reward, epoch)
            # print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            res = pd.DataFrame(info_dict)
            res,metrics = env.evaluation(res,output)
        return res, metrics

