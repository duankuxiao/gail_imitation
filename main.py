import os
from exp.exp import Exp
import torch


def main(config, setting, flag='train'):

    rl_config = config['rl_config']
    exp = Exp(config, setting)
    env = exp.build_env(config=config)
    eval_env = exp.build_env(config=config, flag='eval')
    test_env = exp.build_env(config=config, flag='test')

    expert = exp.build_agent(rl_config, env)
    callback = exp.callback_fuc(rl_config,eval_env=eval_env)

    if flag == 'train':
        ''' train new expert agent '''
        print('train new expert agent ...')
        expert = exp.train(expert, env, callback)
        exp.test_episode(expert, test_env, output=True)

    elif flag == 'load':
        ''' load expert agent  '''
        print('load expert agent ...')
        expert = expert.load(os.path.join(r'D:\gail_imitation\results\sac_gail_fix_ts2016_ep100_wT1_wEc0.01_wPV0.0005_seed9743', 'best_model'), env=env)
        exp.test_episode(expert, test_env,output=True)
    else:
        raise ValueError('flag must be train or load')

    learner = exp.build_agent(rl_config, env)
    '''evaluate the learner before training'''
    exp.test_episode(learner, test_env,output=True)

    il_trainer = exp.build_il_trainer(learner, expert, env)

    # train the learner and evaluate again
    print('train il ....')
    for epoch in range(int(rl_config.epoches/rl_config.test_freq)):
        learner = exp.train_il(il_trainer,epoch=int(rl_config.test_freq))
        # exp.test_episode(learner, env=eval_env,epoch=(epoch+1)*rl_config.test_freq,flag='eval')

    exp.test_episode(learner, test_env,output=True)
    torch.cuda.empty_cache()


def main_rl(config, setting):
    rl_config = config['rl_config']
    exp = Exp(config, setting)
    env = exp.build_env(config=config)
    eval_env = exp.build_env(config=config, flag='eval')
    test_env = exp.build_env(config=config, flag='test')

    expert = exp.build_agent(rl_config, env)
    callback = exp.callback_fuc(rl_config, eval_env=eval_env)

    ''' train new expert agent '''
    print('train new expert agent ...')
    expert = exp.train(expert, env, callback)
    exp.test_episode(expert, test_env, output=True)


if __name__ == '__main__':
    from args import get_config

    config = get_config()
    rl_config = config["rl_config"]
    rl_config.timesteps = 24*12*7  # 46080

    rl_config.policy = 'sac'

    if rl_config.policy == 'ppo':
        rl_config.episodes = 1000000
    elif rl_config.policy == 'sac':
        rl_config.episodes = 500000

    rl_config.epoches = 5000
    rl_config.test_freq = 1000

    rl_config.use_rms_prop = True
    rl_config.normalize = True
    rl_config.policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(qf=[128, 128, 64], pi=[128, 128,64]))
    rl_config.price = 'fix'  # fix normal
    rl_config.use_action = False
    rl_config.use_next_state = False

    rl_config.wT = 1
    rl_config.wEc = 0.01
    rl_config.wPV = 0.001  # 0.001
    rl_config.wCO = 0.002
    rl_config.wS = 0.005
    rl_config.il_policy = 'gail'
    rl_config.random_init = True
    rl_config.verbose = 0

    model_id = 'test'

    setting = '{}_{}_{}_{}_ts{}_ep{}_wT{}_wEc{}_wPV{}_wCO{}_wS{}_seed{}'.format(model_id,rl_config.policy, rl_config.il_policy,rl_config.price,rl_config.timesteps,rl_config.epoches,rl_config.wT,rl_config.wEc,rl_config.wPV,rl_config.wCO,rl_config.wS,rl_config.seed)
    if rl_config.use_action:
        setting = setting + '_act'
    if rl_config.use_next_state:
        setting = setting + '_nextobs'
    if rl_config.pid:
        setting = 'pid_' + setting

    # main(config,setting,flag='train')
    main_rl(config,setting)

