import argparse

def get_config():
    rc_parser = argparse.ArgumentParser(description='RC Model Variable')
    rc_parser.add_argument('--dt', type=int, default=300, help='dt')
    rc_parser.add_argument('--Ri', type=float, default=1.083801e-04, help='Ri')
    rc_parser.add_argument('--Ro', type=float, default=1.259751e+03, help='Ro')  # TODO origin 1.259751e+03
    rc_parser.add_argument('--Rg', type=float, default=8.305594e+03, help='Rg')
    rc_parser.add_argument('--Rn', type=float, default=7.160465e-04, help='Rg')
    rc_parser.add_argument('--Ci', type=float, default=3.087783e+07, help='Ci')
    rc_parser.add_argument('--Ce', type=float, default=5.212502e+08, help='Ce')
    rc_parser.add_argument('--Ai', type=float, default=7.589300e+01, help='Ai')  # TODO origin 7.589300e+01
    rc_parser.add_argument('--Awindow', type=float, default=1.244492e+02, help='Awindow')
    rc_parser.add_argument('--Awall', type=float, default=2.770246e+01, help='Awall')
    rc_parser.add_argument('--Av', type=float, default=1.110868e+04, help='Awall')  # TODO origin 1.110868e+04
    rc_parser.add_argument('--battery_capacity', type=float, default=30000/6, help='Battery capacity  30000 / 6')
    rc_parser.add_argument('--charge_capacity', type=float, default=4500/6, help='Battery charge capacity  21000 / 6')
    rc_parser.add_argument('--discharge_capacity', type=float, default=4500/6, help='Battery discharge capacity  22000 / 6')
    rc_config = rc_parser.parse_args()

    # RL parameters
    parser = argparse.ArgumentParser(description='RL parameters')
    parser.add_argument('--expert_data_path', type=str, default='.\data\expert_data.csv', help='input data')
    parser.add_argument('--train_data_path', type=str, default='.\data\simulation_data_2017_2018.csv', help='train data')
    parser.add_argument('--test_data_path', type=str, default='.\data\simulation_data_2018_2019.csv', help='test data')

    # training
    parser.add_argument('--epoches', type=int, default=10000, help='episodes')

    parser.add_argument('--episodes', type=int, default=500000, help='episodes')
    parser.add_argument('--timesteps', type=int, default=24*12*7, help='timesteps')
    parser.add_argument('--reward', type=str, default='linear', help='dl or linear')
    parser.add_argument('--activation', type=str, default='relu', help='tanh or relu or sigmoid')
    parser.add_argument('--obs', type=str, default='box', help='box or dict')


    parser.add_argument('--T_delta', action='store_true', default=False, help='T_delta')
    parser.add_argument('--T_range', type=float, default=1, help='T_range')
    parser.add_argument('--random_init', action='store_true', default=False, help='random')
    parser.add_argument('--use_action', type=bool, default=False, help='use_action')
    parser.add_argument('--use_next_state', type=bool, default=False, help='use_next_state')
    parser.add_argument('--use_pv_forecast', type=bool, default=False, help='use_pv_forecast')

    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    # parser.add_argument('--action_variable', type=str, default='L', help='[T, setpoint]')
    parser.add_argument('--ac_control', type=str, default='base', help='use pid controller')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--input_policy', type=str, default='MlpPolicy', help='MultiInputPolicy MlpPolicy  agent method')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=10000, help='save frequency')

    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages')
    parser.add_argument('--seed', type=int, default=9743, help='dt')
    parser.add_argument('--Ti_init', type=float, default=22, help='Ti_init')
    parser.add_argument('--wT', type=float, default=1, help='wT')
    parser.add_argument('--wEc', type=float, default=0.1, help='wEc')
    parser.add_argument('--c', type=float, default=0, help='c')
    parser.add_argument('--wSell', type=float, default=0, help='wS')
    parser.add_argument('--wSSR', type=float, default=0, help='cB')
    parser.add_argument('--wPV', type=float, default=0.001, help='wPV')
    parser.add_argument('--wCO', type=float, default=0, help='wPV')
    parser.add_argument('--wBuy', type=float, default=0, help='wBuy')
    parser.add_argument('--w_ec_ac', type=int, default=0.9, help='w_ec_ac')

    parser.add_argument('--w1', type=float, default=0.5, help='wPV')
    parser.add_argument('--w2', type=float, default=0.25, help='wPV')
    parser.add_argument('--w3', type=float, default=0.25, help='wPV')
    parser.add_argument('--w4', type=float, default=0.4, help='wPV')
    parser.add_argument('--w5', type=float, default=0.5, help='wPV')

    parser.add_argument('--policy', type=str, default='sac', help='agent method')
    parser.add_argument('--il_policy', type=str, default='gail', help='bc gail')
    parser.add_argument('--target', type=str, default='fix', help='fix / dynamic / normal')

    # sarsa & qlearning sac pp0
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize')
    parser.add_argument('--use_rms_prop', action='store_true', default=False, help='use_rms_prop')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--e_greed', type=float, default=0.05, help='e_greed')
    parser.add_argument('--ent_coef', type=float, default=0.05, help='ent_coef')

    parser.add_argument('--policy_kwargs', type=str, nargs='+', help='dict(activation_fn=torch.nn.ReLU, net_arch=dict(qf=[128, 64], pi=[128, 64]))')  #
    rl_config = parser.parse_args()

    return dict(rc_config=rc_config, rl_config=rl_config)
