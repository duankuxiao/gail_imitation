import numpy as np
import pandas as pd

from args import get_config
import os

from utils.tools import load_config


def shift_column_up(dataframe, column_list):
    data = dataframe.copy()
    for col in column_list:
        # 检查列是否存在
        if col not in data.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame")
        # 删除指定列的第一个数据
        data[col] = data[col].shift(-1)
        # 删除最后一个数据（因为它会是NaN）
        data[col].iloc[-1] = None
    return data


def random_consecutive_sum_average(dataframe, column_name, num_values, num_samples):
    # 检查列是否存在
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")

    # 检查是否有足够的数据
    if len(dataframe[column_name]) < num_values:
        raise ValueError(f"Not enough data in column '{column_name}' to extract {num_values} consecutive values")

    sums = []
    for _ in range(num_samples):
        start_index = np.random.randint(0, len(dataframe[column_name]) - num_values + 1)
        consecutive_values = dataframe[column_name].iloc[start_index:start_index + num_values]
        sums.append(consecutive_values.sum())

    average_sum = np.mean(sums)

    return average_sum


def reward_function(rl_config,rc_config, data: pd.DataFrame) -> float:
    cost = data['cost']
    battery = data['battery']
    PV = data['PV']
    Ec_total = data['Ec_total']
    charge = data['charge']
    sell = data['sell']
    onhome = data['home']
    T_delta = abs(data['Ti'] - data['temp_target'])
    within_target_range = T_delta < rl_config.T_range

    reward = pd.Series(0, index=data.index)
    reward += rl_config.c * onhome * within_target_range
    reward += - rl_config.wT * T_delta * onhome * (~within_target_range)

    reward += -rl_config.wEc * cost

    battery_condition = battery <= rc_config.battery_capacity * 0.95
    reward += -rl_config.wPV * abs(PV - (Ec_total + charge)) * battery_condition
    reward += -rl_config.wPV * abs(PV - (Ec_total + charge + sell)) * (~battery_condition)
    return reward / rl_config.timesteps


def main(filepath,filename,config):
    file = os.path.join(filepath, filename)
    data = pd.read_csv(file, index_col=0)
    data['sell'] = - data['Ec_true']
    data.loc[data['sell'] <= 0, 'sell'] = 0
    data = shift_column_up(data, ['Ti', 'home', 'temp_target'])

    data['reward'] = reward_function(config['rl_config'], config['rc_config'], data.iloc[:46080, :])
    average_sum = random_consecutive_sum_average(data, 'reward', config['rl_config'].timesteps, 512)
    print(average_sum)
    return average_sum


if __name__ == '__main__':
    filepath = r'D:\gail_imitation\results\sac_gail_normal_ts864_ep150000_wT1_wEc0.1_wPV0.001_seed9743'
    filename = 'sr_eval3.87_cost_eval-21.66_mape1.82_mse0.2588_sr95.53_Ec402.09_cost13618.7' + '.csv'
    config = load_config(os.path.join(filepath,'config.pkl'))

    ep_mean_reward_expert = main(filepath,filename,config)
    filepath = r'D:\gail_imitation\results\baseline'
    filename = 'pricenormal_mape0.0213_mse0.6217_Ec263_cost11194_sr0.9197_pid_Kp670_Ki166_Kd60_dt300' + '.csv'
    ep_mean_reward_baseline = main(filepath,filename,config)

    # file = r'E:\sci\8_GAILforACPV\res\seed.csv'
    # df = pd.read_csv(file, index_col=0)
    # df['max_value'] = df.max(axis=1)
    # df['min_value'] = df.min(axis=1)
    # df['mean_value'] = df.mean(axis=1)
    # df.to_csv(file)




