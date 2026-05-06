import argparse
import datetime
import os.path
import pandas as pd
import numpy as np
import pysolar.solar as pys
from scipy.stats import truncnorm
from collections import namedtuple
import random
import math
import torch.nn as nn
import torch
import pickle


def save_config(config, filepath):
    # 打印文件路径进行检查
    print(f"Saving config to: {filepath}")

    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        pickle.dump(config, f)

def load_config(filepath):
    with open(filepath, 'rb') as f:
        config = pickle.load(f)
    return config


def save_checkpoint(state, filename):
    torch.save(state, filename)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def logsigmoid(input_tensor):
    softplus = nn.Softplus()
    """
    Equivalent to tf.log(tf.sigmoid(a))

    :param input_tensor: (Tensor)
    :return: (Tensor)
    """
    return -softplus(-input_tensor)


def logit_bernoulli_entropy(logits):
    sigmoid = nn.Sigmoid()
    """
    Reference:
    https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51

    :param logits: (Tensor) the logits
    :return: (Tensor) the Bernoulli entropy
    """
    ent = (1. - sigmoid(logits)) * logits - logsigmoid(logits)
    return torch.mean(ent)

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward','info'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


def to_device(device, *args):
    return [x.to(device) for x in args]


def generate_forecast_errors(mpc_pred_len=24, seed=403, mean_error=0, error_increase_per_time_step=0.5, max_error=0.20):
    errors = np.zeros(mpc_pred_len)
    for minutes_ahead in range(mpc_pred_len):
        np.random.seed(seed + minutes_ahead)
        # 计算当前时间点的最大误差界限
        max_error_at_time = error_increase_per_time_step * (minutes_ahead + 1) / 100
        # 根据最大误差界限调整标准差
        std_deviation = max_error_at_time / 3
        if max_error_at_time >= max_error:
            max_error_at_time = max_error
        # 计算截断正态分布的参数
        a, b = -max_error_at_time / std_deviation, max_error_at_time / std_deviation
        # 生成并存储误差值
        errors[minutes_ahead] = truncnorm.rvs(a, b, loc=mean_error, scale=std_deviation, size=1)
    return errors


def get_random_error(t, seed=403, max_error=0.03):
    np.random.seed(seed + t)
    mean = 0
    std = max_error / 3
    lower_bound, upper_bound = -max_error, max_error
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
    sample = truncnorm.rvs(a, b, loc=mean, scale=std, size=1)[0]
    return sample

def parse_target_entropy(value):
    if value == "auto":
        return value
    try:
        # 尝试转换为float
        return float(value)
    except ValueError:
        # 抛出 argparse 的错误，这会被 argparse 捕捉并以合适的方式显示给用户
        raise argparse.ArgumentTypeError("Invalid value for --target_entropy, must be 'auto' or a float.")

def solar_heat_gain_cal(df):
    df.index = pd.to_datetime(df.index)
    latitude = 35.66541646
    longtitude = 139.8928333
    direct_radiation = df['direct_radiation'].values
    diffuse_radiation = df['diffuse_radiation'].values
    azimuth = []
    altitude = []
    for i in range(len(df)):
        date = datetime.datetime(df.index[i].year, df.index[i].month, df.index[i].day, df.index[i].hour, df.index[i].minute, 0, 0,
                                 tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
        azi = pys.get_azimuth(latitude, longtitude, date)
        alt = pys.get_altitude(latitude, longtitude, date)
        azimuth.append(azi)
        altitude.append(alt)
    azimuth, altitude = np.radians(np.array(azimuth)), np.radians(np.array(altitude))

    wall_north_west = (0.5 * diffuse_radiation + direct_radiation * np.cos(altitude) * np.sin(-225 * np.pi / 180 + azimuth)) * 15.44
    wall_sourth_west = (0.5 * diffuse_radiation + direct_radiation * np.cos(altitude) * np.sin(-135 * np.pi / 180 + azimuth)) * 18.54
    wall_north_east = 0.5 * diffuse_radiation * 17.78

    wall_north_west = np.where(wall_north_west > 0, wall_north_west, 0)
    wall_sourth_west = np.where(wall_sourth_west > 0, wall_sourth_west, 0)
    wall_north_east = np.where(wall_north_east > 0, wall_north_east, 0)

    window_north_west = (0.5 * diffuse_radiation + direct_radiation * np.cos(altitude) * np.sin(-225 * np.pi / 180 + azimuth)) * 5.3064
    window_sourth_west = (0.5 * diffuse_radiation + direct_radiation * np.cos(altitude) * np.sin(-135 * np.pi / 180 + azimuth)) * 0.921475
    window_north_west = np.where(window_north_west > 0, window_north_west, 0)
    window_sourth_west = np.where(window_sourth_west > 0, window_sourth_west, 0)

    exterior_wall = wall_north_west * 0.04 + wall_sourth_west * 0.1 + wall_north_east * 0.04
    window = window_sourth_west * 0.1 + window_north_west * 0.135

    df['phi_wall_s'] = exterior_wall
    df['phi_window_s'] = window
    return df


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def THI_to_GHIDNI(dataframe):
    hokui = float(35.712678)
    toukei = float(139.761989)
    hokui_rad = hokui / 180 * np.pi
    toukei_rad = toukei / 180 * np.pi
    df_solar_radiation = dataframe.copy()
    df_solar_radiation["通し日数"] = pd.to_datetime(df_solar_radiation.index, format="%Y-%m-%d %H:%M:%S").dayofyear.values
    df_solar_radiation["通し時刻"] = pd.to_datetime(df_solar_radiation.index, format="%Y-%m-%d %H:%M:%S").hour.values + pd.to_datetime(
        df_solar_radiation.index, format="%Y-%m-%d %H:%M:%S").minute.values / 60

    df_solar_radiation["shiita0"] = (df_solar_radiation["通し日数"] - 1) / 365 * 2 * np.pi
    df_solar_radiation["太陽赤緯(rad)"] = 0.006918 - 0.399912 * np.cos(
        df_solar_radiation["shiita0"]) + 0.070257 * np.sin(df_solar_radiation["shiita0"]) - 0.006758 * np.cos(2 * df_solar_radiation["shiita0"]) + 0.000907 * np.sin(
        2 * df_solar_radiation["shiita0"]) - 0.002697 * np.cos(3 * df_solar_radiation["shiita0"]) + 0.00148 * np.sin(3 * df_solar_radiation["shiita0"])

    df_solar_radiation["均時差(rad)"] = 0.000075 + 0.001868 * np.cos(df_solar_radiation["shiita0"]) - 0.032077 * np.sin(
        df_solar_radiation["shiita0"]) - 0.014615 * np.cos(2 * df_solar_radiation["shiita0"]) - 0.040849 * np.sin(2 * df_solar_radiation["shiita0"])
    df_solar_radiation["均時差(度)"] = df_solar_radiation["均時差(rad)"] / np.pi * 180

    df_solar_radiation["時角(rad)"] = (df_solar_radiation["通し時刻"] - 12) / 12 * np.pi + (toukei - 135) / 180 * np.pi + df_solar_radiation["均時差(rad)"]
    df_solar_radiation["太陽高度(rad)"] = np.arcsin(np.sin(hokui_rad) * np.sin(df_solar_radiation["太陽赤緯(rad)"]) + np.cos(hokui_rad) * np.cos(
            df_solar_radiation["太陽赤緯(rad)"]) * np.cos(df_solar_radiation["時角(rad)"]))
    df_solar_radiation["太陽方位(rad)"] = np.arctan((np.cos(hokui_rad) * np.cos(df_solar_radiation["太陽赤緯(rad)"]) * np.sin(df_solar_radiation["時角(rad)"])) / (
                np.sin(hokui_rad) * np.sin(df_solar_radiation["太陽高度(rad)"]) - np.sin(df_solar_radiation["太陽赤緯(rad)"])))
    df_solar_radiation["太陽高度(度)"] = df_solar_radiation["太陽高度(rad)"] / np.pi * 180

    # 太陽方位(°)計算
    df_solar_radiation["太陽方位(度)"] = df_solar_radiation["太陽方位(rad)"] / np.pi * 180

    df_solar_radiation["太陽方位(度)"].where((df_solar_radiation["時角(rad)"] < 0) & (df_solar_radiation["太陽方位(rad)"] > 0),
        df_solar_radiation["太陽方位(度)"] - 180,inplace=True)
    df_solar_radiation["太陽方位(度)"].where((df_solar_radiation["時角(rad)"] > 0) & (df_solar_radiation["太陽方位(rad)"] < 0),
        df_solar_radiation["太陽方位(度)"] + 180,inplace=True)

    df_solar_radiation["アルベド"] = 0.1  # 仮
    # df_10s["アルベド"] = df_10s["IR下側日射計"] / df_10s["IR上側日射計"]　現状縦置き使えない

    df_solar_radiation["水平面全天日射"] = df_solar_radiation["THI"] * 1000000 / 3600

    # 直散分離 udagawa
    df_solar_radiation["I0"] = 1370 * (1 + 0.033 * np.cos(2 * np.pi * df_solar_radiation["通し日数"] / 365))  # 大気圏外法線面日射量

    df_solar_radiation["太陽高度(度)"].mask(df_solar_radiation["太陽高度(度)"] < 3, 3, inplace=True)
    df_solar_radiation["太陽高度(rad)"] = df_solar_radiation["太陽高度(度)"] * np.pi / 180

    # KTt=Ihol/(Io*sinh)
    df_solar_radiation["KTt"] = df_solar_radiation["水平面全天日射"] / (df_solar_radiation["I0"] * np.sin(df_solar_radiation["太陽高度(rad)"]))
    # KTtc=0.5163+0.333sinh+0.00803sinh^2
    df_solar_radiation["KTtc"] = 0.5163 + 0.333 * np.sin(df_solar_radiation["太陽高度(rad)"]) + 0.00803 * np.sin(df_solar_radiation["太陽高度(rad)"]) ** 2

    # KTt<KTtc Idn=(2.277-1.258*sinh+0.2396sinh^2)Ktt^3Io
    df_solar_radiation["直散分離_法線面直達日射"] = (2.277 - 1.258 * np.sin(df_solar_radiation["太陽高度(rad)"]) + 0.2396 * np.sin(
        df_solar_radiation["太陽高度(rad)"]) ** 2) * df_solar_radiation["KTt"] ** 3 * df_solar_radiation["I0"]
    # Kt >= Kc Idn=(-0.43+1.43KTt)Io
    df_solar_radiation["直散分離_法線面直達日射"].mask(df_solar_radiation["KTt"] > df_solar_radiation["KTtc"],df_solar_radiation["I0"] * (-0.43 + 1.43 * df_solar_radiation["KTt"]),
                                                       inplace=True)

    df_solar_radiation["直散分離_水平面天空日射"] = df_solar_radiation["水平面全天日射"] - df_solar_radiation[
        "直散分離_法線面直達日射"] * np.sin(df_solar_radiation["太陽高度(rad)"])

    # DN の計算値が 4.18MJ/(m2h) を超える場合，計算値を 4.18MJ/(m2h) に置き換え，計算値と 4.18MJ/(m2h)との差を水平面の値に換算し，SH に加算。
    up_limit = 4.18 * 1000000 / 3600
    df_solar_radiation["直散分離_水平面天空日射"].mask(df_solar_radiation["直散分離_法線面直達日射"] > up_limit,df_solar_radiation["水平面全天日射"] - np.sin(
                                                           df_solar_radiation["太陽高度(rad)"]) * up_limit,inplace=True)
    df_solar_radiation["直散分離_法線面直達日射"].mask(df_solar_radiation["直散分離_法線面直達日射"] > up_limit, up_limit, inplace=True)

    # 日射量がマイナス時、０に直す
    df_solar_radiation["直散分離_水平面天空日射"].mask(df_solar_radiation["直散分離_水平面天空日射"] < 0, 0, inplace=True)
    df_solar_radiation["直散分離_法線面直達日射"].mask(df_solar_radiation["直散分離_法線面直達日射"] < 0, 0, inplace=True)

    df_solar_radiation["計算_鉛直面全天日射"] = 0.5 * df_solar_radiation["直散分離_水平面天空日射"] + df_solar_radiation["直散分離_法線面直達日射"] * np.cos(
        df_solar_radiation["太陽高度(rad)"]) * np.cos(df_solar_radiation["太陽方位(rad)"])
    df_solar_radiation["計算_水平面全天日射"] = df_solar_radiation["直散分離_水平面天空日射"] + np.sin(
        df_solar_radiation["太陽高度(rad)"]) * df_solar_radiation["直散分離_法線面直達日射"]

    df_solar_radiation["地物反射入り_計算_鉛直面全天日射"] = df_solar_radiation["計算_鉛直面全天日射"] + df_solar_radiation["アルベド"] * 0.5 * df_solar_radiation["計算_鉛直面全天日射"]
    df_solar_radiation["鉛直面直達日射"] = df_solar_radiation["直散分離_法線面直達日射"] * np.cos(df_solar_radiation["太陽高度(rad)"]) * np.cos(df_solar_radiation["太陽方位(rad)"])
    df_solar_radiation["鉛直面天空日射"] = 0.5 * df_solar_radiation["直散分離_水平面天空日射"]
    df_solar_radiation["鉛直面直達照度"] = 110 * df_solar_radiation["鉛直面直達日射"]
    df_solar_radiation["鉛直面天空照度"] = 120 * df_solar_radiation["鉛直面天空日射"]

    df_solar_radiation["法線面直達日射量"] = df_solar_radiation["直散分離_法線面直達日射"]
    df_solar_radiation["水平面天空日射量"] = df_solar_radiation["直散分離_水平面天空日射"]
    dataframe['diffuse_radiation'] = df_solar_radiation["水平面天空日射量"]
    dataframe['direct_radiation'] = df_solar_radiation["法線面直達日射量"]
    return dataframe


if __name__ == '__main__':
    print('Utility module. Import and call THI_to_GHIDNI or solar_heat_gain_cal from a data script.')
