import numpy as np
import os
import torch
import random

from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

def closest_number(num):
    # 生成范围内的数
    possible_values = [i * 0.5 for i in range(int(16 * 2), int(28 * 2) + 1)]
    # 找到与num最近的数
    closest_val = min(possible_values, key=lambda x: abs(x - num))
    return closest_val

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


def learning_rate_schedule(initial_value: float):
    def func(progress: float) -> float:
        lr = initial_value / (1 + 0.001 * progress)
        return lr
    return func

class SaveOnBestTrainingRewardCallback_(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 假设 self.training_env 封装了 VecNormalize
            x = self.training_env.get_attr("episode_rewards")[0]
            # x = self.training_env.get_attr("ep_rew_mean")
            if len(x) > 0:
                mean_reward = np.mean(x)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.log_dir + "best_model")
                    if self.verbose > 0:
                        print("Best model saved with mean reward: ", self.best_mean_reward)
        return True


if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    data = pd.read_csv(r'E:\sci\7_DR\updated_usa_hourly_data.csv',
                       index_col=0)
    data.index = pd.to_datetime(data.index)
    plt.figure(figsize=(7, 5))
    data_monthly = sns.lineplot(x=data.index.month, y=data['Power'], ci=None)
    monthly_means = data.groupby(data.index.month)['Power'].mean()

    # 保存为 CSV 文件
    output_df = pd.DataFrame({'Month': range(1, 13), 'Average Sunspots': monthly_means.values})
    output_df.to_csv(r'E:\sci\7_DR\updated_usa_hourly_data_monthly.csv', index=False)

    print(data_monthly)
    plt.xlabel('Month')
    plt.ylabel('Number of Sunspots')
    plt.title('Seasonal Plot')
    plt.xticks(range(1, 13), labels=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.show()
