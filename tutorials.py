import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(data):
    fontsize = 16
    # 构造DataFrame
    df = pd.DataFrame(data)

    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    cmap = sns.diverging_palette(250, 14, as_cmap=True)
    # cmap = 'coolwarm'

    # 绘制 max_load 的热力图
    plt.figure(figsize=(6, 5),dpi=600)
    pivot_max_load = df.pivot(index="pm", columns="pop_size", values="max_load")
    sns.heatmap(
        pivot_max_load,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={'shrink': 1.0,'aspect': fontsize-2},
        linewidths=0.5,
        linecolor='black',
        annot_kws={"size": fontsize-2},
        # vmin=140,
        # vmax=300
    )
    plt.xlabel("Population size (pop size)", fontsize=fontsize)
    plt.ylabel("Mutation rate (pm)", fontsize=fontsize)
    # 设置 color bar 的字体大小
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.set_title('Load (kWh)', fontsize=fontsize, pad=6)
    cbar.ax.tick_params(labelsize=fontsize-2)
    # 设置坐标轴刻度方向和大小
    plt.gca().tick_params(axis='x', length=0,labelsize=fontsize,)  # Remove tick lines on x-axis
    plt.gca().tick_params(axis='y', length=0,labelsize=fontsize,)  # Remove tick lines on y-axis
    plt.grid(visible=False)

    # 绘制 cost 的热力图
    plt.figure(figsize=(6, 5),dpi=600)
    pivot_cost = df.pivot(index="pm", columns="pop_size", values="cost")
    sns.heatmap(
        pivot_cost,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={'shrink': 1.0,'aspect': fontsize-2},
        linewidths=0.5,
        linecolor='black',
        annot_kws={"size": fontsize-2},
        # vmin=5,
        # vmax=40
    )
    plt.xlabel("Population size (pop size)", fontsize=fontsize)
    plt.ylabel("Mutation rate (pm)", fontsize=fontsize)
    # 设置 color bar 的字体大小
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.set_title('Cost (dollar)', fontsize=fontsize, pad=6)
    cbar.ax.tick_params(labelsize=fontsize-2)

    # 设置坐标轴刻度方向和大小
    plt.gca().tick_params(axis='x', length=0,labelsize=fontsize,)  # Remove tick lines on x-axis
    plt.gca().tick_params(axis='y', length=0,labelsize=fontsize,)  # Remove tick lines on y-axis
    plt.grid(visible=False)
    plt.show()

if __name__ == '__main__':
    # 0.5 12
    data1 = {
        "pm": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "pop_size": [50, 100, 200, 50, 100, 200, 50, 100, 200, 50, 100, 200],
        "max_load": [161.92, 147.29, 145.97, 145.94, 146.11, 157.65, 242.9, 146.48, 162.84, 145.94, 145.94, 146],
        "cost": [11.16858065, 12.85064516, 10.95696774, 10.95696774, 11.24967742, 10.9863871, 6.144967742, 11.456, 10.99451613, 11.38025806, 10.95696774, 18.67954839]
    }
    # 0.5 6
    data2 = {
        "pm": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "pop_size": [50, 100, 200, 50, 100, 200, 50, 100, 200, 50, 100, 200],
        "max_load": [190.6, 145.94, 162.98, 149.95, 146.68, 145.94, 284.74, 182.4, 277.93, 148.35, 145.94, 146.17],
        "cost": [11.73445, 10.95697, 11.30406, 10.95697, 17.86142, 10.95697, 11.54213, 10.96981, 11.09948, 11.17252, 10.95697, 20.63994]
    }
    # 0.3 12
    data3 = {
        "pm": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "pop_size": [50, 100, 200, 50, 100, 200, 50, 100, 200, 50, 100, 200],
        "max_load": [204.32, 204.32, 204.32, 205.37, 204.32, 204.32, 210.63, 204.32, 204.58, 204.32, 204.38, 204.32],
        "cost": [34.68871, 27.74955, 34.00794, 30.75342, 34.02419, 34.04613, 27.72065, 36.67742, 34.00432, 28.11090, 34.79039, 29.68794]
    }
    # 0.3 6
    data4 = {
        "pm": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "pop_size": [50, 100, 200, 50, 100, 200, 50, 100, 200, 50, 100, 200],
        "max_load": [204.32, 204.32, 204.32, 204.32, 204.32, 204.32, 204.32, 204.32, 204.32, 204.32, 204.32, 204.32],
        "cost": [34.00793548, 34.00793548, 34.48541935, 34.14522581, 34.03187097, 34.01264516, 34.01896774, 34.07980645, 34.00793548, 34.33941935, 34.03006452, 34.20580645]
    }

    main(data1)