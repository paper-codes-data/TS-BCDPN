import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def load_absorption_files(directory, num_negative_samples):
    """
    读取指定目录下的所有样本吸收系数文件，并分类为负样本和正样本。
    """
    negative_samples = []
    positive_samples = []

    # 按文件名排序读取
    files = sorted([f for f in os.listdir(directory) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))
    for idx, file in enumerate(files):
        file_path = os.path.join(directory, file)
        data = np.loadtxt(file_path)  # 读取两列数据
        absorption_values = data[1:, 1]  # 第二列为吸收系数

        # 按样本类别存储
        if idx < num_negative_samples:
            negative_samples.append(absorption_values)
        else:
            positive_samples.append(absorption_values)

    return np.array(negative_samples), np.array(positive_samples), data[1:, 0]  # 返回负样本、正样本和频率数据


def calculate_confidence_interval(data, confidence=0.95):
    """
    计算置信区间。
    """
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    se = np.std(data, axis=0, ddof=1) / np.sqrt(n)  # 标准误差
    margin = se * 1.96  # 95%置信区间
    return mean - margin, mean + margin


def smooth_with_sg_filter(samples, window_length=27, polyorder=2):
    """
    对样本数据应用Savitzky-Golay滤波进行平滑。
    """
    # 检查数据长度，确保窗口大小适合数据
    if samples.shape[1] < window_length:
        raise ValueError(f"Sample length is smaller than the window length. Window length: {window_length}, sample length: {samples.shape[1]}")

    # 对每个样本进行S-G滤波，确保窗口大小为奇数
    if window_length % 2 == 0:
        window_length += 1

    return savgol_filter(samples, window_length, polyorder, axis=1)


def plot_mean_absorption_with_bands(freq, negative_samples, positive_samples,
                                    output_path="absorption_plot_with_bands1.svg"):
    """
    计算两类样本的平均吸收系数，并绘制平均值曲线及宽带。
    """
    # 对负样本和正样本应用S-G滤波
    smoothed_negative_samples = smooth_with_sg_filter(negative_samples)
    smoothed_positive_samples = smooth_with_sg_filter(positive_samples)

    # 计算平均值
    mean_negative = np.mean(smoothed_negative_samples, axis=0)
    mean_positive = np.mean(smoothed_positive_samples, axis=0)

    # 极值范围
    min_negative, max_negative = np.min(smoothed_negative_samples, axis=0), np.max(smoothed_negative_samples, axis=0)
    min_positive, max_positive = np.min(smoothed_positive_samples, axis=0), np.max(smoothed_positive_samples, axis=0)

    # 置信区间
    ci_neg_lower, ci_neg_upper = calculate_confidence_interval(smoothed_negative_samples)
    ci_pos_lower, ci_pos_upper = calculate_confidence_interval(smoothed_positive_samples)

    # 均值 ± 标准差范围
    std_neg_lower, std_neg_upper = mean_negative - np.std(smoothed_negative_samples, axis=0), mean_negative + np.std(
        smoothed_negative_samples, axis=0)
    std_pos_lower, std_pos_upper = mean_positive - np.std(smoothed_positive_samples, axis=0), mean_positive + np.std(
        smoothed_positive_samples, axis=0)

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.xlim(0.1, 2)

    # 平均值曲线
    plt.plot(freq, mean_negative, label="Normal Samples (Mean)", color="lightblue", linewidth=2)
    plt.plot(freq, mean_positive, label="Cancer Samples (Mean)", color="lightcoral", linewidth=2)

    # 标准差范围
    plt.fill_between(freq, std_neg_lower, std_neg_upper, color="lightblue", alpha=0.5,
                     label="Normal Samples (Mean ± 1 Std)")
    plt.fill_between(freq, std_pos_lower, std_pos_upper, color="lightcoral", alpha=0.5,
                     label="Cancer Samples (Mean ± 1 Std)")

    # 图形设置
    plt.xlim(0.2, 1.5)
    # plt.ylim(-30, 75)
    plt.xlabel("Frequency (THz)", fontsize=18)  # 横坐标标签字体
    plt.ylabel("Absorption Coefficient ($\\mathrm{cm^{-1}}$)", fontsize=18)  # 纵坐标标签字体
    # plt.title("Absorption Coefficients with Statistical Bands", fontsize=16)  # 标题字体
    plt.legend(fontsize=18)  # 图注字体
    plt.grid(True)

    # 保存为SVG格式
    plt.savefig(output_path, format='svg')
    plt.show()


if __name__ == "__main__":
    # 数据目录和参数
    data_directory = "../data/absorption_coefficient"
    num_negative_samples = 440  # 负样本数量

    # 加载数据
    negative_samples, positive_samples, freq = load_absorption_files(data_directory, num_negative_samples)

    # 绘制平均值曲线及宽带
    plot_mean_absorption_with_bands(freq, negative_samples, positive_samples)
