import os
import numpy as np
import matplotlib.pyplot as plt


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
        absorption_values = data[:, 1]  # 第二列为吸收系数

        # 按样本类别存储
        if idx < num_negative_samples:
            negative_samples.append(absorption_values)
        else:
            positive_samples.append(absorption_values)

    return np.array(negative_samples), np.array(positive_samples), data[:, 0]  # 返回负样本、正样本和频率数据


def calculate_confidence_interval(data, confidence=0.95):
    """
    计算置信区间。
    """
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    se = np.std(data, axis=0, ddof=1) / np.sqrt(n)  # 标准误差
    margin = se * 1.96  # 95%置信区间
    return mean - margin, mean + margin


def plot_mean_absorption_with_bands(freq, negative_samples, positive_samples,
                                    output_path="absorption_plot_with_bands.png"):
    """
    计算两类样本的平均吸收系数，并绘制平均值曲线及宽带。
    """
    # 计算平均值
    mean_negative = np.mean(negative_samples, axis=0)
    mean_positive = np.mean(positive_samples, axis=0)

    # 极值范围
    min_negative, max_negative = np.min(negative_samples, axis=0), np.max(negative_samples, axis=0)
    min_positive, max_positive = np.min(positive_samples, axis=0), np.max(positive_samples, axis=0)

    # 置信区间
    ci_neg_lower, ci_neg_upper = calculate_confidence_interval(negative_samples)
    ci_pos_lower, ci_pos_upper = calculate_confidence_interval(positive_samples)

    # 均值 ± 标准差范围
    std_neg_lower, std_neg_upper = mean_negative - np.std(negative_samples, axis=0), mean_negative + np.std(
        negative_samples, axis=0)
    std_pos_lower, std_pos_upper = mean_positive - np.std(positive_samples, axis=0), mean_positive + np.std(
        positive_samples, axis=0)

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.xlim(0.1, 2)

    # 平均值曲线
    plt.plot(freq, mean_negative, label="Negative Samples (Mean)", color="lightblue", linewidth=2)
    plt.plot(freq, mean_positive, label="Positive Samples (Mean)", color="lightcoral", linewidth=2)

    # 极值范围
    # plt.fill_between(freq, min_negative, max_negative, color="blue", alpha=0.1, label="Negative Samples (Min-Max)")
    # plt.fill_between(freq, min_positive, max_positive, color="red", alpha=0.1, label="Positive Samples (Min-Max)")

    # 置信区间
    # plt.fill_between(freq, ci_neg_lower, ci_neg_upper, color="blue", alpha=0.2, label="Negative Samples (95% CI)")
    # plt.fill_between(freq, ci_pos_lower, ci_pos_upper, color="red", alpha=0.2, label="Positive Samples (95% CI)")

    # 标准差范围
    plt.fill_between(freq, std_neg_lower, std_neg_upper, color="lightblue", alpha=0.6,
                     label="Negative Samples (Mean ± 1 Std)")
    plt.fill_between(freq, std_pos_lower, std_pos_upper, color="lightcoral", alpha=0.6,
                     label="Positive Samples (Mean ± 1 Std)")

    # 图形设置
    plt.xlim(0.2, 1)
    plt.ylim(-30, 75)
    plt.xlabel("Frequency (THz)", fontsize=14)  # 横坐标标签字体
    plt.ylabel("Absorption Coefficient ($\\mathrm{cm^{-1}}$)", fontsize=14)  # 纵坐标标签字体
    plt.title("Absorption Coefficients with Statistical Bands (Smoothed)", fontsize=16)  # 标题字体
    plt.legend(fontsize=12)  # 图注字体
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    # 数据目录和参数
    data_directory = "../data/abc"
    num_negative_samples = 440  # 负样本数量

    # 加载数据
    negative_samples, positive_samples, freq = load_absorption_files(data_directory, num_negative_samples)

    # 绘制平均值曲线及宽带
    plot_mean_absorption_with_bands(freq, negative_samples, positive_samples)
