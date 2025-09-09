import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter

# 定义S-G卷积参数
window_size = 51  # 窗口大小，通常选择奇数
order = 3  # 多项式阶数，一般选择小于窗口大小

# 读取txt文件
# 参照物，空气
data_reference = np.genfromtxt('../data/ref-sg-73.6-40/paraffin.txt', delimiter='\t')
time_series_reference = data_reference[:, 0]
amplitudes_reference = data_reference[:, 1]  # 只读取10组实验的平均值


# 小鼠肺切片
# 读取txt文件
def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter='\t')
    time_series = data[:, 0]
    amplitudes = data[:, 1]
    return time_series, amplitudes


# 进行S-G滤波
def smooth_data(amplitudes):
    smoothed_amplitudes = savgol_filter(amplitudes, window_size, order)
    return smoothed_amplitudes


def fourier_transform(time_series, amplitudes):
    """
    绘制频谱振幅和相位图。

    参数：
    - time_series: 时间序列数据
    - amplitudes: 信号的振幅数据
    - label.txt.txt: 图例标签
    """

    # 计算采样率
    sampling_rate = 1 / (time_series[1] - time_series[0])

    # 执行傅立叶变换
    freq = fftfreq(len(time_series), d=(time_series[1] - time_series[0]))
    fft_values = fft(amplitudes)

    # 计算振幅和相位
    magnitude = np.abs(fft_values)
    phase = np.angle(fft_values)
    # phase = (phase * np.pi) / 180

    # 仅保留正频率部分
    freq = freq[:len(freq) // 2]
    magnitude = magnitude[:len(magnitude) // 2]
    phase = phase[:len(phase) // 2]

    """

    N = len(time_series)
    T = (time_series[1] - time_series[0]) * 1e-12  # 采样周期，单位s
    freq = fftfreq(N, T)  # 频率，单位Hz
    fft_result = frequency_domain_amplitude(amplitudes)
    magnitude_17 = np.abs(fft_result)
    phase = np.angle(fft_result)
    freq = freq * 1e-12  # 转换频率为THz
    """

    return freq, magnitude, phase


def calculate_refractive_index(time_series_reference1, amplitudes_reference1, time_series_analyte, amplitudes_analyte,
                               label):
    """
    计算折射率。

    参数：
    - time_series_reference: 参考信号的时间序列数据
    - time_series_analyte: 样品信号的时间序列数据

    返回：
    - n_omega: 折射率数组
    """
    d = 1e-3
    c = 2.99792458e8
    # Fourier变换得到参考信号的频域信息
    freq_reference, amplitude_reference, phase_reference = fourier_transform(time_series_reference1,
                                                                             amplitudes_reference1)

    # Fourier变换得到样品信号的频域信息
    freq_analyte, amplitude_analyte, phase_analyte = fourier_transform(time_series_analyte, amplitudes_analyte)

    # # 检查输入数据是否包含无效值
    # if np.any(np.isnan(phase_reference)) or np.any(np.isnan(phase_analyte)):
    #     raise ValueError("输入数据包含NaN值，请检查输入数据的有效性。")
    # if np.any(np.isinf(phase_reference)) or np.any(np.isinf(phase_analyte)):
    #     raise ValueError("输入数据包含无穷大值，请检查输入数据的有效性。")
    # # 计算折射率
    # omega = np.pi * 2 * freq_reference * 1e12  # 角频率（Hz）
    #
    # # 避免除以零的情况
    # omega_nonzero = np.where(omega != 0, omega, 1)  # 将零值替换为1，避免除以零
    # n_omega = (np.abs((phase_analyte - phase_reference)) * c / (omega_nonzero * d)) + 1
    #
    # # 忽略inf和nan
    # n_omega = np.where(np.logical_or(np.isinf(n_omega), np.isnan(n_omega)), 0, n_omega)

    # 计算折射率
    omega = 2 * np.pi * freq * 1e12

    phi = np.abs(phase_analyte - phase_reference)
    # phi_s = phase_analyte
    # phi_ref = phase_reference
    n_omega = (phi * c) / (omega * d) + 1
    # plt.plot(freq_analyte, n_omega, label.txt.txt=label.txt.txt)
    # print(n_omega)
    return n_omega


def calculate_absorption_coefficient(time_series_reference1, amplitudes_reference1, time_series_analyte,
                                     amplitudes_analyte,
                                     label):
    """
    计算吸收系数。

    参数：
    - time_series_reference: 参考信号的时间序列数据
    - time_series_analyte: 样品信号的时间序列数据
    - d: 样品厚度（单位：米）

    返回：
    - alpha_omega: 吸收系数数组
    """
    d = 0.001
    c = 2.99792458e8

    # Fourier变换得到参考信号的频域信息
    freq_reference, amplitude_reference, phase_reference = fourier_transform(time_series_reference1,
                                                                             amplitudes_reference1)

    # Fourier变换得到样品信号的频域信息
    freq_analyte, amplitude_analyte, phase_analyte = fourier_transform(time_series_analyte, amplitudes_analyte)

    # 计算折射率
    n_omega = calculate_refractive_index(time_series_reference1, amplitudes_reference1, time_series_analyte,
                                         amplitudes_analyte,
                                         label)
    # 计算吸收系数
    rho_omega = np.abs(amplitude_analyte / amplitude_reference)  # ρ(ω)样品信号和参考信号幅值比
    phi_omega = phase_analyte - phase_reference  # φ(ω)样品信号相对参考信号的延迟时间(相位信息)

    # 消光系数
    k = np.log(4 * n_omega / (rho_omega * (n_omega + 1) ** 2)) * c / (freq_analyte * d)
    alpha_omega = 2 * k * freq_analyte / c

    # 单位m^-1转为cm^-1
    alpha_omega = alpha_omega * 1e-2

    # alpha_omega = (2 / d) * np.log((4 * n_omega) / (rho_omega * (n_omega + 1) ** 2))
    # 计算吸光度
    # A = -np.log10(np.abs(amplitude_analyte / amplitude_reference) ** 2)
    # plt.plot(freq_analyte, alpha_omega, label.txt.txt=label.txt.txt)
    # plt.plot(freq_analyte, A, label.txt.txt=label.txt.txt)

    return alpha_omega


# 保存数据
def save_data(file_path, fre_series, data):
    with open(file_path, 'w') as f:
        for i in range(len(fre_series)):
            f.write(f"{fre_series[i]}\t{data[i]}\n")


# 创建新文件夹用于保存结果
output_folder = '../data/expanded_new_time/negative/abc-550'
os.makedirs(output_folder, exist_ok=True)

# 获取所有文件路径
input_folder = '../data/expanded_new_time/negative/550'
file_paths = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".txt"):
            file_paths.append(os.path.join(root, file))

# 使用正则表达式提取样本编号并排序
file_paths.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

# 参考信号的S-G平滑时域信号
smoothed_amplitudes_reference = smooth_data(amplitudes_reference)

# 设置输出图片的尺寸为1920x1080
plt.figure(figsize=(19.20, 10.80), dpi=300)

# 创建空的图例对象
legend_normal = plt.Line2D([], [], color='blue', label='Normal')
legend_cancer = plt.Line2D([], [], color='red', label='Cancer')

# 计数器，用于跟踪处理的文件数量
count = 0
for file_path in file_paths:
    # 读取数据
    time_series, amplitudes = read_data(file_path)

    # S-G滤波
    smoothed_amplitudes = smooth_data(amplitudes)

    # 傅里叶变换，计算频率和幅值
    freq, magnitude, phase = fourier_transform(time_series, amplitudes)

    # 计算吸收系数
    absorption_coefficient = calculate_absorption_coefficient(time_series_reference,
                                                              smoothed_amplitudes_reference, time_series,
                                                              smoothed_amplitudes,
                                                              os.path.basename(file_path))
    # 计算折射率
    # refractive_index = calculate_refractive_index(time_series_reference,
    #                                               amplitudes_reference, time_series, amplitudes,
    #                                               os.path.basename(file_path))
    # 计算吸收系数,S-G平滑后的时域信号
    # absorption_coefficient = calculate_absorption_coefficient(time_series_reference,
    #                                                           smoothed_amplitudes_reference, time_series, smoothed_amplitudes,
    #                                                           os.path.basename(file_path))
    # 保存数据
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    save_data(output_file_path, freq, absorption_coefficient)

    # 绘图
    # 根据计数器的值选择要绘制的颜色，并添加相应的标签
    if count < 440:
        plt.plot(freq, absorption_coefficient, color='blue', label='Normal' if count == 0 else '',
                 linewidth=0.5)
    else:
        plt.plot(freq, absorption_coefficient, color='red', label='Cancer' if count == 440 else '', linewidth=0.5)

    # 增加计数器的值
    count += 1

# 添加图例
plt.legend(handles=[legend_normal, legend_cancer])

# plt.xlabel('Frequency (THz)')
# plt.ylabel('refractive index')
# plt.title('Frequency-Refractive Index')
# plt.xlim(0.2, 1.6)
# plt.legend(loc='upper left')
# plt.grid(True)


plt.xlabel('Frequency (THz)')
plt.ylabel('Absorption Coefficient (cm^-1)')
plt.title('Frequency-Absorption Coefficient')
plt.xlim(0.2, 1.6)
plt.legend(loc='upper left')
plt.grid(True)

# plt.savefig('Mouse_breast_cancer_time_removed_refractive_index.svg', format='svg')
plt.tight_layout()
plt.show()
