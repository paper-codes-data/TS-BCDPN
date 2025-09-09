import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter

# 设定文件路径和文件范围
directory = '../data/Mouse_breast_time-domain_spectroscopy'  # 路径
start_index = 820  # 开始绘制的文件索引
end_index = 850  # 结束绘制的文件索引

# 正则表达式匹配文件名，按序号1，2，3...读取
pattern = re.compile(r'(\d+)\.txt')  # 匹配形如 'AIR.txt', '2.txt' 的文件

# 获取文件列表并按正则表达式中的数字顺序排序
files = [f for f in os.listdir(directory) if pattern.match(f)]
files.sort(key=lambda x: int(pattern.match(x).group(1)))


# 小波去噪和S-G滤波的函数
def smooth_data(amplitudes, wavelet='db6', level=4, threshold_method='soft', window_size=31, order=3):
    # 小波去噪
    coeffs = pywt.wavedec(amplitudes, wavelet, level=level)
    threshold = lambda x: pywt.threshold(x, value=np.median(np.abs(x)) / 0.6745, mode=threshold_method)
    denoised_coeffs = [threshold(c) if i > 0 else c for i, c in enumerate(coeffs)]
    denoised_amplitudes = pywt.waverec(denoised_coeffs, wavelet)

    # S-G滤波
    smoothed_amplitudes = savgol_filter(denoised_amplitudes, window_size, order)

    return smoothed_amplitudes


# 设置输出图片的尺寸为1920x1080
plt.figure(figsize=(19.20, 10.80), dpi=300)

# 遍历文件，绘制原始和去噪后的曲线
for i in range(start_index, min(end_index, len(files))):
    file_path = os.path.join(directory, files[i])
    data = np.loadtxt(file_path)  # 假设文件是两列，第一列是时间，第二列是振幅
    time = data[:, 0]  # 时间数据
    amplitude = data[:, 1]  # 振幅数据

    # 调用平滑函数
    smoothed_amplitude = smooth_data(amplitude)

    # 绘制原始数据曲线
    plt.plot(time, amplitude, label=f'Original {files[i]}', linewidth=0.5)

    # 绘制去噪后的数据曲线
    # plt.plot(time, smoothed_amplitude, label=f'Smoothed {files[i]}', linewidth=1.0)

# 绘制设置
plt.title('Comparison of Original and Smoothed Spectra')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
