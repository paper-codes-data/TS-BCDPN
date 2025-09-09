import os
import re

import numpy as np
import matplotlib.pyplot as plt

# 输入和输出文件夹路径
input_folder = '../data/ref-sg'  # 实际输入文件夹路径
output_folder = '../data/frequency_domain_amplitude-air/'  # 将输出路径设为文件夹路径

# 检查输出文件夹是否存在，不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有文件名
input_files = os.listdir(input_folder)

# 使用正则表达式提取样本编号并排序
input_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

# 初始化绘图
plt.figure(figsize=(19.2, 10.8), dpi=300)

# 循环处理每个文件
for i, input_file in enumerate(input_files):
    # 构建输入文件的完整路径
    input_file_path = os.path.join(input_folder, input_file)

    # 读取时域数据
    data = np.loadtxt(input_file_path)

    # 获取时间和振幅列
    time_series = data[:, 0]
    amplitude_column_2 = data[:, 1]  # 选择第二列振幅进行傅里叶变换

    # 计算采样率
    sampling_rate = 1 / (time_series[1] - time_series[0])  # 假设时间间隔是均匀的

    # 执行傅里叶变换，获取频率和线性幅值
    frequency = np.fft.fftfreq(len(time_series), d=1 / sampling_rate)
    magnitude = np.abs(np.fft.fft(amplitude_column_2))

    # 转换成dB幅值
    magnitude_db = 20 * np.log10(magnitude + np.finfo(float).eps)  # 避免对数计算中的零值

    # 只保留正频谱部分
    frequency = frequency[:len(frequency) // 2]
    magnitude_db = magnitude_db[:len(magnitude_db) // 2]

    # 保存频域数据到新文件
    output_file_path = os.path.join(output_folder, input_file.replace('.txt', '.txt'))
    output_data = np.column_stack((frequency, magnitude_db))  # 使用 dB 幅值
    np.savetxt(output_file_path, output_data, fmt="%.6e", delimiter="\t")

    # 将当前文件的频谱绘制在同一张图上
    plt.plot(frequency, magnitude_db, label=f"{input_file}")

    print(f"Processed {input_file}, saved frequency domain data to {output_file_path}")

# 设置图例和标签
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0.2, 1.8)
plt.title("Frequency Spectrum of All Files")
plt.legend(loc="upper right", fontsize="small")
plt.grid(True)

# 显示绘图
plt.show()
