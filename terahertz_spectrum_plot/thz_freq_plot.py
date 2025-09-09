import os
import numpy as np
import matplotlib.pyplot as plt

# 设置文件夹路径
data_path = "../data/fft"

# 文件范围设置
start_idx = 430  # 从第几个文件开始（包括）
end_idx = 445   # 到第几个文件结束（包括）

# 获取所有按编号排序的文件列表
file_list = sorted([f for f in os.listdir(data_path) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))

# 筛选指定范围的文件
file_list = file_list[start_idx - 1:end_idx]  # 索引从0开始，因此需要减1

# 初始化图像
plt.figure(figsize=(19.2, 10.8), dpi=300)

# 遍历筛选后的文件并绘图
for file_name in file_list:
    file_path = os.path.join(data_path, file_name)
    # 读取文件
    data = np.loadtxt(file_path)
    frequency = data[:, 0]  # 第一列为频率
    amplitude = data[:, 1]  # 第二列为振幅

    # 绘制曲线
    plt.plot(frequency, amplitude, label=file_name)

# 设置图例、标题和坐标轴
plt.title("Terahertz Frequency Domain Amplitude", fontsize=14)
plt.xlabel("Frequency (THz)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.xlim(0.2, 1.6)
plt.legend(title="Files", fontsize=10)
plt.grid(True)
plt.tight_layout()

# 显示图像
plt.show()
