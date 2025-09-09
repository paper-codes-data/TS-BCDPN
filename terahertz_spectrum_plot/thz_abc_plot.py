import os
import matplotlib.pyplot as plt
import numpy as np
import re

# 文件夹路径
input_folder = '../data/wgan_data/expanded_time/time_to_abc/abc_200'

# 获取文件夹中所有文件路径，并按照样本编号排序
file_paths = [
    os.path.join(input_folder, filename)
    for filename in os.listdir(input_folder)
    if filename.endswith('.txt')
]

# 使用正则表达式提取样本编号并排序
file_paths.sort(key=lambda x: int(re.search(r'_(\d+)', os.path.basename(x)).group(1)))

# 设置输出图片的尺寸为1920x1080
plt.figure(figsize=(19.20, 10.80), dpi=300)

count = 0
# 遍历每个文件
for file_path in file_paths:
    # 读取数据，假设文件中有两列数据
    data = np.loadtxt(file_path)  # 假设文件中的数据可以用np.loadtxt来读取
    freq = data[:, 0]  # 第一列作为横坐标
    absorption_coefficient = data[:, 1]  # 第二列作为纵坐标

    # 绘图，并在每条线上加上文件名作为标注
    if count < 100:
        plt.plot(freq, absorption_coefficient, color='blue', label='Normal' if count == 1 else "")
    else:
        plt.plot(freq, absorption_coefficient, color='red', label='Cancer' if count == 101 else "")
    count += 1
# 添加图例
plt.legend(loc='upper right', fontsize='small')

plt.xlabel('Thz')
plt.ylabel('Absorption Coefficient')
plt.title('Absorption Coefficient of Mouse Lung Cancer Samples')
plt.xlim(0.2, 1.6)
plt.grid(True)
plt.tight_layout()
plt.show()
