import os
import numpy as np

# 设置输入和输出文件夹路径
input_folder = "../data/frequency_domain_amplitude-ref"
intermediate_folder = "./feature_file/test"
output_file = "./feature_vector/spectra_amplitude-0.2-1.05/feature_matrix-labels.txt"

# 创建中间文件夹和输出文件夹（如果不存在）
if not os.path.exists(intermediate_folder):
    os.makedirs(intermediate_folder)

# 获取输入文件夹中的所有文件
input_files = os.listdir(input_folder)

# 初始化一个列表，用于存储所有文件的振幅数据
amplitude_data_list = []

# 循环处理每个文件
for input_file in input_files:
    # 构建输入文件的完整路径
    input_file_path = os.path.join(input_folder, input_file)

    # 读取文件数据
    data = np.loadtxt(input_file_path)

    # 提取前0.2-1.6tHZ数据
    selected_data = data[20:105, :]
    #
    # 构建中间文件的完整路径
    intermediate_file_path = os.path.join(intermediate_folder, input_file)
    #
    # 保存0.5-1.6tHZ数据到中间文件
    np.savetxt(intermediate_file_path, selected_data, fmt="%.6e", delimiter="\t")
    #
    # 提取振幅数据（第2-5列）
    amplitude_data = selected_data[:, 1:]
    # amplitude_data = data[:, 1:]

    # 将振幅数据转置为特征向量，并添加到列表中
    amplitude_data_list.extend(amplitude_data.T)

# 合并所有特征向量形成矩阵
input_matrix_2 = np.vstack(amplitude_data_list)

# 保存矩阵到输出文件
np.savetxt(output_file, input_matrix_2, fmt="%.6e", delimiter="\t")

print("处理完成！")
