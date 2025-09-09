import os
import numpy as np

# 定义路径
sample_dir = "../data/fft"
air_ref_dir = "../data/frequency_domain_amplitude-air"
output_dir = "../data/frequency_domain_amplitude-ref"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取参考信号
air_ref_file = os.path.join(air_ref_dir, "paraffin.txt")  # 假设空气信号文件名为 "paraffin.txt"
air_data = np.loadtxt(air_ref_file)  # 读取参考文件
air_freq = air_data[:, 0]  # 参考信号频域
air_amplitude = air_data[:, 1]  # 参考信号振幅

# 获取样品文件并排序
sample_files = sorted(
    [f for f in os.listdir(sample_dir) if f.endswith(".txt")],
    key=lambda x: int(x.split(".")[0])  # 按文件名中的数字排序
)

# 遍历样品文件
for sample_file in sample_files:
    sample_path = os.path.join(sample_dir, sample_file)
    output_path = os.path.join(output_dir, sample_file)

    # 读取样品文件
    sample_data = np.loadtxt(sample_path)
    sample_freq = sample_data[:, 0]  # 样品信号频域
    sample_amplitude = sample_data[:, 1]  # 样品信号振幅

    # 确保频域一致
    if not np.array_equal(sample_freq, air_freq):
        raise ValueError(f"频域不匹配: {sample_file}")

    # 计算归一化振幅
    normalized_amplitude = sample_amplitude / air_amplitude

    # 保存结果
    np.savetxt(output_path, np.column_stack((sample_freq, normalized_amplitude)), fmt="%.6f", delimiter="\t")

print("所有文件已成功处理并保存到 data/frequency_domain_amplitude-ref 目录。")
