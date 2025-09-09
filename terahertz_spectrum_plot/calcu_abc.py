import os
import numpy as np
import matplotlib.pyplot as plt

# 样品厚度和光速
d = 1e-3  # 厚度，单位m
c = 2.99792458e8  # 光速，单位m/s

# 频率范围（0.2 - 1.6 THz）
freq = np.arange(0.2, 1.6, 0.01)  # 间隔 0.01 THz

# 读取参考振幅文件
ref_file_path = '../data/wgan_data/expanded_freq/ref/ref_freq_domain.txt'
amplitudes_reference = np.loadtxt(ref_file_path)

# 计算折射率
def calculate_refractive_index(amp_ref, amp_sample, phase_ref, phase_sample):
    omega = 2 * np.pi * freq * 1e12  # 角频率
    phi = np.abs(phase_sample - phase_ref)
    n_omega = (phi * c) / (omega * d) + 1
    return n_omega

# 计算吸收系数
def calculate_absorption_coefficient(amp_ref, amp_sample, n_omega):
    rho_omega = np.abs(amp_sample / amp_ref)
    k = np.log(4 * n_omega / (rho_omega * (n_omega + 1) ** 2)) * c / (freq * d * 1e12)
    alpha_omega = 2 * k * freq * 1e12 / c
    return alpha_omega * 1e-2  # 转换为 cm^-1

# 创建输出文件夹
output_folder = 'data/wgan_data/expanded_freq/positive/absorptionCoefficient'
os.makedirs(output_folder, exist_ok=True)

# 处理样品文件
input_folder = 'data/wgan_data/expanded_freq/positive/magnitude_17'
for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):
        sample_file_path = os.path.join(input_folder, file_name)
        amplitudes_sample = np.loadtxt(sample_file_path)

        # 计算相位信息（这里假设 phase_ref 和 phase_sample 是预先计算得到的）
        # 这里可以通过你的特定方法获取 phase_ref 和 phase_sample
        phase_reference = np.zeros_like(freq)  # 示例数据
        phase_sample = np.zeros_like(freq)  # 示例数据

        # 计算折射率和吸收系数
        n_omega = calculate_refractive_index(amplitudes_reference, amplitudes_sample, phase_reference, phase_sample)
        alpha_omega = calculate_absorption_coefficient(amplitudes_reference, amplitudes_sample, n_omega)

        # 保存结果
        output_file_path = os.path.join(output_folder, file_name + '_abc.txt')
        with open(output_file_path, 'w') as f:
            for f_val, a_val in zip(freq, alpha_omega):
                f.write(f"{f_val}\t{a_val}\n")

# 可视化结果
plt.figure(figsize=(10, 6))
for file_name in os.listdir(output_folder):
    if file_name.endswith("_abc.txt"):
        data = np.loadtxt(os.path.join(output_folder, file_name))
        plt.plot(data[:, 0], data[:, 1], label=file_name.replace('_abc.txt', ''))
plt.xlabel('Frequency (THz)')
plt.ylabel('Absorption Coefficient (cm^-1)')
plt.title('Frequency-Absorption Coefficient')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
