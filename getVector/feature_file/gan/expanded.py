import os
import numpy as np

# 源文件目录路径
source_folder = 'gan-time-domain_10-40ps-0/original_data'

# 需要生成的模拟样本数
num_samples = 10

# 随机调整范围
adjustment_range = (1, 2)

# 创建用于存储生成样本的目录
output_folder = './gan-time-domain_10-40ps-0/expanded_data/normal'
os.makedirs(output_folder, exist_ok=True)

# 获取源文件目录下的所有文件
file_list = os.listdir(source_folder)

# 遍历每个文件
for source_file in file_list:
    # 检查文件扩展名是否为.txt，以确保处理的是文本文件
    if source_file.endswith('.txt'):
        # 构建源文件的完整路径
        source_file_path = os.path.join(source_folder, source_file)

        # 加载数据
        data = np.loadtxt(source_file_path)
        frequencies = data[:, 0]
        absorption_coefficients = data[:, 1]

        # 生成新样本
        for i in range(num_samples):
            # 对每个频率点进行随机调整
            random_adjustment = np.random.uniform(adjustment_range[0], adjustment_range[1], size=len(absorption_coefficients))
            random_sign = np.random.choice([-1, 1], size=len(absorption_coefficients))  # 随机选择+1或-1

            # 创建具有随机调整的新吸收系数（随机加或减）
            new_absorption_coefficients = absorption_coefficients + random_sign * random_adjustment

            # 新文件路径
            new_file_name = f'{os.path.splitext(source_file)[0]}_generated_{i + 1}.txt'
            new_file_path = os.path.join(output_folder, new_file_name)

            # 将新数据保存到文件
            np.savetxt(new_file_path, np.column_stack((frequencies, new_absorption_coefficients)), fmt='%.6f', delimiter='\t')

            print(f'生成样本 {i + 1} from {source_file}: {new_file_path}')

print('生成完成。')
