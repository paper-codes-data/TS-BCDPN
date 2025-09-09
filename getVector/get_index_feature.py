import numpy as np

# 加载特征文件
feature_file = r".\feature_vector\test\test.txt"
X = np.loadtxt(feature_file)

# 提供的特征索引（从0开始编号）
selected_features_idx = [125, 85, 124, 129, 138, 126, 76, 131, 118, 130, 137, 95, 84, 109, 110, 111, 128, 123, 78, 98,
                         114, 117, 92, 115, 116, 127, 93, 86, 96, 132, 135, 136, 113, 119, 120, 122, 134, 133, 107, 103,
                         108, 77, 112, 94, 91, 101, 102, 106, 100, 104, 105, 79, 81, 82, 74, 90, 87, 99, 89, 83, 80, 75,
                         97, 139, 88, 73, 69, 121, 72, 67, 71, 63, 61, 66, 40, 70, 48, 45, 68, 60]

# 根据选出的特征索引对原始特征进行处理，只保留这些特征
X_selected = X[:, selected_features_idx]

# 输出处理后的特征维度
print(f"处理后的特征维度：{X_selected.shape}")

# 保存处理后的特征数据到文件
np.savetxt("./feature_vector/test/mic_test_80.txt", X_selected, fmt="%.6e", delimiter="\t")
