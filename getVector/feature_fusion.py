import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 读取特征文件
file1 = './feature_vector/absorption_coefficient/mic_selected_features_40.txt'
file2 = './feature_vector/spectra_amplitude-0.2-1.6/mi_selected_features_40.txt'
file3 = './feature_fusion/time_features_output.txt'

features1 = np.loadtxt(file1)
features2 = np.loadtxt(file2)
features3 = np.loadtxt(file3)

# 归一化特征数据
# scaler = MinMaxScaler()
# # X_scaled = scaler.fit_transform(X)
#
# # 标准化数据
# # scaler = StandardScaler()
# features1 = scaler.fit_transform(features1)
# features2 = scaler.fit_transform(features2)

# 拼接
concatenated_features = np.hstack((features1, features2, features3))
np.savetxt('./feature_fusion/FAC+FFDA+FTD.txt', concatenated_features, delimiter='\t')
print(f"Concatenated Features Shape: {concatenated_features.shape}")

# 根据AUC加权平均
# AUC1 = 0.9294
# AUC2 = 0.9354
# weighted_average_features = (features1 * AUC1 + features2 * AUC2) / (AUC1 + AUC2)
# np.savetxt('./feature_fusion/weighted_average_features.txt', weighted_average_features, delimiter='\t')
# print(f"Weighted Average Features Shape: {weighted_average_features.shape}")

# # 取平均值
# averaged_features = (features1 + features2) / 2
# np.savetxt('./feature_fusion/averaged_features.txt', averaged_features)
# print(f"Averaged Features Shape: {averaged_features.shape}")
#
# # 哈达玛积
# hadamard_features = features1 * features2
# np.savetxt('./feature_fusion/hadamard_features.txt', hadamard_features, delimiter='\t')
# print(f"Hadamard Features Shape: {hadamard_features.shape}")
#
# # 加法
# addition_features = features1 + features2
# np.savetxt('./feature_fusion/addition_features.txt', addition_features, delimiter='\t')
# print(f"Addition Features Shape: {addition_features.shape}")

print("特征融合已完成，并保存到文件中。")
