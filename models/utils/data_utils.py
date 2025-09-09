import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def sliding_window(data, labels, window_size=2, step_size=1):
    """
    使用滑动窗口重新采样数据。

    参数:
        data (np.ndarray): 原始特征数据，形状为 (样本数, 特征数)。
        labels (np.ndarray): 原始标签数据，形状为 (样本数, )。
        window_size (int): 窗口大小。
        step_size (int): 滑动步长。

    返回:
        X (np.ndarray): 重新采样后的特征数据，形状为 (新样本数, 窗口大小, 特征数)。
        Y (np.ndarray): 对应的标签，形状为 (新样本数, )。
    """
    X, Y = [], []
    for i in range(0, len(data) - window_size + 1, step_size):
        X.append(data[i:i + window_size, :])  # 提取窗口内的所有特征
        Y.append(labels[i + window_size - 1])  # 使用窗口末尾对应的标签
    return np.array(X), np.array(Y)

def shuffle_data(features, labels, num_shuffles=100, random_seed=42):
    """
    随机打乱数据多次，并保证每次运行时打乱顺序一致。
    """
    np.random.seed(random_seed)
    for _ in range(num_shuffles):
        indices = np.random.permutation(len(features))  # 生成随机排列的索引
        features = features[indices]
        labels = labels[indices]
    return features, labels

def load_and_preprocess_data(time_path, freq_path, test_size=0.2, batch_size=64):
    """
    加载并预处理时域和频域数据文件。

    参数:
        time_path (str): 时域数据文件路径
        freq_path (str): 频域数据文件路径
        test_size (float): 测试集占比
        batch_size (int): DataLoader 的批量大小

    返回:
        train_loader: 训练集的 DataLoader
        test_loader: 测试集的 DataLoader
    """
    # 加载数据
    time_data = np.loadtxt(time_path, delimiter='\t')
    freq_data = np.loadtxt(freq_path, delimiter='\t')

    # 划分特征和标签
    time_labels = time_data[:, -1]  # 时域数据最后一列为标签
    time_features = time_data[:, :-1]  # 时域数据去除最后一列，保留特征

    freq_labels = freq_data[:, -1]  # 频域数据最后一列为标签
    freq_features = freq_data[:, :-1]  # 频域数据去除最后一列，保留特征

    # 标准化特征
    scaler_time = StandardScaler()
    scaler_freq = StandardScaler()

    time_features = scaler_time.fit_transform(time_features)
    freq_features = scaler_freq.fit_transform(freq_features)

    # 使用滑动窗口重新采样时域数据
    time_features, time_labels = sliding_window(time_features, time_labels, window_size=2, step_size=1)

    # 打乱数据
    time_features, time_labels = shuffle_data(time_features, time_labels)
    freq_features, freq_labels = shuffle_data(freq_features, freq_labels)

    # 调整数据形状以适应 LSTM 输入
    num_samples, window_size, num_features = time_features.shape
    time_features = time_features.reshape(num_samples, window_size, num_features)

    print(time_features.shape)

    # 划分数据集
    time_train, time_test, freq_train, freq_test, labels_train, labels_test = train_test_split(
        time_features, freq_features, time_labels, test_size=test_size, random_state=42, stratify=time_labels
    )

    # 创建 TensorDataset
    train_dataset = TensorDataset(
        torch.tensor(time_train, dtype=torch.float32),
        torch.tensor(freq_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(time_test, dtype=torch.float32),
        torch.tensor(freq_test, dtype=torch.float32),
        torch.tensor(labels_test, dtype=torch.long)
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# def main():
#     train_loader, test_loader = load_and_preprocess_data("../../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt",
#                                                          "../../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt")
#
#
# if __name__ == "__main__":
#     main()