import torch
import torch.nn as nn


class MultiScaleConvFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiScaleConvFeatureExtractor, self).__init__()

        # 多尺度卷积层
        # self.conv3 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)  # 卷积核大小为3
        self.conv5 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)  # 卷积核大小为5
        self.conv7 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)  # 卷积核大小为7
        self.bn = nn.BatchNorm1d(128)  # 拼接后的通道数 32*3
        self.relu = nn.ReLU()

        # Dropout 层
        self.dropout = nn.Dropout(p=0.2)  # 添加 Dropout 层

        # 残差块
        self.resblock1 = ResNetBlock(128, 100)
        self.resblock2 = ResNetBlock(100, 100)

        # 全局池化
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(100 * 2, 256)  # 拼接后 256*2
        self.fc2 = nn.Linear(256, output_dim)

        # 分类器
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, x):
        # 多尺度卷积层
        # x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        # 拼接多尺度卷积的输出
        x = torch.cat([x5, x7], dim=1)
        x = self.bn(x)
        x = self.relu(x)

        # Dropout 层
        x = self.dropout(x)

        # 残差块
        x = self.resblock1(x)
        x = self.resblock2(x)

        # 全局池化
        max_pool = self.global_max_pool(x).squeeze(-1)  # 全局最大池化
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # 全局平均池化

        # 特征拼接
        x = torch.cat([max_pool, avg_pool], dim=1)

        # 全连接层
        features = self.fc(x)
        features = self.fc2(features)

        # 分类
        # logits = self.classifier(features)

        return features


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class LSTMFeatureExtractor(nn.Module):
    # 保留原有代码实现
    def __init__(self, input_dim, hidden_dim=256, output_dim=64, num_layers=2, dropout_rate=0.6):
        super(LSTMFeatureExtractor, self).__init__()

        # 定义 LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        # 定义全连接层，将 LSTM 的输出映射到最终特征空间
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入 x 的形状：[batch_size, seq_length, input_dim]
        # seq_length=2，input_dim=特征维度（如 1008）

        # 通过 LSTM 层 (只取最后一个时间步的输出),前面的时间步为最后一个时间步提供依赖关系，仅取最后一个时间步作为样本特征
        lstm_out, _ = self.lstm(x)  # 输出的 lstm_out 形状：[batch_size, seq_length, hidden_dim]
        # lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出，形状：[batch_size, hidden_dim]
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出，形状：[batch_size, hidden_dim]

        # 通过全连接层映射到输出维度
        fc1_out = self.fc1(lstm_out)

        # 激活
        fc1_out = self.relu(fc1_out)

        fc2_out = self.fc2(fc1_out)

        return fc2_out


"""
不同融合方式
"""


# class FeatureFusion(nn.Module):  # 拼接
#     def __init__(self, time_dim, freq_dim, fusion_dim):
#         super(FeatureFusion, self).__init__()
#         self.fc = nn.Linear(time_dim + freq_dim, fusion_dim)
#         # Dropout 层
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU()
#
#     def forward(self, time_features, freq_features):
#         fused_features = torch.cat([time_features, freq_features], dim=1)
#         fused_features = self.dropout(fused_features)
#         return self.relu(self.fc(fused_features))


# class FeatureFusion(nn.Module):  # 自注意力机制
#     def __init__(self, time_dim, freq_dim, fusion_dim, attention_heads=4):
#         super(FeatureFusion, self).__init__()
#         self.fc = nn.Linear(time_dim + freq_dim, fusion_dim)
#         self.relu = nn.ReLU()
#
#         # 自注意力层
#         self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=attention_heads, batch_first=True)
#
#         # Dropout 层
#         self.dropout = nn.Dropout(p=0.2)
#
#         # 最终的全连接层
#         self.final_fc = nn.Linear(fusion_dim, fusion_dim)
#
#     def forward(self, time_features, freq_features):
#         # print(time_features.shape)
#         # print(freq_features.shape)
#         # 特征拼接
#         fused_features = torch.cat([time_features, freq_features], dim=1)
#
#         # 初步映射到融合维度
#         fused_features = self.relu(self.fc(fused_features))
#
#         # 添加 Dropout
#         fused_features = self.dropout(fused_features)
#
#         # 添加维度以适配自注意力机制 (需要形状为 [batch_size, seq_len, feature_dim])
#         fused_features = fused_features.unsqueeze(1)  # 添加序列维度 seq_len=1
#
#         # 自注意力机制
#         attention_out, _ = self.attention(fused_features, fused_features, fused_features)
#
#         # 去掉序列维度
#         attention_out = attention_out.squeeze(1)
#
#         # 添加 Dropout
#         attention_out = self.dropout(attention_out)
#
#         # 最终全连接映射
#         output = self.final_fc(attention_out)
#         return output

# class FeatureFusion(nn.Module):  # 注意力
#     def __init__(self, time_dim, freq_dim, fusion_dim, attention_heads=4):
#         super(FeatureFusion, self).__init__()
#         self.feature_dim = time_dim  # 假设 time_dim 和 freq_dim 相同
#         self.attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=attention_heads, batch_first=True)
#
#         # 映射到最终融合维度
#         self.fc = nn.Linear(self.feature_dim, fusion_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, time_features, freq_features):
#         # 堆叠时间域和频率域特征，形成注意力输入
#         fused_input = torch.stack([time_features, freq_features], dim=1)  # [batch_size, 2, feature_dim]
#
#         # 注意力机制融合
#         attention_out, _ = self.attention(fused_input, fused_input, fused_input)  # [batch_size, 2, feature_dim]
#
#         # 平均池化融合两个特征
#         pooled_features = attention_out.mean(dim=1)  # [batch_size, feature_dim]
#
#         # print(pooled_features.size())
#
#         # 映射到最终融合维度
#         output = self.relu(self.fc(pooled_features))  # [batch_size, fusion_dim]
#         # print(output.size())
#         return output


# class FeatureFusion(nn.Module):  # Hadamard积融合
#     def __init__(self, time_dim, freq_dim, fusion_dim):
#         super(FeatureFusion, self).__init__()
#         assert time_dim == freq_dim, "Time and frequency dimensions must be the same for Hadamard product fusion."
#
#         self.feature_dim = time_dim  # 保留 time_dim 和 freq_dim
#         self.fc = nn.Linear(self.feature_dim, fusion_dim)  # 将 Hadamard 积结果映射到目标融合维度
#         self.relu = nn.ReLU()
#
#         # Dropout 层
#         self.dropout = nn.Dropout(p=0.2)
#
#         # 最终的全连接层
#         self.final_fc = nn.Linear(fusion_dim, fusion_dim)
#
#     def forward(self, time_features, freq_features):
#         # 确保 time_features 和 freq_features 的维度一致
#         assert time_features.size() == freq_features.size(), "Time and frequency features must have the same shape."
#
#         # Hadamard积融合（逐元素相乘）
#         fused_features = time_features * freq_features  # [batch_size, feature_dim]
#
#         # 将融合后的特征映射到目标融合维度
#         fused_features = self.relu(self.fc(fused_features))
#
#         # 添加 Dropout
#         fused_features = self.dropout(fused_features)
#
#         # 最终全连接映射
#         output = self.final_fc(fused_features)
#         return output


class FeatureFusion(nn.Module):  # 均值融合
    def __init__(self, time_dim, freq_dim, fusion_dim):
        super(FeatureFusion, self).__init__()
        assert time_dim == freq_dim, "Time and frequency dimensions must be the same for mean fusion."

        self.feature_dim = time_dim  # 假设 time_dim 和 freq_dim 相等
        self.fc = nn.Linear(self.feature_dim, fusion_dim)  # 将均值融合后的特征映射到目标融合维度
        self.relu = nn.ReLU()

        # Dropout 层
        self.dropout = nn.Dropout(p=0.5)

        # 最终的全连接层
        self.final_fc = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, time_features, freq_features):
        # 确保 time_features 和 freq_features 的维度一致
        assert time_features.size() == freq_features.size(), "Time and frequency features must have the same shape."

        # 直接求均值融合
        fused_features = (time_features + freq_features) / 2  # [batch_size, feature_dim]

        # 将融合后的特征映射到目标融合维度
        fused_features = self.relu(self.fc(fused_features))

        # 添加 Dropout
        fused_features = self.dropout(fused_features)

        # 最终全连接映射
        output = self.final_fc(fused_features)
        return output



