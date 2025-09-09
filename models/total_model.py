import torch
import torch.nn as nn
from feature_extractors import MultiScaleConvFeatureExtractor, LSTMFeatureExtractor, FeatureFusion


class TotalModel(nn.Module):
    def __init__(self, time_dim, freq_dim, fusion_dim, num_classes):
        super(TotalModel, self).__init__()
        self.time_extractor = LSTMFeatureExtractor(input_dim=time_dim, output_dim=fusion_dim // 2)
        self.freq_extractor = MultiScaleConvFeatureExtractor(input_dim=freq_dim, output_dim=fusion_dim // 2)
        self.fusion = FeatureFusion(time_dim=fusion_dim // 2, freq_dim=fusion_dim // 2, fusion_dim=fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, time_data, freq_data):
        time_features = self.time_extractor(time_data)
        freq_features = self.freq_extractor(freq_data.unsqueeze(1))  # 频域数据需要增加通道维度
        fused_features = self.fusion(time_features, freq_features)
        logits = self.classifier(fused_features)
        return logits
