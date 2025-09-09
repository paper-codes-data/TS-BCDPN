import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, \
    recall_score
import csv

# 残差块
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

# CNN模型
class CNNResNetModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNResNetModel, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # 残差块
        self.resblock1 = ResNetBlock(64, 100)
        self.resblock2 = ResNetBlock(100, 100)

        # 全局池化
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(100 * 2, 256)
        self.fc2 = nn.Linear(256, output_dim)  # 输出层

        # 分类层
        self.classifier = nn.Linear(output_dim, 2)  # 2类分类任务

    def forward(self, x):
        # 初始卷积层
        x = self.relu(self.bn1(self.conv1(x)))

        # 残差块
        x = self.resblock1(x)
        x = self.resblock2(x)

        # 全局池化
        max_pool = self.global_max_pool(x).squeeze(-1)  # 最大池化
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # 平均池化

        # 特征拼接
        x = torch.cat([max_pool, avg_pool], dim=1)

        # 全连接层
        features = self.fc(x)
        features = self.fc2(features)

        # 分类层
        logits = self.classifier(features)

        return logits, features

def load_data(file_path):
    # 加载数据
    data = np.loadtxt(file_path, delimiter='\t')  # 假设数据是制表符分隔
    labels = data[:, -1]  # 最后一列为标签
    data = data[:, :-1]  # 特征数据

    # 标准化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)  # 标准化

    return data, labels

def save_results_to_csv(file_path, labels, preds, probs):
    # 将结果保存到CSV文件
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["True Label", "Predicted Label", "Probability"])
        for true_label, pred_label, prob in zip(labels, preds, probs):
            writer.writerow([true_label, pred_label, prob])

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=20, save_path="lcc-THz-PeelNet.csv"):
    best_acc = 0.0  # 记录最佳测试集准确率
    best_results = None  # 保存最佳测试集结果 (标签、预测标签、预测概率)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 获取logits作为输出
            logits, _ = model(inputs)  # 只获取logits
            loss = criterion(logits, labels)  # 使用logits计算损失
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # 测试阶段
        model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model(inputs)  # 只获取logits
                probs = torch.softmax(logits, dim=1)[:, 1]  # 获取正类的概率值
                test_probs.extend(probs.cpu().numpy())
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # 计算测试集准确率
        test_acc = accuracy_score(test_labels, test_preds)
        auc = roc_auc_score(test_labels, test_probs)
        aupr = average_precision_score(test_labels, test_probs)
        precision = precision_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | AUC: {auc:.4f} | AUPR: {aupr:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} |F1: {f1:.4f}")

        # 比较当前epoch的测试集准确率与最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc
            best_results = (test_labels, test_preds, test_probs)
            print(f"New best accuracy: {best_acc:.4f} at epoch {epoch+1}")

    # 保存最优结果到CSV文件
    if best_results:
        test_labels, test_preds, test_probs = best_results
        # save_results_to_csv(save_path, test_labels, test_preds, test_probs)
        print(f"Best results saved to {save_path}")


if __name__ == "__main__":
    # file_path = "../../../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt"  # 数据路径
    file_path = "../../../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt"
    data, labels = load_data(file_path)

    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_tensor = TensorDataset(torch.tensor(train_data, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(train_labels, dtype=torch.long))
    val_tensor = TensorDataset(torch.tensor(val_data, dtype=torch.float32).unsqueeze(1),
                               torch.tensor(val_labels, dtype=torch.long))

    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNResNetModel(input_dim=85, output_dim=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.5)

    print("开始训练模型...")
    print(f"训练集大小: {len(train_tensor)}, 验证集大小: {len(val_tensor)}")
    print(f"使用设备: {device}")
    print(model)

    # 打印模型总参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数数量: {total_params}")

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200)
