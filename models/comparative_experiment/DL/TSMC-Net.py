import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, f1_score, \
    recall_score


# 多尺度特征提取模块
# 定义PyTorch模型
class UATCDenseNet(nn.Module):
    def __init__(self, input_length, num_classes):
        super(UATCDenseNet, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1)  # Conv1D Layer
        self.pool = nn.AvgPool1d(kernel_size=2)  # Average Pooling Layer
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1)  # Conv1D Layer

        # 计算池化后的尺寸
        # 假设输入是 (batch_size, 1, input_length)，经过两次池化后，尺寸变为 (batch_size, 3, input_length//4)
        self.flatten = nn.Flatten()  # Flatten layer
        self.fc1 = nn.Linear(3 * (input_length // 4), 48)  # Fully connected layer after flattening
        self.fc2 = nn.Linear(48, num_classes)  # Output layer (根据传入的类别数)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 数据加载函数
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    labels = data[:, -1]
    data = data[:, :-1]

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, labels


# 训练与验证函数
# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=20, save_path="UATC-DenseNet.csv"):
    best_auc = 0.0  # 记录最佳AUC
    best_acc = 0.0  # 记录最佳AUC
    best_results = None  # 保存最佳测试集结果 (标签、预测标签、预测概率)
    best_test_metrics = {}  # 保存最佳测试集的评价指标

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # 测试阶段
        model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类的概率值
                test_probs.extend(probs.cpu().numpy())
                test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # 计算测试集指标
        test_acc = accuracy_score(test_labels, test_preds)
        auc = roc_auc_score(test_labels, test_probs)
        aupr = average_precision_score(test_labels, test_probs)
        precision = precision_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | AUC: {auc:.4f} | AUPR: {aupr:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # 更新最佳结果
        if test_acc > best_acc:
            best_auc = auc
            best_results = (test_labels, test_preds, test_probs)  # 保存当前最佳结果
            best_test_metrics = {
                'testAcc': test_acc,
                'AUC': auc,
                'AUPR': aupr,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
            }

    # 保存最佳结果到CSV文件
    if best_results:
        best_labels, best_preds, best_probs = best_results
        results_df = pd.DataFrame({
            '真实标签': best_labels,
            '预测标签': best_preds,
            '预测概率': best_probs
        })
        results_df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"最优测试集结果已保存到 {save_path}！")

    # 打印最优测试集的指标
    if best_test_metrics:
        print("最优测试集指标：")
        print(f"Test Acc: {best_test_metrics['testAcc']:.4f} | AUC: {best_test_metrics['AUC']:.4f} | "
              f"AUPR: {best_test_metrics['AUPR']:.4f} | Precision: {best_test_metrics['Precision']:.4f} | Recall: {best_test_metrics['recall']:.4f} |"
              f"F1: {best_test_metrics['F1']:.4f}")


if __name__ == "__main__":
    # file_path =  "../../../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt"
    file_path = "../../../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt"
    data, labels = load_data(file_path)
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_tensor = TensorDataset(torch.tensor(train_data, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(train_labels, dtype=torch.long))
    val_tensor = TensorDataset(torch.tensor(val_data, dtype=torch.float32).unsqueeze(1),
                               torch.tensor(val_labels, dtype=torch.long))

    train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UATCDenseNet(input_length=train_data.shape[1], num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练模型...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=300)
