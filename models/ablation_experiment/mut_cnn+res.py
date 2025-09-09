import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, f1_score, \
    recall_score


# 多尺度卷积特征提取模块
class MultiScaleConvFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiScaleConvFeatureExtractor, self).__init__()

        # self.conv3 = nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1)  # 卷积核大小为3
        self.conv5 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)  # 卷积核大小为5
        self.conv7 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)  # 卷积核大小为7
        self.bn = nn.BatchNorm1d(128)  # 拼接后的通道数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # 添加 Dropout 层

        self.resblock1 = ResNetBlock(128, 100)
        self.resblock2 = ResNetBlock(100, 100)

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(100 * 2, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, x):
        # x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        x = torch.cat([x7, x5], dim=1)
        # x = x7
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.resblock1(x)
        x = self.resblock2(x)

        max_pool = self.global_max_pool(x).squeeze(-1)
        avg_pool = self.global_avg_pool(x).squeeze(-1)

        x = torch.cat([max_pool, avg_pool], dim=1)
        features = self.fc(x)
        features = self.fc2(features)
        logits = self.classifier(features)

        return logits, features


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    labels = data[:, -1]
    data = data[:, :-1]

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, labels


def evaluate_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    aupr = average_precision_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, auc, aupr, precision, recall, f1


def test_model(model, test_loader):
    model.eval()
    test_labels_all = []
    test_preds_all = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            test_labels_all.extend(labels.cpu().numpy())
            test_preds_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    acc, auc, aupr, precision, recall, f1 = evaluate_metrics(np.array(test_labels_all), np.array(test_preds_all))
    print(f"测试集结果 | ACC: {acc:.4f} | AUC: {auc:.4f} | AUPR: {aupr:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=20):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_labels_all = []
        val_preds_all = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_labels_all.extend(labels.cpu().numpy())
                val_preds_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        auc = roc_auc_score(val_labels_all, val_preds_all)
        aupr = average_precision_score(val_labels_all, val_preds_all)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | AUC: {auc:.4f} | AUPR: {aupr:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_multi_scale_conv_feature_extractor.pth")
            print(f"最佳模型在第 {epoch + 1} 轮保存，验证集准确率: {best_val_acc:.4f}")

    print("训练完成。最佳验证准确率:", best_val_acc)


if __name__ == "__main__":
    file_path = "../../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt"
    data, labels = load_data(file_path)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_tensor = TensorDataset(torch.tensor(train_data, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(train_labels, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(test_data, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleConvFeatureExtractor(input_dim=85, output_dim=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.5)

    print("开始训练模型...")
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=400)

    print("加载最佳模型并测试...")
    model.load_state_dict(torch.load("best_multi_scale_conv_feature_extractor.pth"))
    test_model(model, test_loader)
