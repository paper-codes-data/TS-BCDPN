import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=64, num_layers=2, dropout_rate=0.7):
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
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        fc1_out = self.fc1(lstm_out)
        fc1_out = self.relu(fc1_out)
        fc2_out = self.fc2(fc1_out)

        return fc2_out


# 加载数据的函数
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    labels = data[:, -1]
    data = data[:, :-1]

    # 标准化数据
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, labels


# 模型训练函数
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50,
          save_path="best_spectral_feature_extractor.pth"):
    best_val_acc = 0.0
    best_auc = 0.0
    best_aupr = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    best_recall = 0.0

    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_preds / total_preds
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_acc, auc_score, aupr, precision, recall, recall, f1 = validate(model, val_loader, device)
        print(f"Validation - Accuracy: {val_acc:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # If the current validation accuracy is the best, save the model and the metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_auc = auc_score
            best_aupr = aupr
            best_precision = precision
            best_f1 = f1
            best_recall = recall
            best_model_state = model.state_dict()
            # torch.save(best_model_state, save_path)

    print("Training completed.")
    print(f"Best validation results - Accuracy: {best_val_acc:.4f}, AUC: {best_auc:.4f}, AUPR: {best_aupr:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f},F1: {best_f1:.4f}")
    return model


# 验证过程函数
def validate(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类的概率

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # 计算指标
    acc = np.mean(all_labels == (all_preds > 0.5))  # 使用阈值0.5来计算准确率
    auc_score = roc_auc_score(all_labels, all_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    aupr = auc(recall, precision)
    precision_score_val = precision_score(all_labels, (all_preds > 0.5))
    recall = recall_score(all_labels, (all_preds > 0.5))
    recall_score_val = recall_score(all_labels, (all_preds > 0.5))
    f1 = f1_score(all_labels, (all_preds > 0.5))

    return acc, auc_score, aupr, precision_score_val, recall_score_val, recall, f1


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = "../../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt"
    data, labels = load_data(data_path)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
    val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).long())

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    model = LSTMFeatureExtractor(input_dim=data.shape[1], hidden_dim=128, output_dim=64, num_layers=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=400,
                  save_path="best_lstm_feature_extractor.pth")


if __name__ == "__main__":
    main()
