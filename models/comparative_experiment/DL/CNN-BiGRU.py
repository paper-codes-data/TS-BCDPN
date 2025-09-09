import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score



class CNN_BiGRU_Recognition(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN_BiGRU_Recognition, self).__init__()

        # CNN 模块：5个卷积层和5个平均池化层
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # 输出通道16
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # 输出通道32
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # 输出通道64
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 输出通道128
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # 输出通道256
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        # BiGRU 模块：3层BiGRU
        self.bigru = nn.GRU(input_size=256, hidden_size=512, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=1024, hidden_size=256, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(input_size=512, hidden_size=96, batch_first=True, bidirectional=True)

        # Dropout 层和识别网络
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(192, output_dim)  # 96 * 2 (BiGRU输出) -> output_dim
        self.softmax = nn.Softmax(dim=1)

        # 动态计算展平维度（fc的输入）
        self._calculate_flattened_dim(input_dim)

    def _calculate_flattened_dim(self, input_dim):
        temp_input = torch.zeros(1, 1, input_dim)  # 假设batch_size=1
        temp_output = self.cnn(temp_input)
        self.cnn_output_dim = temp_output.size(2)  # 输出序列长度

    def forward(self, x):
        # 输入 x 形状: [batch_size, 1, input_dim]
        x = self.cnn(x)  # CNN输出: [batch_size, 256, seq_length]
        x = x.permute(0, 2, 1)  # 交换维度: [batch_size, seq_length, channels]

        # BiGRU 处理
        x, _ = self.bigru(x)  # 输出: [batch_size, seq_length, 1024]
        x, _ = self.bigru2(x)  # 输出: [batch_size, seq_length, 512]
        x, _ = self.bigru3(x)  # 输出: [batch_size, seq_length, 192]

        # 取最后时间步的输出
        x = x[:, -1, :]  # [batch_size, 192]

        # Dropout 和全连接
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, output_dim]
        x = self.softmax(x)  # 输出概率
        return x
def load_data(file_path):
    # 加载数据
    data = np.loadtxt(file_path, delimiter='\t')  # 假设数据是制表符分隔
    labels = data[:, -1]  # 最后一列为标签
    data = data[:, :-1]  # 特征数据

    # 标准化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
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
        val_probs_all = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_labels_all.extend(labels.cpu().numpy())
                val_preds_all.extend(preds.cpu().numpy())
                val_probs_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        auc = roc_auc_score(val_labels_all, val_probs_all)
        aupr = average_precision_score(val_labels_all, val_probs_all)
        f1 = f1_score(val_labels_all, np.round(val_probs_all))
        precision = precision_score(val_labels_all, np.round(val_probs_all))
        recall = val_correct / val_total

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"AUC: {auc:.4f} | AUPR: {aupr:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} |F1: {f1:.4f} |  "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 打印并保存最后一个 epoch 的结果
    print("\n最后一个Epoch的测试集结果:")
    print(f"ACC: {val_acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # 保存结果到 CSV 文件
    # import csv
    # with open("CNN-BiGRU.csv", mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["True Label", "Predicted Label", "Predicted Probability"])
    #     for label, pred, prob in zip(val_labels_all, val_preds_all, val_probs_all):
    #         writer.writerow([label, pred, prob])



if __name__ == "__main__":
    # 数据文件路径
    file_path = "../../../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt"
    # file_path = "../../../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt"
    data, labels = load_data(file_path)

    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_tensor = TensorDataset(torch.tensor(train_data, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(train_labels, dtype=torch.long))
    val_tensor = TensorDataset(torch.tensor(val_data, dtype=torch.float32).unsqueeze(1),
                               torch.tensor(val_labels, dtype=torch.long))

    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_data.shape[1]  # 输入特征维度
    output_dim = 2  # 假设二分类任务
    model = CNN_BiGRU_Recognition(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.5)

    print("开始训练模型...")
    print(f"训练集大小: {len(train_tensor)}, 测试集大小: {len(val_tensor)}")
    print(f"使用设备: {device}")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数数量: {total_params}")

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=300)
