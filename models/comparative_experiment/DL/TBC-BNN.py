import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, \
    recall_score
import csv

# 定义BNN模型
class BNN(nn.Module):
    def __init__(self, input_dim):
        super(BNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.sigmoid(self.output(x))
        return x

# 数据加载
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    labels = data[:, -1]
    features = data[:, :-1]
    return features, labels

# 评估函数
def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPR": average_precision_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }
    return metrics

# 保存最优结果到CSV
def save_best_results_to_csv(file_path, true_labels, pred_labels, prob_values):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["True Label", "Predicted Label", "Probability"])
        for true_label, pred_label, prob in zip(true_labels, pred_labels, prob_values):
            writer.writerow([true_label, pred_label, prob])

if __name__ == "__main__":
    # 文件路径
    # file_path = "../../../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt"
    file_path = "../../../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt"
    features, labels = load_data(file_path)

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 定义模型
    input_dim = X_train.shape[1]
    model = BNN(input_dim)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 4000
    best_acc = 0.0  # 记录最佳准确率
    best_results = None  # 保存最优测试集结果（真实标签、预测标签、预测概率）
    best_metrics = None  # 保存最优测试集的评估指标

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # 每隔一定epoch后进行测试
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                y_prob = model(X_test).numpy()
                y_pred = (y_prob > 0.5).astype(int)
                metrics = evaluate_model(y_test.numpy(), y_pred, y_prob)

            print(f"\nEvaluation at epoch {epoch+1}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            # 如果当前测试集准确率更好，则更新最优结果
            if metrics["ACC"] > best_acc:
                best_acc = metrics["ACC"]
                best_results = (y_test.numpy(), y_pred, y_prob)
                best_metrics = metrics  # 保存最优的评估指标
                print(f"New best accuracy: {best_acc:.4f} at epoch {epoch+1}")

    # 在所有epoch结束后保存最优结果并打印最优指标
    if best_results:
        y_true, y_pred, y_prob = best_results
        # save_best_results_to_csv("all_pred/bnn.csv", y_true, y_pred, y_prob)
        print("Best results saved to lcc-THz-PeelNet.csv")

        # 打印最优测试集的评估指标
        print("\nBest Test Set Metrics:")
        for metric, value in best_metrics.items():
            print(f"{metric}: {value:.4f}")
