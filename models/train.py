import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from total_model import TotalModel
# from utils.data_utils import load_and_preprocess_data
from utils.data_utils import load_and_preprocess_data


def train_model(model, train_loader, test_loader, num_epochs=600, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for time_data, freq_data, labels in train_loader:
            time_data, freq_data, labels = time_data.to(device), freq_data.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(time_data, freq_data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

        # 验证阶段
        evaluate_model(model, test_loader, device)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for time_data, freq_data, labels in test_loader:
            time_data, freq_data, labels = time_data.to(device), freq_data.to(device), labels.to(device)
            logits = model(time_data, freq_data)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算并输出分类报告
    print(classification_report(all_labels, all_preds, digits=4))

    # 计算 AUC
    auc = roc_auc_score(all_labels, all_preds)
    print(f"AUC: {auc:.4f}")

    # 计算 Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_loader, test_loader = load_and_preprocess_data("../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt",
                                                         "../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt")
    model = TotalModel(time_dim=1008, freq_dim=140, fusion_dim=128, num_classes=2)
    train_model(model, train_loader, test_loader)
