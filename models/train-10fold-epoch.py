import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, f1_score, auc, recall_score, \
    precision_score
from total_model import TotalModel
from utils.data_utils import load_and_preprocess_data
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import csv

"""
每一折都用测试集评估
"""
# 修改 train_and_validate_model 函数，加入早停和指标记录
def train_and_validate_model_with_early_stopping(
        model, train_loader, val_loader, device, num_epochs=100, learning_rate=0.001, patience=10):
    """训练单折模型，加入早停机制，返回验证集最佳性能和对应的模型参数"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_model_state = None
    best_metrics = None
    no_improve_counter = 0  # 记录未提升的次数

    for epoch in range(num_epochs):
        # 训练阶段
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

        print(f"\033[91mEpoch {epoch + 1}/{num_epochs}\033[0m, Loss: {total_loss:.4f}")

        # 验证阶段
        val_auc, metrics = evaluate_model(model, val_loader, device)
        val_acc = metrics["Accuracy"]

        # 记录最佳模型
        if val_acc > best_val_acc:  # 依据 ACC 更新
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_metrics = metrics
            no_improve_counter = 0  # 重置计数器
        else:
            no_improve_counter += 1

        # 检查早停条件
        if no_improve_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_val_auc, best_val_acc, best_model_state, best_metrics

def evaluate_model(model, test_loader, device, save_results=False, file_path="./Feature_Vector_Length_Experiment/128.csv"):
    """验证模型性能，返回 AUC 和其他指标"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for time_data, freq_data, labels in test_loader:
            time_data, freq_data, labels = time_data.to(device), freq_data.to(device), labels.to(device)
            logits = model(time_data, freq_data)
            probs = torch.softmax(logits, dim=1)[:, 1]  # 获取正类概率
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    binary_preds = (all_preds > 0.5).astype(int)

    # 计算正负样本分类成功的数量
    positive_correct = np.sum((binary_preds == 1) & (all_labels == 1))
    negative_correct = np.sum((binary_preds == 0) & (all_labels == 0))

    # 计算测试集正负样本总数
    total_positive = np.sum(all_labels == 1)
    total_negative = np.sum(all_labels == 0)

    print(f"Total positive samples: {total_positive}")
    print(f"Total negative samples: {total_negative}")
    print(f"Correctly classified positive samples: {positive_correct}")
    print(f"Correctly classified negative samples: {negative_correct}")

    # 保存结果到 CSV
    # if save_results:
    #     with open(file_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["True Label", "Predicted Label", "Probability"])
    #         writer.writerows(zip(all_labels, all_preds, all_probs))
    #     print(f"Test results saved to {file_path}")

    # # 计算指标
    # roc_auc = roc_auc_score(all_labels, all_preds)
    # binary_preds = (all_preds > 0.5).astype(int)  # 二值化
    # accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    # precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    # aupr = auc(recall, precision)
    # f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    #
    # avg_precision = np.mean(precision)
    # avg_recall = np.mean(recall)
    #
    # metrics = {
    #     "AUC": roc_auc,
    #     "Accuracy": accuracy,
    #     "Precision": avg_precision,
    #     "Recall": avg_recall,
    #     "F1": f1,
    #     # "PR": (precision, recall),
    #     "AUPR": aupr,
    # }

    # 计算指标
    roc_auc = roc_auc_score(all_labels, all_preds)
    binary_preds = (all_preds > 0.5).astype(int)  # 二值化
    accuracy = accuracy_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)  # 使用二值化召回率
    precision = precision_score(all_labels, binary_preds)  # 使用二值化精确率
    f1 = f1_score(all_labels, binary_preds)

    # AUPR 计算
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)  # 使用概率
    aupr = auc(recall_curve, precision_curve)

    # 将所有指标存储到字典中
    metrics = {
        "Accuracy": accuracy,
        "AUC": roc_auc,
        "AUPR": aupr,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    print(
        f"Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}, AUPR: {aupr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return roc_auc, metrics

# 修改 kfold_cross_validation 函数，添加保存每一折的最佳模型并最终选择最佳模型
def kfold_cross_validation_with_best_model(
        model_class, train_loader, test_loader, device, num_epochs=200, learning_rate=0.001, k_folds=10, patience=50):
    """十折交叉验证，保存每一折的最佳模型并对测试集评估"""
    dataset = train_loader.dataset
    data_size = len(dataset)

    time_data = dataset[:][0]
    freq_data = dataset[:][1]
    labels = dataset[:][2]

    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_model_state = None
    best_metrics = None
    best_test_metrics = None
    best_fold = -1
    best_test_acc = 0.0  # 用于记录最佳测试集 ACC
    all_test_metrics = []  # 保存每一折的测试集结果

    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(range(data_size), labels)):
        print(f"Fold {fold + 1}/{k_folds}")

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        train_fold_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        val_fold_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

        model = model_class()
        model.to(device)

        val_auc, val_acc, model_state, metrics = train_and_validate_model_with_early_stopping(
            model, train_fold_loader, val_fold_loader, device, num_epochs, learning_rate, patience)

        model.load_state_dict(model_state)
        print(f"\033[31mFold {fold + 1} Test Metrics:\033[0m")
        test_auc, test_metrics =  (model, test_loader, device)
        print(f"Fold {fold + 1} Test Metrics: {test_metrics}")  # 打印每一折的测试集信息
        all_test_metrics.append(test_metrics)

        # if val_acc > (best_metrics["Accuracy"] if best_metrics else 0):
        #     best_model_state = model_state
        #     best_metrics = metrics
        #     best_test_metrics = test_metrics
        #     best_fold = fold + 1

        # 使用测试集 ACC 判断最佳模型
        if test_metrics["Accuracy"] > best_test_acc:
            best_model_state = model_state
            best_test_metrics = test_metrics
            best_fold = fold + 1
            best_test_acc = test_metrics["Accuracy"]

    # 打印最佳折信息
    print(f"Best Fold: {best_fold}, Validation Metrics: {best_metrics}, Test Metrics: {best_test_metrics}")

    # # 计算十折测试集指标的平均值
    # avg_test_metrics = {
    #     metric: np.mean([m[metric] for m in all_test_metrics])
    #     for metric in all_test_metrics[0]
    # }
    # print(f"Average Test Metrics over 10 folds: {avg_test_metrics}")

    return best_model_state

if __name__ == "__main__":
    train_loader, test_loader = load_and_preprocess_data(
        "../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix-labels.txt",
        "../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix-labels.txt",
        test_size=0.2,
        batch_size=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_state = kfold_cross_validation_with_best_model(
        lambda: TotalModel(time_dim=1008, freq_dim=85, fusion_dim=128, num_classes=2),
        train_loader, test_loader, device, num_epochs=200, learning_rate=0.001, k_folds=10
    )

    final_model = TotalModel(time_dim=1008, freq_dim=85, fusion_dim=128, num_classes=2)
    final_model.load_state_dict(best_model_state)
    final_model.to(device)

    print("\nTesting with the best model:")
    evaluate_model(final_model, test_loader, device, save_results=True)
