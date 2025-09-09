import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, \
    recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


# 加载数据
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')  # 以制表符分隔
    labels = data[:, -1]  # 标签列
    features = data[:, :-1]  # 特征列

    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # 标准化

    return features, labels


# 评估指标
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


# 主函数
if __name__ == "__main__":
    file_path = "../../../../getVector/feature_vector/absorption_coefficient/0.2-1.05/feature_matrix.txt"
    # file_path = "../../../../getVector/feature_vector/time_domain_waveform_-73.6~-40/feature_matrix.txt"
    features, labels = load_data(file_path)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 定义模型
    models = {
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "CatBoost": CatBoostClassifier(iterations=100, verbose=0, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LDA": LinearDiscriminantAnalysis(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LV-SVM": SVC(kernel='linear', probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # 存储结果
    results = {}

    for name, model in models.items():
        print(f"\nTraining and evaluating model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
            X_test)

        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics

        print(f"Results for {name}: {metrics}")

    # 总结所有模型的指标
    print("\nSummary of Results:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")
