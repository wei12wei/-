# 匯入必要套件
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 載入資料
df = pd.read_csv("eccentricity.csv")

# 合成三類資料（模擬有三種類別的情境）
df_multi = pd.concat([
    df.sample(5000, random_state=1).assign(gear_fault_desc='eccentricity'),
    df.sample(5000, random_state=2).assign(gear_fault_desc='misalignment'),
    df.sample(5000, random_state=3).assign(gear_fault_desc='normal')
], ignore_index=True)

# 選擇特徵與目標
features = ['sensor1', 'sensor2', 'speedSet', 'load_value']
X = df_multi[features]
y = df_multi['gear_fault_desc']

# 編碼類別標籤
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 資料切分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# 模型定義
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# 儲存結果
results = {}

# 訓練與評估每個模型
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 多類別 AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')

    # 評估指標
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "auc_roc": auc
    }

# 顯示比較結果
result_df = pd.DataFrame(results).T
print("\n三種模型的分類效能比較：\n")
print(result_df.round(4))
