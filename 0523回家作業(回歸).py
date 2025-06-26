import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 設定字型避免中文亂碼
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 讀取資料
df = pd.read_csv("eccentricity.csv")
df = df.dropna()

print(f"樣本數：{df.shape[0]}, 特徵數（含目標）：{df.shape[1]}")
print("欄位名稱：", df.columns.tolist())

# 特徵與目標欄位
features = ['speedSet', 'load_value']
target = 'sensor1'

X = df[features]
y = df[target]

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 訓練/測試資料切分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 多項式特徵
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 定義模型
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (deg=2)": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# RMSE 計算函式
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

results = {}
y_preds = {}

# 訓練與評估模型
for name, model in models.items():
    if "Polynomial" in name:
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse_val = rmse(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results[name] = {"R²": r2, "RMSE": rmse_val, "MAE": mae, "MAPE": mape}
    y_preds[name] = y_pred

# 結果表格
result_df = pd.DataFrame(results).T.sort_values("R²", ascending=False)
print("\n模型評估指標：")
print(result_df)

# 找出最佳模型
best_model_name = result_df.index[0]
best_y_pred = y_preds[best_model_name]

# 真實 vs 預測圖
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=best_y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("實際值")
plt.ylabel("預測值")
plt.title(f"{best_model_name} - 實際 vs 預測")
plt.tight_layout()
plt.savefig("true_vs_pred.png")
plt.close()

# 殘差圖
errors = y_test - best_y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, bins=30)
plt.title(f"{best_model_name} - 殘差分布")
plt.xlabel("殘差（預測誤差）")
plt.tight_layout()
plt.savefig("residuals.png")
plt.close()

# R² 模型比較圖
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(result_df)))
plt.bar(result_df.index, result_df["R²"], color=colors)
plt.title("各模型 R² 比較")
plt.ylabel("R²")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("r2_comparison.png")
plt.close()

# 結論
print(f"\n✅ 結論：")
print(f"最佳模型為：{best_model_name}")
print(f"其 R² = {result_df.loc[best_model_name, 'R²']:.4f}，RMSE = {result_df.loc[best_model_name, 'RMSE']:.4f}")
print("該模型可能效果較好，因為資料具備 {}。".format(
    '非線性特徵' if 'Polynomial' in best_model_name else '線性或正則化需求'
))
