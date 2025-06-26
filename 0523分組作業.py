import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklea   rn.metrics import mean_squared_error

# 讀取資料與初步處理
df = pd.read_csv("eccentricity.csv")
df = df.dropna()

print(f"樣本數：{df.shape[0]}, 特徵數（含目標）：{df.shape[1]}")
print("欄位名稱：", df.columns.tolist())

# 設定特徵與目標
features = ['speedSet', 'load_value']
target = 'sensor1'

X = df[features]
y = df[target]

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立多項式特徵
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

results = {}
y_preds = {}


# 計算 RMSE 的函式 (用 sklearn 新版建議)
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


# 訓練與評估
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

# 顯示結果表
result_df = pd.DataFrame(results).T.sort_values("R²", ascending=False)
print("\n模型評估指標：")
print(result_df)

# 找出最佳模型
best_model_name = result_df.index[0]
best_y_pred = y_preds[best_model_name]

# 真實 vs 預測圖
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=best_y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"{best_model_name} - True vs Predicted")
plt.tight_layout()
plt.savefig("true_vs_pred.png")
plt.close()

# 誤差分布圖
errors = y_test - best_y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, bins=30)
plt.title(f"{best_model_name} - Residual Distribution")
plt.xlabel("Prediction Error")
plt.tight_layout()
plt.savefig("residuals.png")
plt.close()

# R² 比較圖（改用 matplotlib）
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(result_df)))
plt.bar(result_df.index, result_df["R²"], color=colors)
plt.title("R² Comparison Across Models")
plt.ylabel("R²")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("r2_comparison.png")
plt.close()

# 結論
print(f"\n✅ 結論：")
print(f"最佳模型為：{best_model_name}")
print(f"其 R² = {result_df.loc[best_model_name, 'R²']:.4f}，RMSE = {result_df.loc[best_model_name, 'RMSE']:.4f}")
print(f"該模型可能效果較好，因為資料具備 {'非線性特徵' if 'Polynomial' in best_model_name else '線性或輕微正則化需求'}。")
print("建議可針對此模型進行進一步調參或部署。")
