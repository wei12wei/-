import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 建立資料
data = {
    "Model": [
        "Polynomial (deg=2)",
        "Ridge",
        "Linear",
        "Lasso",
        "ElasticNet"
    ],
    "R2": [0.018, 0.006929, 0.006929, -0.000002, -0.000002],
    "RMSE": [0.006164, 0.006198, 0.006198, 0.006220, 0.006220],
    "MAE": [0.003975, 0.004025, 0.004025, 0.004116, 0.004116],
    "MAPE": [0.157702, 0.159673, 0.159673, 0.163297, 0.163297]
}

df = pd.DataFrame(data)

# melt 成長格式以便繪圖
df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# 繪製圖表
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")
plt.title("Model Evaluation Metrics Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
