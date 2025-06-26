import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 CSV 檔案
df = pd.read_csv("eccentricity.csv")

from scipy.stats import chi2_contingency
# 只選有用的欄位進行卡方分析
# 你可以根據你資料選其他分類變量，例如 'gear_fault_desc', 'bearing_fault_desc'
cat1 = 'gear_fault'
cat2 = 'bearing_fault'

# 建立列聯表（contingency table）
contingency = pd.crosstab(df[cat1], df[cat2])

# 卡方檢定
chi2, p, dof, expected = chi2_contingency(contingency)

print("Chi-Square Test")
print("Chi2 Statistic:", chi2)
print("p-value:", p)

# 計算 standardized residuals
residuals = (contingency - expected) / expected**0.5

# 顯示殘差熱力圖
plt.figure(figsize=(10, 6))
sns.heatmap(residuals, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title(f"Chi-Square Residual Heatmap\np-value = {p:.4f}", fontsize=16)
plt.xlabel(cat2)
plt.ylabel(cat1)
plt.tight_layout()
plt.show()

# 只選取數值型欄位（排除時間和文字欄）
numeric_df = df.select_dtypes(include=['number'])

# --- 皮爾森相關係數圖 ---
pearson_corr = numeric_df.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title("Pearson Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig("pearson_corr.png")
plt.show()

# --- 肯德爾相關係數圖 ---
kendall_corr = numeric_df.corr(method='kendall')
plt.figure(figsize=(10, 8))
sns.heatmap(kendall_corr, annot=True, cmap='YlGnBu', fmt=".2f", square=True, linewidths=0.5)
plt.title("Kendall Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig("kendall_corr.png")
plt.show()


# ======== 直方圖與散點圖 ========

# 1. Sensor1 直方圖
plt.figure(figsize=(10, 5))
plt.hist(df['sensor1'], bins=30, color='orange', edgecolor='black', alpha=0.7)
plt.title("Distribution of Sensor1 (X)", fontsize=16)
plt.xlabel("Sensor1 (X)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("hist_sensor1.png")
plt.show()

# 2. Sensor2 直方圖
plt.figure(figsize=(10, 5))
plt.hist(df['sensor2'], bins=30, color='green', edgecolor='black', alpha=0.7)
plt.title("Distribution of Sensor2 (Y)", fontsize=16)
plt.xlabel("Sensor2 (Y)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("hist_sensor2.png")
plt.show()

# 3. SpeedSet vs Sensor1
plt.figure(figsize=(10, 6))
plt.scatter(df['speedSet'], df['sensor1'], color='blue', alpha=0.6)
plt.title("Speed vs Sensor1", fontsize=16)
plt.xlabel("Speed (speedSet)", fontsize=14)
plt.ylabel("Sensor1 (X)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("scatter_speed_sensor1.png")
plt.show()

# 4. SpeedSet vs Sensor2
plt.figure(figsize=(10, 6))
plt.scatter(df['speedSet'], df['sensor2'], color='blue', alpha=0.6)
plt.title("Speed vs Sensor2", fontsize=16)
plt.xlabel("Speed (speedSet)", fontsize=14)
plt.ylabel("Sensor2 (Y)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("scatter_speed_sensor2.png")
plt.show()

# 5. Load vs Sensor1
plt.figure(figsize=(10, 6))
plt.scatter(df['load_value'], df['sensor1'], color='blue', alpha=0.6)
plt.title("Load vs Sensor1", fontsize=16)
plt.xlabel("Load (load_value)", fontsize=14)
plt.ylabel("Sensor1 (X)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("scatter_load_sensor1.png")
plt.show()

# 6. Load vs Sensor2
plt.figure(figsize=(10, 6))
plt.scatter(df['load_value'], df['sensor2'], color='blue', alpha=0.6)
plt.title("Load vs Sensor2", fontsize=16)
plt.xlabel("Load (load_value)", fontsize=14)
plt.ylabel("Sensor2 (Y)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("scatter_load_sensor2.png")
plt.show()
