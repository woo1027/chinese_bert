import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


# 中文字體設定（如需）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取資料
game = pd.read_csv("D:/面試資料/R/vgsales_utf8.csv", encoding="utf-8")

# 欄位命名
game.columns = ['排名', '遊戲名稱', '平台', '發行年份', '遊戲類型', '發行商',
                '北美銷售額', '歐洲銷售額', '日本銷售額', '其他地區銷售額', '總銷售額']

# 移除"排名"欄
game = game.drop(columns=['排名'])

# 查看欄位是否有 "N/A"
for col in game.columns:
    na_count = (game[col] == "N/A").sum()
    if na_count > 0:
        print(f"欄位 {col} 有 {na_count} 個 \"N/A\"")


# 顯示缺失值數量
print(game.isna().sum())

# # 將年份轉為數字
# game['發行年份'] = pd.to_numeric(game['發行年份'], errors='coerce')

game['發行年份'] = pd.to_numeric(game['發行年份'], errors='coerce')
game['總銷售額'] = pd.to_numeric(game['總銷售額'], errors='coerce')
game.dropna(subset=['發行年份', '總銷售額'], inplace=True)
#
#
# # 類別特徵轉換
# label_cols = ['平台', '遊戲類型', '發行商']
# for col in label_cols:
#     game[col] = LabelEncoder().fit_transform(game[col].astype(str))
#
# # 特徵與目標欄位
# features = ['平台', '遊戲類型', '發行年份', '發行商']
# X = game[features].values
# y = game['總銷售額'].values.reshape(-1, 1)
#
#
# # 特徵/目標標準化
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y)
#
#
# # 資料切分
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
#
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from xgboost import XGBRegressor
#
# models = {
#     'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
#     'Decision Tree': DecisionTreeRegressor(random_state=42),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'SVM': SVR(kernel='rbf'),
#     'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
# }
#
# print("\n🔍 各模型預測表現：")
# for name, model in models.items():
#     model.fit(X_train, y_train.ravel())
#     y_pred = model.predict(X_test)
#
#     # 反標準化
#     y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
#     y_true = scaler_y.inverse_transform(y_test)
#
#     mse = mean_squared_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     print(f"{name:<20} ➤ MSE: {mse:.2f}, R²: {r2:.2f}")



features = ['平台', '遊戲類型', '發行年份', '發行商']
target = ['總銷售額']

mean_sales = game[target].mean()
game['Best_Seller'] = (game[target] > mean_sales).astype(int)

# 類別特徵轉換
game = game[features + ['Best_Seller']].dropna()
label_cols = ['平台', '遊戲類型', '發行商']
for col in label_cols:
    game[col] = LabelEncoder().fit_transform(game[col].astype(str))

X = game[features]
y = game['Best_Seller']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化（對某些模型效果更好）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)


# 6. 定義模型們
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', probability = True),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}


# 模型評估：混淆矩陣
plt.figure(figsize=(15, 10))
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, 3, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name}\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 計算指標
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 顯示指標（註解在子圖下方）
    plt.gca().text(0, -0.6,
                   f'Acc: {acc:.2f}\nPrec: {prec:.2f}\nRec: {rec:.2f}\nF1: {f1:.2f}',
                   fontsize=10, ha='left', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()

# ROC 曲線
# 選擇要跑的模型
selected_models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}
plt.figure(figsize=(10, 8))

for name, model in selected_models.items():
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # 預測正類機率
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # 參考線
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 曲線')
plt.legend()
plt.grid()
plt.show()

# # 每年銷售額
# sales_by_year = game.groupby('發行年份', as_index=False)['總銷售額'].sum()
#
# # 折線圖：年度銷售額
# plt.figure(figsize=(10, 5))
# sns.lineplot(data=sales_by_year, x='發行年份', y='總銷售額', marker='o', color='black')
# plt.title("年度銷售總額")
# plt.xlabel("年份")
# plt.ylabel("總銷售額(百萬)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 不同遊戲類型的銷售數量
# type_count = game['遊戲類型'].value_counts().reset_index()
# type_count.columns = ['遊戲類型', '數量']
#
# plt.figure(figsize=(10, 5))
# sns.barplot(data=type_count, x='遊戲類型', y='數量', color='darkgreen')
# for i, row in type_count.iterrows():
#     plt.text(i, row['數量'] + 10, row['數量'], ha='center', size=8)
# plt.title("遊戲類別發布數量")
# plt.xlabel("遊戲類別")
# plt.ylabel("數量")
# plt.xticks(rotation=10)
# plt.tight_layout()
# plt.show()
#
# # 不同遊戲類型的銷售金額
# type_sales = game.groupby('遊戲類型', as_index=False)['總銷售額'].sum()
#
# plt.figure(figsize=(10, 5))
# sns.barplot(data=type_sales.sort_values('總銷售額', ascending=False),
#             x='遊戲類型', y='總銷售額', color='steelblue')
# for i, row in type_sales.sort_values('總銷售額', ascending=False).iterrows():
#     plt.text(i, row['總銷售額'] + 1, round(row['總銷售額'], 1), ha='center', size=8)
# plt.title("遊戲類別銷售總額")
# plt.xlabel("遊戲類別")
# plt.ylabel("銷售額(百萬)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 不同平台的銷售金額
# platform_sales = game.groupby('平台', as_index=False)['總銷售額'].sum()
#
# plt.figure(figsize=(10, 5))
# sns.barplot(data=platform_sales.sort_values('總銷售額', ascending=False),
#             x='平台', y='總銷售額', color='steelblue')
# for i, row in platform_sales.sort_values('總銷售額', ascending=False).iterrows():
#     plt.text(i, row['總銷售額'] + 1, round(row['總銷售額'], 1), ha='center', size=8)
# plt.title("平台銷售總額")
# plt.xlabel("平台")
# plt.ylabel("銷售額(百萬)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 多平台遊戲統計
# duplicate_games = game.groupby('遊戲名稱')['平台'].nunique().reset_index()
# duplicate_games = duplicate_games[duplicate_games['平台'] > 1]
#
# multi_platform_games = game[game['遊戲名稱'].isin(duplicate_games['遊戲名稱'])] \
#     .sort_values(by=['遊戲名稱', '平台'])
#
# print(multi_platform_games)
#
# # 發行商銷售金額前 10 名
# publisher_sales = game.groupby('發行商', as_index=False)['總銷售額'].sum()
# top10_publishers = publisher_sales.sort_values('總銷售額', ascending=False).head(10)
#
# plt.figure(figsize=(10, 5))
# sns.barplot(data=top10_publishers, x='發行商', y='總銷售額', color='steelblue')
# for i, row in top10_publishers.iterrows():
#     plt.text(i, row['總銷售額'] + 1, round(row['總銷售額'], 1), ha='center', size=8)
# plt.title("發行商銷售總額")
# plt.xlabel("發行商")
# plt.ylabel("銷售額(百萬)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 地區與遊戲類型比例（長資料格式）
# df_long = game.melt(
#     id_vars=['遊戲類型'],
#     value_vars=['北美銷售額', '歐洲銷售額', '日本銷售額', '其他地區銷售額'],
#     var_name='Region',
#     value_name='Sales'
# )
#
# region_grouped = df_long.groupby(['Region', '遊戲類型'], as_index=False)['Sales'].sum()
# region_grouped['Percentage'] = region_grouped.groupby('Region')['Sales'].apply(lambda x: 100 * x / x.sum())
#
# top_labels = region_grouped.loc[region_grouped.groupby('Region')['Percentage'].idxmax()]
#
# # 繪製極坐標圖（pie charts 使用 subplot + polar）
# import plotly.express as px
#
# fig = px.sunburst(
#     region_grouped,
#     path=['Region', '遊戲類型'],
#     values='Sales',
#     color='遊戲類型',
#     title='各地區不同遊戲類型銷售比例'
# )
# fig.show()

