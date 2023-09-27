# coding: utf-8
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score , silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import shap
import os
import random
os.environ["OMP_NUM_THREADS"] = "1"
# 計算新進工程師的比例
def calculate_ratio(engineers, engineer_dict):
    # 將工程師名單分割成 pandas 的 Series
    engineer_series = pd.Series(engineers.split(','))
    # 使用 isin 函數來得到一個布林值的 Series，表示每個工程師是否是新進工程師
    new_engineers = engineer_series.isin([engineer for engineer in engineer_dict if engineer_dict[engineer] <= 3]).sum()
    # 計算新進工程師的比例
    ratio = new_engineers / len(engineer_series) if len(engineer_series) != 0 else 0
    return ratio

# 計算平均經費規模
def calculate_average_salary(salary_range_str):
    # 分割經費規模範圍
    salary_range = salary_range_str.split('-')
    # 使用正則表達式過濾出數字部分
    filtered_salary_range = [re.sub(r'\D', '', salary) for salary in salary_range]
    # 計算平均經費規模
    average_salary = sum(int(salary) for salary in filtered_salary_range) / len(filtered_salary_range) + random.randint(-10,10)

    return average_salary

# 預處理舊的專案和工程師資料
def preprocess_old_data(project, engineer):
    # 刪除專案起訖時間欄位
    project = project.drop("專案起訖時間", axis=1)
    # 計算平均經費規模
    project['經費規模'] = project['經費規模'].apply(calculate_average_salary)
    # 將難易度映射為數字
    difficulty_mapping = {'易': 1, '中': 2, '難': 3}
    project['難中易'] = project['難中易'].map(difficulty_mapping)
    # 對專案類型進行 one-hot 編碼
    project = pd.get_dummies(project, columns=['專案類型'])
    # 移除工程師名稱中的 "工程師" 字樣
    engineer['工程師'] = engineer['工程師'].str.replace('工程師', '')
    # 僅保留有工程師與單位(年)資料的列
    engineer_dict = engineer[['工程師', '單位(年)']].dropna()
    # 計算工程師的平均年資
    engineer_dict['單位(年)'] = engineer_dict['單位(年)'].apply(lambda x: int((int(x.split('-')[0]) + int(x.split('-')[1])) / 2)) 
    # 將工程師與單位(年)的資料轉換為字典形式
    engineer_dict = dict(engineer_dict.values)
    # 將工程師資料合併到專案資料中
    project = pd.concat([project, engineer["負責工程師"], engineer["是否有團隊協作"]], axis=1)
    # 計算新進工程師比例
    project['新進工程師比例'] = project['負責工程師'].apply(lambda x: calculate_ratio(x, engineer_dict))
    # 將負責工程師欄位轉換為平均年資
    project['平均年資'] = project['負責工程師'].str.split(',').apply(lambda x: np.mean([int(engineer_dict[engineer]) for engineer in x]))
    return project
#載入資料
def load_data(project_file_pattern, project_engineer_pattern,engineer_pattern):
        project_files = glob.glob(project_file_pattern)
        project_engineer_files = glob.glob(project_engineer_pattern)
        engineer_files = glob.glob(engineer_pattern)
        engineer_data_frames = [pd.read_csv(file, encoding='utf-8') for file in engineer_files]
        engineer_data = pd.concat(engineer_data_frames, axis=0)
        projects = []
        for project_file, project_engineer_file in zip(project_files, project_engineer_files):
            project = pd.read_csv(project_file)
            project_engineer = pd.read_csv(project_engineer_file)
            project = preprocess_old_data(project, project_engineer)
            projects.append(project)
        return pd.concat(projects, axis=0).fillna(False),engineer_data
#專案成效前處理
def preprocess_effectiveness(project):
    # 將 '專案成效' 列分割為兩個單獨的列
    project['成效分數'] = project['專案成效'].apply(lambda x: int(x.split('/')[0]))  # 分數
    project['成效描述'] = project['專案成效'].apply(lambda x: '/'.join(x.split('/')[1:]))  # 描述
    # 刪除原始的 '專案成效' 列
    project.drop('專案成效', axis=1, inplace=True)
    # 定義每個指標的關鍵字模式
    keywords = {
        '需求明確度': [('需求明確', 8), ('需求不明確', -8)],
        '需求變更頻率': [('需求調整', -5), ('需求變更', -5), ('來回討論', -10)],
        '專案工時': [('期限內完成', 20), ('限期內完工', 20), ('評估工時長', -10), ('完成時間過長', -20)]
    }
    # 初始化指標列
    for indicator in keywords.keys():
        project[indicator] = 0
    # 基於 '成效描述' 中的關鍵字出現情況填充指標列
    for indicator, words in keywords.items():
        for word, w in words:
            project[indicator] += project['成效描述'].str.contains(word).astype(int) * w
    project['專案工時差'] =  project['評估工時'] - project['開發工時']
    project['專案工時'] = project['專案工時'] + project['專案工時差']
    # 分割標籤和特徵
    X = project[['專案工時', '需求明確度', '需求變更頻率']]
    y = project['成效分數']
    # 切割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 特徵縮放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 使用 MLPRegressor 模型進行訓練
    mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    # 準備解釋器
    explainer = shap.KernelExplainer(mlp.predict, X_train)
    # 計算 SHAP 值
    shap_values = explainer.shap_values(X_test)
    # 使用 SHAP 值的平均絕對值作為特征重要性
    importance_scores = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    # 將 '需求變更頻率' 的值進行反轉
    project['需求變更頻率'] = project['需求變更頻率'].max() - project['需求變更頻率']
    # 對所有指標應用 Min-Max Scaling
    for indicator in importance_scores.index:
        project[indicator] = (project[indicator] - project[indicator].min()) / (project[indicator].max() - project[indicator].min())
    # 計算每個指標的分數
    for indicator in importance_scores.index:
        if indicator == '專案工時':
            project[indicator + '分數'] = project[indicator] * project['成效分數'] * importance_scores[indicator]
        elif indicator == '需求變更頻率':
            project[indicator + '分數'] = abs((project['需求變更頻率'].max() - project['需求變更頻率']) * project['成效分數'] * importance_scores[indicator])
        else:
            project[indicator + '分數'] = abs(project[indicator] * project['成效分數'] * importance_scores[indicator])
    # 計算每個專案的總得分
    project['總得分'] = project[[indicator + '分數' for indicator in importance_scores.index]].sum(axis=1)
    # 根據每個專案的總得分調整每個指標的得分
    for indicator in importance_scores.index:
        project[indicator + '分數'] = project[indicator + '分數'] / project['總得分'] * project['成效分數']
    project.rename(columns={'需求變更頻率分數': '需求變更穩定度分數'}, inplace=True)
    # 刪除不必要的列
    project = project.drop(['總得分','成效描述'] + [indicator for indicator in importance_scores.index], axis=1)
    data = project.copy()
    return data,project
#製作工程師列表
def engineer_df(data,engineer_data):
    # 建立一個空的列表，用於儲存拆分後的紀錄。
    new_records = []
    data['負責工程師'] = data['負責工程師'].str.split(',')

    # 遍歷原始資料框的每一行。
    for _, row in data.iterrows():
        # 獲取原始記錄中負責的工程師列表。
        engineers = row['負責工程師']

        # 針對每個工程師，創建一個新的記錄，並將原始記錄的其他值複製過來。
        for engineer in engineers:
            new_record = row.copy()
            new_record['負責工程師'] = engineer
            
            # 將新記錄添加到列表中。
            new_records.append(new_record)

    # 將列表轉換為 DataFrame
    new_df = pd.DataFrame(new_records)
    # 算平均前先轉成浮點數
    float_columns = ['經費規模', '評估工時', '難中易', '是否有團隊協作']
    new_df[float_columns] = new_df[float_columns].astype(float)

    # 計算每個負責工程師的平均經費、平均開發時間、平均難易度、平均協作人數及 app、客製化、維護案、網站開發的數量 專案成效
    agg_columns = {
        '經費規模': 'mean',
        '評估工時': 'mean',
        '難中易': 'mean',
        '專案類型_APP開發': 'sum',
        '專案類型_客製化專案': 'sum',
        '專案類型_維護案': 'sum',
        '專案類型_網站開發': 'sum',
        '是否有團隊協作': 'mean',
        '成效分數': 'mean',
        '專案工時分數': 'mean',
        '需求明確度分數': 'mean',	
        '需求變更穩定度分數': 'mean'
    }
    avg_count_df = new_df.groupby('負責工程師').agg(agg_columns)

    # 將索引從工程師改回數值
    avg_count_df.reset_index(inplace=True)
    avg_count_df = avg_count_df.rename(columns={'負責工程師': '工程師'})

    # 與年資合併
    engineer_df = pd.merge(avg_count_df, engineer_data.loc[:, ['工程師','單位(年)']], how="outer")
    engineer_df = engineer_df.fillna(0) #若工程師沒有專案資料則填0
    # 與年資合併
    engineer_df = pd.merge(avg_count_df, engineer_data.loc[:,:], how="outer")
    engineer_df = engineer_df.fillna(0) #若工程師沒有專案資料則填0
    engineer_df.rename(columns={'是否有團隊協作': '協作人數', '單位(年)': '年資'}, inplace=True)

    # 選擇用於分群的特徵（除了工程師名稱以外的所有特徵）
    features = engineer_df.iloc[:, 1:]
    # 進行特徵標準化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 計算不同 k 值的 Silhouette 分數
    silhouette_scores = []
    best_k = 3        #初始化最佳群數
    best_score = -1   #初始化最佳分數
    max_clusters = 6  # 限制最大群集數量

    # 計算不同 k 值的 Silhouette 分數，但不超過最大群集數量
    for i in range(3, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42,n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        silhouette_avg = silhouette_score(scaled_features, clusters)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = i
    # 使用最佳的 k 值進行 K-means 分群
    kmeans = KMeans(n_clusters=best_k, random_state=42,n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    # 將分群結果添加到原始 DataFrame
    engineer_df['類型'] = clusters
    return(engineer_df)
#將分群後的工程師加入project資料
def project2project_engineer(project,engineer_df):
    # 轉換為 DataFrame
    engineer_cluster_df = engineer_df
    # 創建工程師到群集的映射
    engineer_to_cluster_mapping = engineer_cluster_df.set_index('工程師')['類型'].to_dict()

    # 將 "負責工程師" 列分割並轉換為列表
    project['負責工程師'] = project['負責工程師'].str.split(",")
    # 對每個工程師列表進行映射，得到群集列表
    project['負責工程師群集'] = project['負責工程師'].apply(lambda engineers: [engineer_to_cluster_mapping[eng] for eng in engineers])

    # 獲取所有獨特的群集標籤
    unique_clusters = project['負責工程師群集'].explode().dropna().unique()

    # 為每個群集創建一個新列，計算每行中該群集的工程師數量
    for cluster in unique_clusters:
        project[f'工程師類型_{cluster}'] = project['負責工程師群集'].apply(lambda clusters: clusters.count(cluster))

    # 移除 "負責工程師" 和 "負責工程師群集" 列
    project.drop(columns=['負責工程師', '負責工程師群集',], inplace=True)
    #轉換成數字
    project = project.apply(pd.to_numeric, errors='coerce')
    project = project.fillna(0)
    return project
