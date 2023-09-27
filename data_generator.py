# coding: utf-8
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import zscore
from sklearn.neighbors import KernelDensity
import preprocessdata as predata
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy.stats import spearmanr
def gen_data(proj, new_corr=None):
    """
    根據原始數據和新環境的相關係數生成合成數據。
    
    參數:
    - proj: 原始數據 DataFrame
    - new_corr: 新環境的相關係數字典
    
    返回:
    - 合成數據 DataFrame
    """
    
    # 數值型特征列名
    num_cols = proj.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()

    data_scaled = scaler.fit_transform(proj[num_cols])
    
    # 執行 PCA
    pca = PCA()
    data_pca = pca.fit_transform(data_scaled)
    
    #對每個主成分使用 KDE 生成新的值
    num_samples = int(len(proj) * 2)
    new_data_pca = []
    for i in range(data_pca.shape[1]):
        kde = KernelDensity(kernel='gaussian').fit(data_pca[:, i].reshape(-1, 1))
        new_data_pca.append(kde.sample(num_samples).flatten())
    
    new_data_pca = np.array(new_data_pca).T
    
    # 使用 PCA 的逆轉換得到原始特征空間中的新樣本
    new_data_scaled = pca.inverse_transform(new_data_pca)
    
    # 將標準化的數據轉換回原始的尺度
    new_data = scaler.inverse_transform(new_data_scaled)
    new_data = pd.DataFrame(new_data, columns=num_cols)
    #計算原資料的斯皮爾曼相關係數
    columns_to_check = ['經費規模', '評估工時', '難中易', '是否有團隊協作', '平均年資']
    original_corr = {}
    for col in columns_to_check:
        correlation, _ = spearmanr(proj[col], proj['開發工時'])
        original_corr[col] = correlation
    # 若提供新環境的相關係數，進行資料轉換
    if new_corr:
        # 使用 new_corr 更新 original_corr
        original_corr.update(new_corr)
        new_data = transform(new_data, original_corr)
    else:
        new_data = transform(new_data, original_corr)
    # 自動定義專案類型
    proj_types = [col for col in proj.columns if col.startswith("專案類型_")]
    new_data[proj_types] = 0  # 初始化為0

    # 計算原始數據中各個專案類型的頻率
    proj_types = [col for col in proj.columns if col.startswith("專案類型_")]
    proj_type_frequencies = proj[proj_types].sum() / len(proj)

    # 在生成新數據時，按照這些頻率來隨機選擇專案類型
    for idx in new_data.index:
        chosen_proj_type = np.random.choice(proj_types, p=proj_type_frequencies.values)
        new_data.at[idx, chosen_proj_type] = 1

    # 計算 "成效分數"
    score_cols = [col for col in new_data.columns if "分數" in col and col != '成效分數']
    new_data['成效分數'] = new_data[score_cols].sum(axis=1)

    # 合併原始數據與新數據
    return pd.concat([proj, new_data])
def transform(data, new_corr):
    """
    轉換數據以反映新環境的相關性。
    
    參數:
    - data: 原始或KDE生成的數據
    - new_corr: 新環境的相關係數字典
    
    返回:
    - 轉換後的數據
    """
    
    def obj_fn(w):
        data['開發工時'] = sum(w[i] * data[feat] for i, feat in enumerate(new_corr.keys()))
        corr_with_target = data.corr()['開發工時']
        return sum((corr_with_target[feat] - target) ** 2 for feat, target in new_corr.items())

    # 優化找到最佳權重
    init_w = np.ones(len(new_corr))
    result = minimize(obj_fn, init_w, method='Nelder-Mead')
    opt_w = {feat: weight for feat, weight in zip(new_corr.keys(), result.x)}
    data['開發工時'] = sum(opt_w[feat] * data[feat] for feat in opt_w.keys())

    # 裁剪所有欄位以確保非負
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].clip(lower=0)

    return data
def gendata(project,new_environment_correlations):
    while True:
        combined_data = gen_data(project, new_environment_correlations)
        # 如果 "開發工時" 都不超過 500，則跳出迴圈
        if (combined_data['開發工時'] <= 500).all():
            project = combined_data
            break
    return combined_data