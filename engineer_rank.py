# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import glob
import pandas as pd
import preprocessdata as predata
import os
os.environ["OMP_NUM_THREADS"] = "1"
def rank_engineers(output, final_df, rank_score_list):
    # 將排名分數列表的每個元素轉換為浮點數
    rank_score_list = [float(item) for item in rank_score_list]
    
    # 正規化排名分數列表，使其總和為1
    rank_score_list = np.array(rank_score_list) / sum(rank_score_list)
    
    # 從輸出DataFrame中提取特定列的值
    df1_values = output[['經費規模', '評估工時', '難中易', '年資', '協作人數', '工程師類型']].values
    # 從最終DataFrame中提取特定列的值
    df2_values = final_df[['經費規模', '評估工時', '難中易', '年資', '協作人數', '類型']].values
    
    # 計算df1_values和df2_values之間的cosine相似度
    cosine_sim_matrix = cosine_similarity(df1_values, df2_values)
    
    # 從輸出DataFrame中獲取所有唯一的專案類型
    project_types = output['專案類型'].unique()
    # 為每個專案類型創建一個字典，該字典包含需要正規化的列名
    rank_data_dict = {
        project_type: final_df[['喜好_' + project_type, '擅長_' + project_type, '專案類型_' + project_type]].values
        for project_type in project_types
    }
    
    # 為每個專案類型計算最小值和最大值，用於後續的正規化
    min_max_vals_dict = {
        project_type: (data.min(axis=0), data.max(axis=0))
        for project_type, data in rank_data_dict.items()
    }
    
    # 獲取所有年資大於3的工程師的索引
    old_engineers_idx = np.where(final_df['年資'].values > 3)[0]
    # 獲取所有年資小於或等於3的工程師的索引
    new_engineers_idx = np.where(final_df['年資'].values <= 3)[0]
    
    # 從輸出DataFrame中提取專案類型、新進工程師比例和協作人數的值
    project_type_vals = output['專案類型'].values
    new_engineer_ratio_vals = output['新進工程師比例'].values
    cooperation_num_vals = output['協作人數'].values
    
    results = []
    # 遍歷每一行的輸出數據
    for idx, (project_type, new_engineer_ratio, cooperation_num, similarities) in enumerate(zip(project_type_vals, new_engineer_ratio_vals, cooperation_num_vals, cosine_sim_matrix)):
        # 根據專案類型從字典中獲取對應的數據
        rank_data = rank_data_dict[project_type]
        # 從字典中獲取對應的最小值和最大值
        min_vals, max_vals = min_max_vals_dict[project_type]
        
        # 正規化數據，使其在0和1之間
        normalized_rank_data = (rank_data - min_vals) / (max_vals - min_vals)
        
        # 計算最終的分數，該分數考慮了cosine相似度和正規化的數據
        scores = np.dot(np.column_stack((similarities, normalized_rank_data)), rank_score_list)
        
        # 根據新進工程師比例和協作人數計算新進和資深工程師的數量
        new_count = int(round(new_engineer_ratio * cooperation_num))
        old_count = int(cooperation_num - new_count)

        # 根據計算的分數對新進工程師和資深工程師進行排序
        new_engineers_sorted = new_engineers_idx[np.argsort(scores[new_engineers_idx])][::-1]
        old_engineers_sorted = old_engineers_idx[np.argsort(scores[old_engineers_idx])][::-1]

        # 從排序後的列表中選取前幾名工程師
        engineers_ranked = final_df['工程師'].values[new_engineers_sorted[:new_count]].tolist() + \
                           final_df['工程師'].values[old_engineers_sorted[:old_count]].tolist()
        # 將選取的工程師名稱轉換為字符串格式
        engineers_ranked_str = ",".join(engineers_ranked)
        
        # 儲存所有相關信息到結果列表中
        data = {
            '經費規模': output['經費規模'].iloc[idx],
            '評估工時': output['評估工時'].iloc[idx],
            '難中易': output['難中易'].iloc[idx],
            '專案類型': project_type,
            '年資': output['年資'].iloc[idx],
            '協作人數': cooperation_num,
            '新進工程師人數': new_count,
            '資深工程師人數': old_count,
            '預估工時(天)': output['預估工時(天)'].iloc[idx],
            '專案成效': output['成效分數'].iloc[idx],
            '工程師類型': output['工程師類型'].iloc[idx],
            '工程師': engineers_ranked_str
        }
        results.append(data)
    
    # 將結果列表轉換為DataFrame格式
    rank = pd.DataFrame(results)
    # 將預估工時和專案成效進行四捨五入
    rank['預估工時(天)'] = rank['預估工時(天)'].round(1)
    rank['專案成效'] = rank['專案成效'].round(1)
    
    # 返回排序後的工程師DataFrame
    return rank

def engineer_ranking(output,rank_score_list,project_file_pattern, project_engineer_pattern, engineer_pattern):
    project,engineer_data = predata.load_data(project_file_pattern, project_engineer_pattern,engineer_pattern)
    data,project = predata.preprocess_effectiveness(project)
    engineer_df = predata.engineer_df(data,engineer_data)
    rank = rank_engineers(output, engineer_df,rank_score_list)
    return rank