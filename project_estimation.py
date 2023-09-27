# coding: utf-8
import pandas as pd
import numpy as np
import re
import glob
import torch
from createmodel import Neuron, Network
import preprocessdata as predata
import os
import joblib
import random
os.environ["OMP_NUM_THREADS"] = "1"

# 定義專案估計類別
class ProjectEstimation:
    def __init__(self, project_file_pattern, project_engineer_pattern, model_path,input_scale_file_path,output_scale_file_path):
        # 初始化時，載入專案、工程師資料和機器學習模型
        self.project = self.load_data(project_file_pattern, project_engineer_pattern)
        self.project = self.project.fillna(False)
        self.project = self.project.drop("專案成效", axis=1)
        self.model = Network()
        self.model.load_state_dict(torch.load(model_path))
        self.scaler_features = joblib.load(input_scale_file_path)
        self.scaler_targets = joblib.load(output_scale_file_path)
    # 計算新進工程師的比例
    @staticmethod
    def calculate_ratio(engineers, engineer_dict):
        # 使用 numpy 分割工程師名單
        engineer_series = np.array(engineers.split(','))
        # 使用 numpy 判斷哪些工程師是新進工程師
        new_engineers = np.isin(engineer_series, [engineer for engineer in engineer_dict if engineer_dict[engineer] <= 3]).sum()
        # 計算新進工程師的比例
        ratio = new_engineers / len(engineer_series) if len(engineer_series) != 0 else 0
        return ratio

    # 計算平均經費規模
    @staticmethod
    def calculate_average_salary(salary_range_str):
        # 使用 numpy 分割經費規模範圍
        salary_range = np.array(salary_range_str.split('-'))
        # 使用 numpy 與正則表達式過濾出數字部分
        filtered_salary_range = np.array([int(s) if s else 0 for s in [re.sub(r'\D', '', salary) for salary in salary_range]])
        # 使用 numpy 計算平均經費規模
        average_salary = filtered_salary_range.mean() + random.randint(-10,10)
        return average_salary

    # 預處理舊的專案和工程師資料
    @staticmethod
    def preprocess_old_data(project, engineer):
        # 刪除不必要的欄位
        project = project.drop("專案起訖時間", axis=1)
        # 計算平均經費規模
        project['經費規模'] = project['經費規模'].apply(ProjectEstimation.calculate_average_salary)
        # 定義難易度映射
        difficulty_mapping = {'易': 1, '中': 2, '難': 3}
        # 轉換難易度為數字
        project['難中易'] = project['難中易'].map(difficulty_mapping)
        # 對專案類型進行 one-hot 編碼
        project = pd.get_dummies(project, columns=['專案類型'])
        # 移除工程師名稱中的 "工程師" 字樣
        engineer['工程師'] = engineer['工程師'].str.replace('工程師', '')
        # 創建工程師字典，其中包含工程師名稱和他們的平均年資
        engineer_dict = engineer[['工程師', '單位(年)']].dropna()
        engineer_dict['單位(年)'] = engineer_dict['單位(年)'].apply(
            lambda x: int((int(x.split('-')[0]) + int(x.split('-')[1])) / 2))
        engineer_dict = dict(engineer_dict.values)
        # 合併工程師資料至專案資料
        project = pd.concat([project, engineer["負責工程師"], engineer["是否有團隊協作"]], axis=1)
        # 計算新進工程師比例
        project['新進工程師比例'] = project['負責工程師'].apply(lambda x: ProjectEstimation.calculate_ratio(x, engineer_dict))
        # 將負責工程師欄位轉換為平均年資
        project['負責工程師'] = project['負責工程師'].str.split(',').apply(
            lambda x: np.mean([engineer_dict[engineer] for engineer in x]))
        project = project.rename(columns={'負責工程師': '平均年資'})
        return project
    #預處理輸入資料的方法。
    def preprocess_input(self, input_data):
        # 將難易度轉換為數字表示
        difficulty_dict = {'易': 1, '中': 2, '難': 3}
        input_data[2] = difficulty_dict.get(input_data[2], input_data[2])
        
        # 計算平均經費規模
        input_data[0] = ProjectEstimation.calculate_average_salary(input_data[0])
        
        # 獲取專案資料的欄位名稱列表
        project_column_names = self.project.columns.tolist()
        
        # 編碼專案類型
        project_type = "專案類型_" + input_data[3]
        encoded_project_type = [int(project_type == col_name) for col_name in project_column_names[4:-3]]
        
        # 將輸入資料進行編碼
        preprocessed_input = input_data[:3] + encoded_project_type + input_data[-2:]
        preprocessed_input = preprocessed_input[:-2]
        
        return preprocessed_input
    # 讀取專案和工程師資料
    @staticmethod
    def load_data(project_file_pattern, project_engineer_pattern):
        # 使用 glob 找到所有匹配的文件
        project_files = glob.glob(project_file_pattern)
        engineer_files = glob.glob(project_engineer_pattern)
        projects = []
        # 對每對專案和工程師文件進行預處理，然後將它們加入到專案列表中
        for project_file, engineer_file in zip(project_files, engineer_files):
            project = pd.read_csv(project_file)
            engineer = pd.read_csv(engineer_file)
            project = ProjectEstimation.preprocess_old_data(project, engineer)
            projects.append(project)
        # 合併所有專案
        return pd.concat(projects, axis=0)

    # 估算專案
    def estimate(self, inputProject, lower, upper):
        
        # 預處理輸入資料
        inputProject = self.preprocess_input(inputProject)
        new_Ratio = 0
        exp_level = 1
        Teamworker = 1
        
        # 準備輸入資料
        inputProject = np.array(inputProject + [exp_level, Teamworker, new_Ratio, 0, 0, 0], dtype=float)
        input_length = len(inputProject)
        
        # 使用 numpy 生成所有可能的組合
        types = np.arange(10, 13)
        years = np.arange(lower[0], upper[0] + 1, 2)
        collaborators = np.arange(lower[1], upper[1] + 1, 1)
        ratios = np.linspace(0.1, 1, 10)
        
        combinations = np.array(np.meshgrid(types, years, collaborators, ratios)).T.reshape(-1, 4)
        expanded_input = np.tile(inputProject, (combinations.shape[0], 1))
        
        expanded_input[:, 7] = combinations[:, 1]
        expanded_input[:, 8] = combinations[:, 2]
        expanded_input[:, 9] = combinations[:, 3]
        
        # 對於每個類型設置正確的位置為 1
        for idx, type_val in enumerate(combinations[:, 0].astype(int)):
            expanded_input[idx, input_length - 3 + type_val - 10] = 1
        
        # 使用模型一次性進行預測
        input = self.scaler_features.transform(expanded_input)
        output_tensors,_,_,_,_ = self.model(torch.tensor(input, dtype=torch.float))
        output_tensors = self.scaler_targets.inverse_transform(output_tensors.detach().numpy())
        effectiveness_scores = np.abs(output_tensors[:, 4])
        dev_times = np.abs(output_tensors[:, 5])
        # 過濾結果，只保留在指定範圍內的開發時間
        valid_indices = np.where((dev_times < inputProject[1] + upper[3]) & (dev_times > 0.1))
        filtered_outputs = expanded_input[valid_indices]
        filtered_scores = effectiveness_scores[valid_indices]
        filtered_times = dev_times[valid_indices]
        # 創建輸出 DataFrame
        columns = ["經費規模", '評估工時', '難中易', '專案類型_APP開發', '專案類型_客製化專案',
                   '專案類型_維護案', '專案類型_網站開發', '年資', '協作人數', '新進工程師比例', 
                   '成效分數', '預估工時(天)', '工程師類型_0', '工程師類型_1', '工程師類型_2']
        output = pd.DataFrame(columns=columns)
        output[columns[:10]] = filtered_outputs[:, :10]
        output['成效分數'] = filtered_scores
        output['預估工時(天)'] = filtered_times
        output[columns[12:]] = filtered_outputs[:, -3:]
        if output.shape==(0,15):
            return output
        # 轉換 one-hot 編碼回原始類別
        one_hot_columns_auto = [col for col in output.columns if col.startswith("專案類型_")]
        output['專案類型'] = output[one_hot_columns_auto].idxmax(axis=1).astype(str).str.split("_").str[1]
        output.drop(columns=one_hot_columns_auto, inplace=True)
        
        one_hot_columns_auto2 = [col for col in output.columns if col.startswith("工程師類型_")]
        output['工程師類型'] = output[one_hot_columns_auto2].idxmax(axis=1).astype(str).str.split("_").str[1]
        output.drop(columns=one_hot_columns_auto2, inplace=True)
        return output


# 估算專案的主函數
def estimate_project(inputProject, project_file_pattern, project_engineer_pattern,model_path,input_scale_file_path,output_scale_file_path,lower, upper):
    estimator = ProjectEstimation(project_file_pattern, project_engineer_pattern, model_path,input_scale_file_path,output_scale_file_path)
    output = estimator.estimate(inputProject, lower, upper)
    return output
