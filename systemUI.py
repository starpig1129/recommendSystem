# 引入必要的模組
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import json
import pandas as pd
import warnings
import shutil
import webbrowser

# 引入自定義模組
import project_estimation as pe
import engineer_rank as er
from createmodel import create_model

# 設定 Flask 應用
app = Flask(__name__)

# 路徑設定
project_file_pattern = './data/專案紀錄*.csv'
project_engineer_pattern = './data/工程師資料*.csv'
engineer_pattern = './data/engineer.csv'
MODEL_FOLDER = os.path.join(os.getcwd(), "model")

# 忽略 sklearn 警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
#載入資料函數
def load_settings():
    if os.path.exists('setting.json'):
        with open('setting.json', 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    else:
        return {}

@app.route('/')
def index():
    # 從 setting.json 中讀取使用者先前的設定值，如果檔案不存在則使用預設值
    saved_data = load_settings()
    return render_template('recommendation_system.html',
                           budget=saved_data.get('budget', ''),
                           hours=saved_data.get('hours', ''),
                           difficulty=saved_data.get('difficulty', ''),
                           project_type=saved_data.get('project_type', ''),
                           orderedList=saved_data.get('orderedList', ''),
                           collaborator_max=saved_data.get('collaborator_max', ''),
                           collaborator_min=saved_data.get('collaborator_min', ''),
                           experience_max=saved_data.get('experience_max', ''),
                           experience_min=saved_data.get('experience_min', ''),
                           newengineer_ratio_max=saved_data.get('newengineer_ratio_max', '10'),
                           newengineer_ratio_min=saved_data.get('newengineer_ratio_min', '0'),
                           hours_exceed_max=saved_data.get('hours_exceed_max', ''),
                           experience=saved_data.get('experience', ''),
                           expertise=saved_data.get('expertise', ''),
                           preference=saved_data.get('preference', ''))

@app.route('/submit_project_data', methods=['POST'])
def submit_project_data():
    # 從 setting.json 中讀取先前的設定，如果檔案不存在則使用預設值
    saved_data = load_settings()
    # 從表單中收集資料，並更新 saved_data 字典
    saved_data['budget'] = request.form.get('budget')
    saved_data['hours'] = request.form.get('hours')
    saved_data['difficulty'] = request.form.get('difficulty')
    saved_data['project_type'] = request.form.get('project_type')
    saved_data['orderedList'] = request.form.get('orderedList')
    (saved_data['collaborator_max']) = request.form.get('collaborator-max')
    saved_data['collaborator_min'] = request.form.get('collaborator-min')
    if float(saved_data['collaborator_min']) >= float(saved_data['collaborator_max']):
        saved_data['collaborator_max'] = int(saved_data['collaborator_min']) +2
    saved_data['experience_max'] = request.form.get('experience-max')
    saved_data['experience_min'] = request.form.get('experience-min')
    if float(saved_data['experience_min']) >= float(saved_data['experience_max']):
        saved_data['experience_max'] = float(saved_data['experience_min']) +2
    saved_data['newengineer_ratio_max'] = request.form.get('new-engineer-ratio-max')
    saved_data['newengineer_ratio_min'] = request.form.get('new-engineer-ratio-min')
    saved_data['hours_exceed_max'] = request.form.get('hours-exceed-max')
    saved_data['similarity'] = 1  # 固定值
    saved_data['experience'] = request.form.get('experience')
    saved_data['expertise'] = request.form.get('expertise')
    saved_data['preference'] = request.form.get('preference')
    
    # 將更新後的 saved_data 保存到 setting.json
    with open('setting.json', 'w') as json_file:
        json.dump(saved_data, json_file, indent=4)

    # 使用 saved_data 設定其他必要變數
    budget = saved_data['budget']
    hours = saved_data['hours']
    difficulty = saved_data['difficulty']
    project_type = saved_data['project_type']
    orderedList = saved_data['orderedList']
    collaborator_max = saved_data['collaborator_max']
    collaborator_min = saved_data['collaborator_min']
    experience_max = saved_data['experience_max']
    experience_min = saved_data['experience_min']
    hours_exceed_max = saved_data['hours_exceed_max']
    similarity = saved_data['similarity']
    experience = saved_data['experience']
    expertise = saved_data['expertise']
    preference = saved_data['preference']
    model = saved_data['model']
    model_file_pattern = saved_data['model_file_pattern']
    input_scale_file_pattern = saved_data['input_scale_file_pattern']
    output_scale_file_pattern = saved_data['output_scale_file_pattern']
    inputProject = [budget, hours, difficulty, project_type]
    rank_score_list = [similarity, experience, expertise, preference]
    sorted_list = orderedList.split(',')

    # 設定下限與上限
    lower = [float(experience_min), float(collaborator_min), float(0)]
    upper = [float(experience_max), float(collaborator_max), float(10), float(hours_exceed_max)]

    # 使用工具估算專案
    estimate_project = pe.estimate_project(inputProject, project_file_pattern, project_engineer_pattern, model_file_pattern,
                                           input_scale_file_pattern, output_scale_file_pattern, lower, upper)
    
    # 進行工程師排名
    if estimate_project.shape == (0, 15):
        columns = ['協作人數', '新進工程師人數', '資深工程師人數', '預估工時(天)', '專案成效', '工程師']
        output = pd.DataFrame(columns=columns)
        output[:] = "無結果"
    else:
        rank = er.engineer_ranking(estimate_project, rank_score_list, project_file_pattern, project_engineer_pattern, engineer_pattern)
        output = rank.reset_index(drop=True)
        if not sorted_list:
            sorted_list = ['預估工時(天)', '協作人數', '專案成效', '新進工程師人數', '資深工程師人數']
        output = output.sort_values(by=sorted_list,ascending=[True if i != '專案成效' else False for i in sorted_list])
        output = output[output.apply(lambda row: len(row['工程師'].split(',')) == row['協作人數'], axis=1)]
        if output.shape[1] > 5:
            output = output.drop_duplicates(subset=[sorted_list[0]])
        output = output.loc[:, ['協作人數', '新進工程師人數', '資深工程師人數', '預估工時(天)', '專案成效', '工程師']]
        output = output.drop_duplicates(subset=["工程師"])
        output = output.reset_index(drop=True)
        
    return render_template('recommendation_system.html',
                           data=request.form,
                           tables=[output.to_html(classes='data')],
                           titles=output.columns.values,
                           budget=saved_data.get('budget', ''),
                           hours=saved_data.get('hours', ''),
                           difficulty=saved_data.get('difficulty', ''),
                           project_type=saved_data.get('project_type', ''),
                           orderedList=saved_data.get('orderedList', ''),
                           collaborator_max=saved_data.get('collaborator_max', ''),
                           collaborator_min=saved_data.get('collaborator_min', ''),
                           experience_max=saved_data.get('experience_max', ''),
                           experience_min=saved_data.get('experience_min', ''),
                           hours_exceed_max=saved_data.get('hours_exceed_max', ''),
                           experience=saved_data.get('experience', ''),
                           expertise=saved_data.get('expertise', ''),
                           preference=saved_data.get('preference', ''))

@app.route('/engineer_update')
def engineer_update():
    # 處理工程師資料更新的路由
    return render_template('engineer_update.html')

@app.route('/data/engineer.csv')
def serve_csv():
    # 提供工程師的 csv 檔案
    return send_from_directory('data', 'engineer.csv')

@app.route('/update_csv', methods=['POST'])
def update_csv():
    # 更新工程師的 csv 檔案
    csv_data = request.data.decode('utf-8')
    try:
        with open('data/engineer.csv', 'w', encoding='utf-8') as f:
            f.write(csv_data)
        return jsonify({'message': 'CSV 更新成功！'})
    except Exception as e:
        return jsonify({'message': f"更新失敗，原因：{str(e)}"})


@app.route('/model_update')
def model_update():
    # 處理模型更新的路由
    subfolders_in_model = [d for d in os.listdir(MODEL_FOLDER) if os.path.isdir(os.path.join(MODEL_FOLDER, d))]
    saved_data = load_settings()
    current_model = saved_data['model']
    return render_template('model_update.html', settings=saved_data, folders=subfolders_in_model, current_model=current_model)

@app.route('/select_folder')
def select_folder():
    # 處理選擇模型資料夾的路由
    saved_data = load_settings()
    model = request.args.get('folder_name')
    saved_data['model'] = model
    saved_data['model_file_pattern'] = './model/' + model + '/model.pth'
    saved_data['input_scale_file_pattern'] = './model/' + model + '/feater_scaler.pkl'
    saved_data['output_scale_file_pattern'] = './model/' + model + '/target_scaler.pkl'
    with open('setting.json', 'w') as json_file:
        json.dump(saved_data, json_file, indent=4)

    return f"Selected folder: {model}"

@app.route('/create-model', methods=['POST'])
def create_model_buttom():
    # 處理創建模型的路由
    data = request.json
    saved_data = {}
    saved_data = load_settings()
    
    if 'modelName' in data:
        saved_data['模型名稱'] = data['modelName']
    if 'fundingScale' in data:
        saved_data['經費規模'] = data['fundingScale']
    if 'estimatedHours' in data:
        saved_data['評估工時'] = data['estimatedHours']
    if 'difficulty' in data:
        saved_data['難中易'] = data['difficulty']
    if 'collaboration' in data:
        saved_data['是否有團隊協作'] = data['collaboration']
    if 'seniority' in data:
        saved_data['年資'] = data['seniority']
    if 'generateData' in data:
        saved_data['generateData'] = data['generateData']
    model = data['modelName']
    saved_data['model_file_pattern'] = './model/' + model + '/model.pth'
    saved_data['input_scale_file_pattern'] = './model/' + model + '/feater_scaler.pkl'
    saved_data['output_scale_file_pattern'] = './model/' + model + '/target_scaler.pkl'
    checkbox_order = ['fundingScale_checked','estimatedHours_checked','difficulty_checked', 'seniority_checked', 'collaboration_checked']
    checkbox_states_list = [data.get(checkbox, False) for checkbox in checkbox_order]
    saved_data['checkbox_states'] = checkbox_states_list
    with open('setting.json', 'w', encoding='utf-8') as json_file:
        json.dump(saved_data, json_file, indent=4)
    new_environment_correlations = {
        '經費規模': float(data['fundingScale']),
        '評估工時': float(data['estimatedHours']),
        '難中易': float(data['difficulty']),
        '平均年資': float(data['seniority']),
        '是否有團隊協作': float(data['collaboration']),
    }
    print('checkbox_states_list=',checkbox_states_list)
    keys_order = ['經費規模', '評估工時', '難中易', '平均年資', '是否有團隊協作']
    # 根據checkbox_states_list移除new_environment_correlations字典中的項目
    for i, state in enumerate(checkbox_states_list):
        if state:
            del new_environment_correlations[keys_order[i]]
    gen_data = saved_data.get('generateData')
    model = data['modelName']
    model_file_pattern = saved_data.get('model_file_pattern', './model/' + model + '/model.pth')
    input_scale_file_pattern = saved_data.get('input_scale_file_pattern', './model/' + model + '/feater_scaler.pkl')
    output_scale_file_pattern = saved_data.get('output_scale_file_pattern', './model/' + model + '/target_scaler.pkl')
    create_model(gen_data, new_environment_correlations,
                 project_file_pattern,
                 project_engineer_pattern,
                 engineer_pattern,
                 model_file_pattern,
                 input_scale_file_pattern,
                 output_scale_file_pattern)
    return jsonify({"message": "模型已成功創建！"})

@app.route('/select_model', methods=['POST'])
def select_model():
    # 處理選擇模型的路由
    data = request.json
    model = data['selectedModel']
    
    saved_data = load_settings()
    saved_data['model'] = model
    saved_data['model_file_pattern'] = './model/' + model + '/model.pth'
    saved_data['input_scale_file_pattern'] = './model/' + model + '/feater_scaler.pkl'
    saved_data['output_scale_file_pattern'] = './model/' + model + '/target_scaler.pkl'
    with open('setting.json', 'w', encoding='utf-8') as json_file:
        json.dump(saved_data, json_file, indent=4)
    return jsonify({"message": "选中的模型已保存！"})

@app.route('/get-model-data', methods=['GET'])
def get_model_data():
    # 返回目前的模型設定數據
    saved_data = load_settings()
    return jsonify(saved_data)

@app.route('/get_current_model')
def get_current_model():
    # 返回目前選擇的模型名稱
    saved_data = load_settings()
    return jsonify({'current_model': saved_data['model']})

@app.route('/edit_folder')
def edit_folder():
    # 修改模型資料夾名稱
    old_name = request.args.get('old_name')
    new_name = request.args.get('new_name')
    os.rename(os.path.join(MODEL_FOLDER, old_name), os.path.join(MODEL_FOLDER, new_name))
    return "Folder name updated successfully."

@app.route('/delete_folder')
def delete_folder():
    # 刪除指定模型資料夾
    folder_name = request.args.get('folder_name')
    shutil.rmtree(os.path.join(MODEL_FOLDER, folder_name))
    return "Folder deleted successfully."

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=False)
