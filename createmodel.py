# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocessdata as predata
import data_generator as dg
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mutual_info_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

# 定義神經元類別
class Neuron(nn.Module):
    def __init__(self, input_dim, output_dim, threshold_function=None):
        """
        初始化神經元
        :param input_dim: 輸入維度
        :param output_dim: 輸出維度
        :param threshold_function: 閾值函數
        """
        super(Neuron, self).__init__()
        self.input_dim = input_dim 
        # 定義全連接層
        self.fc = nn.Linear(input_dim, output_dim)
        # 使用者自定義的閾值函數
        self.threshold_function = threshold_function
    def forward(self, x):
        """
        前向傳播
        
        :param x: 輸入資料
        :return: 輸出結果
        """
        x = self.fc(x)
        if self.threshold_function:
            x = self.threshold_function(x)
        return x
class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        
        # 非線性轉換
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        
        # 線性轉換
        self.fc2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
#輔助loss
class OutputDecoder(nn.Module):
    def __init__(self, dim):
        super(OutputDecoder, self).__init__()
        
        # 非線性轉換
        self.fc1 = nn.Linear(dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x
#建立網路
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.Neuron_layer0 = 13#輸入
        self.Neuron_layer1 = 13
        self.Neuron_layer2 = 7#額外loss
        self.Neuron_layer3 = 10
        self.Neuron_layer4 = 5#額外loss
        self.Neuron_layer5 = 10
        self.Neuron_layer6 = 2
        # 定義每層的神經元
        self.layer0 = nn.ModuleList([Neuron(1, 1) for _ in range(self.Neuron_layer0)])
        self.layer1 = nn.ModuleList([Neuron(10, 10) for _ in range(self.Neuron_layer1)])
        self.layer2 = nn.ModuleList([Neuron(10, 10) for _ in range(self.Neuron_layer2)])
        self.layer3 = nn.ModuleList([Neuron(10, 10) for _ in range(self.Neuron_layer3)])
        self.layer4 = nn.ModuleList([Neuron(10, 10) for _ in range(self.Neuron_layer4)])
        self.layer5 = nn.ModuleList([Neuron(10, 10) for _ in range(self.Neuron_layer5)])
        self.layer6 = nn.ModuleList([Neuron(10, 1) for _ in range(self.Neuron_layer6)])
        self.decoder = Decoder(130)
        self.decoder4 = Decoder(50)
        self.l2_outputs0_layer = OutputDecoder(10)
        self.l4_outputs0_layer = OutputDecoder(10)
        self.l4_outputs1_layer = OutputDecoder(10)
        self.l4_outputs2_layer = OutputDecoder(10)
        # 定義連接
        self.connections = {
            0: {#經費規模
                1: (self.layer1, [0,1,2,3,10]),
                2: (self.layer4, [1]),
                3: (self.layer6, [0, 1])
            },
            1: {#評估工時
                1: (self.layer1, [0,1,2,3,10]),
                2: (self.layer4, [1]),
                3: (self.layer6, [0,1])
            },
            2: {#難中易
                1: (self.layer1, [0,1,2,3,7,9,10]),
                2: (self.layer2, [3]),
                3: (self.layer6, [0, 1])
            },
            3: {#是否有團隊協作
                1: (self.layer1, [0,1,2,3,4,10]),
                2: (self.layer4, [1]),
                3: (self.layer6, [0, 1])
            },
            4: {#新進工程師比例
                1: (self.layer1, [3,4,5,9,10])
            },
            5: {#平均年資
                1: (self.layer1, [4,5,6,9])
            },
            6: {#專案類型_APP開發
                1: (self.layer1, [5,6,7])
            },
            7: {#專案類型_客製化專案
                1: (self.layer1, [2,6,7,8,9]),
                2: (self.layer2, [3]),
                3: (self.layer6, [1])
            },
            8: {#專案類型_維護案
                1: (self.layer1, [7,8])
            },
            9: {#專案類型_網站開發
                1: (self.layer1, [2,4,5,7,9])
            },
            10: {#工程師類型_0
                1: (self.layer1, [0,1,2,3,4,10]),
                2: (self.layer4, [1]),
                3: (self.layer6, [0,1])
            },
            11: {#工程師類型_1
                1: (self.layer1, [7]),
                2: (self.layer2, [3]),
                3: (self.layer4, [0]),
                4: (self.layer6, [1])
            },
            12: {#工程師類型_2
                1: (self.layer1, [12])
            }
        }
        # 為每個連結定義權重
        self.weights = {}
        for src, targets in self.connections.items():
            self.weights[src] = {}
            for dst, (layer, indices) in targets.items():
                weight_name = f"weight_{src}_{dst}"
                setattr(self, weight_name, nn.Parameter(torch.ones(len(indices))))
                self.weights[src][dst] = getattr(self, weight_name)
    def forward(self, x):
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        mish = torch.nn.functional.mish
        # 第0層的輸出
        layer0_outputs = [neuron(x[:, i:i+1]) for i, neuron in enumerate(self.layer0)]
        # 動態產生各層的輸入
        layer1_inputs = [torch.zeros(x.size(0), neuron.input_dim).to(x.device) for neuron in self.layer1]
        layer2_inputs = [torch.zeros(x.size(0), neuron.input_dim).to(x.device) for neuron in self.layer2]
        layer3_inputs = [torch.zeros(x.size(0), neuron.input_dim).to(x.device) for neuron in self.layer3]
        layer4_inputs = [torch.zeros(x.size(0), neuron.input_dim).to(x.device) for neuron in self.layer4]
        layer5_inputs = [torch.zeros(x.size(0), neuron.input_dim).to(x.device) for neuron in self.layer5]
        layer6_inputs = [torch.zeros(x.size(0), neuron.input_dim).to(x.device) for neuron in self.layer6]

        # 根據第0層的輸出更新其他層的輸入
        for i, outputs in enumerate(layer0_outputs):
            for key, (layer, indices) in self.connections[i].items():
                for idx, index in enumerate(indices):
                    weight = self.weights[i][key][idx]
                    if layer == self.layer1:
                        layer1_inputs[index] += weight * outputs
                    elif layer == self.layer2:
                        layer2_inputs[index] += weight * outputs
                    elif layer == self.layer4:
                        layer4_inputs[index] += weight * outputs
                    elif layer == self.layer6:
                        layer6_inputs[index] += weight * outputs

        # 計算第1層的輸出
        layer1_outputs = [leaky_relu(neuron(inputs)) for neuron, inputs in zip(self.layer1, layer1_inputs)]

        # 使用 autoencoder
        layer1_inputs_flattened = torch.cat(layer1_inputs, dim=1)
        layer1_outputs_flattened = torch.cat(layer1_outputs, dim=1)
        decoded = self.decoder(layer1_outputs_flattened)

        # 計算第二層的輸入
        for output in layer1_outputs:
            for i in range(len(layer2_inputs)):
                layer2_inputs[i] += output

        # 計算第二層的輸出
        layer2_outputs = [leaky_relu(neuron(inputs)) for neuron, inputs in zip(self.layer2, layer2_inputs)]

        # 計算第三層的輸入
        for output in layer2_outputs:
            for i in range(len(layer3_inputs)):
                layer3_inputs[i] += output
        # 計算第三層的輸出
        layer3_outputs = [leaky_relu(neuron(inputs)) for neuron, inputs in zip(self.layer3, layer3_inputs)]

        # 計算第四層的輸入
        for output in layer3_outputs:
            for i in range(len(layer4_inputs)):
                layer4_inputs[i] += output
        # 計算第四層的輸出
        layer4_outputs = [leaky_relu(neuron(inputs)) for neuron, inputs in zip(self.layer4, layer4_inputs)]

        # 使用 autoencoder
        layer4_inputs_flattened = torch.cat(layer4_inputs, dim=1)
        layer4_outputs_flattened = torch.cat(layer4_outputs, dim=1)
        decoded4 = self.decoder4(layer4_outputs_flattened)

        # 計算第五層的輸入
        for output in layer4_outputs:
            for i in range(len(layer5_inputs)):
                layer5_inputs[i] += output
        # 計算第五層的輸出
        layer5_outputs = [leaky_relu(neuron(inputs)) for neuron, inputs in zip(self.layer5, layer5_inputs)]
        # 計算第六層的輸入
        for output in layer5_outputs:
            for i in range(len(layer6_inputs)):
                layer6_inputs[i] += output
        # 計算第六層的輸出        
        layer6_outputs = [leaky_relu(neuron(inputs)) for neuron, inputs in zip(self.layer6, layer6_inputs)]

        layer2_output = self.l2_outputs0_layer(layer2_outputs[3])
        layer4_output0 = self.l4_outputs0_layer(layer4_outputs[0])
        layer4_output1 = self.l4_outputs1_layer(layer4_outputs[1])
        layer4_output2 = self.l4_outputs2_layer(layer4_outputs[2])

        
        # 組合所有的輸出
        combined_outputs = [
            layer2_output, 
            layer4_output0, 
            layer4_output1, 
            layer4_output2, 
            layer6_outputs[0], 
            layer6_outputs[1]
        ]

        # 重新組合所有平均輸出
        output = torch.cat(combined_outputs, dim=1)
        return output,decoded, layer1_inputs_flattened,decoded4,layer4_inputs_flattened


def create_model(gen_data,new_environment_correlations,project_file_pattern,project_engineer_pattern,engineer_pattern,model_file_pattern,input_scale_file_pattern,output_scale_file_pattern):
    project,engineer_data = predata.load_data(project_file_pattern, project_engineer_pattern,engineer_pattern)
    data,project = predata.preprocess_effectiveness(project)
    engineer_df = predata.engineer_df(data,engineer_data)
    project = predata.project2project_engineer(project,engineer_df)
    if gen_data :
        project = dg.gendata(project,new_environment_correlations)
    #cuda
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    #分出標籤
    features = project.drop(["成效分數", "開發工時", "專案工時差", "需求變更穩定度分數", "需求明確度分數", "專案工時分數"], axis=1).astype(float)
    order = [
        '經費規模', '評估工時', '難中易', '是否有團隊協作', '新進工程師比例', '平均年資', '專案類型_APP開發', 
        '專案類型_客製化專案', '專案類型_維護案', '專案類型_網站開發', '工程師類型_0', '工程師類型_1', '工程師類型_2'
    ]
    features = features[order]
    targets = project[["專案工時差","需求變更穩定度分數","需求明確度分數","專案工時分數","成效分數", "開發工時"]].astype(float)
    print('features=',features.shape)
    print('targets',targets.shape)

    #切割資料
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2)
    #正規化
    scaler_features = MinMaxScaler()
    features_train = scaler_features.fit_transform(features_train)
    features_test = scaler_features.transform(features_test)

    #正規化
    scaler_targets = MinMaxScaler()
    targets_train = scaler_targets.fit_transform(targets_train)
    targets_test = scaler_targets.transform(targets_test)

    #轉為tensor
    features_train = torch.tensor(features_train, dtype=torch.float).to(device)
    features_test = torch.tensor(features_test, dtype=torch.float).to(device)
    targets_train = torch.tensor(targets_train, dtype=torch.float).to(device)
    targets_test = torch.tensor(targets_test, dtype=torch.float).to(device)

    #確定資料維度
    print("Features train shape:", features_train.shape)
    print("Targets train shape:", targets_train.shape)
    print("Features test shape:", features_test.shape)
    print("Targets test shape:", targets_test.shape)
    # 創建網絡實例並執行
    network = Network().to(device)
    #設定參數
    best_loss = float('inf')
    stop_counter = 0
    early_stop_threshold = 100  # 決定何時停止訓練的閾值
    lr = 0.01
    num_epochs = 2000
    loss_fn = nn.SmoothL1Loss(beta=0.75)  # 損失函數
    optimizer = torch.optim.RMSprop(network.parameters(), lr=lr, alpha=0.9, eps=1e-8)
    # 初始化學習率調度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=80, verbose=True)
    # 訓練迴圈
    losses = []
    loss_transmission_hours = []
    loss_stability_score = []
    loss_clarity_score = []
    loss_project_hours_score = []
    loss_effectiveness_score = []
    loss_development_hours = []
    for epoch in range(num_epochs):
        # 向前傳遞
        outputs, decoded, original ,decoded4,original4 = network(features_train)
        
        # 定義每個損失的權重
        loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0]).to(device)

        # 計算主要的損失
        main_loss = torch.sum(loss_weights * loss_fn(outputs, targets_train))
        # 計算autoencoder的損失
        reconstruction_loss = loss_fn(decoded, original)*1

        reconstruction_loss4 = loss_fn(decoded4, original4)*1
        # 組合這兩個損失
        loss = main_loss + reconstruction_loss + reconstruction_loss4
        # 向後傳遞和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #轉換回原尺度
        outputs_original_scale = scaler_targets.inverse_transform(outputs.detach().cpu().numpy())
        targets_train_original_scale = scaler_targets.inverse_transform(targets_train.detach().cpu().numpy())
        #計算MAE在原尺度
        mae_each_target = np.mean(np.abs(outputs_original_scale - targets_train_original_scale), axis=0)

        # 保存loss
        losses.append(np.mean(mae_each_target))
        loss_transmission_hours.append(mae_each_target[0])
        loss_stability_score.append(mae_each_target[1])
        loss_clarity_score.append(mae_each_target[2])
        loss_project_hours_score.append(mae_each_target[3])
        loss_effectiveness_score.append(mae_each_target[4])
        loss_development_hours.append(mae_each_target[5])
        
        # 打印每10個epoch的損失值
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {[float(torch.sqrt(loss_fn(outputs[:, i], targets_train[:, i]))) for i in range(6)]}')
            print(f'Epoch {epoch+1}/{num_epochs}, DecoderLoss:{reconstruction_loss}')
            print(f'Epoch {epoch+1}/{num_epochs}, Decoder4Loss:{reconstruction_loss4}')
        # 在驗證集上計算損失
        with torch.no_grad():
            val_outputs, _, _, _, _ = network(features_test)
            val_loss = torch.sum(loss_weights * loss_fn(val_outputs, targets_test))
        scheduler.step(val_loss)
        # 每次迭代後保存最佳模型
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            # 如果沒有目標資料夾則創建
            os.makedirs(os.path.dirname(model_file_pattern), exist_ok=True)
            torch.save(network.state_dict(), model_file_pattern)
        # 檢查是否需要早停
        if val_loss.item() > best_loss:
            stop_counter += 1
        else:
            best_loss = val_loss.item()
            stop_counter = 0
        # 如果連續多個epoch損失沒有改善，停止訓練
        if stop_counter >= early_stop_threshold:
            print("Early stopping!,epoch=",epoch)
            break
    with torch.no_grad():
        outputs_test, _, _, _, _ = network(features_test)
    # 將正規化的預測轉換回原始尺度
    outputs_test_original_scale = scaler_targets.inverse_transform(outputs_test.detach().cpu().numpy())

    # 將正規化的目標轉換回原始尺度
    targets_test_original_scale = scaler_targets.inverse_transform(targets_test.detach().cpu().numpy())

    # 將預測和目標保存到一個DataFrame中
    df_results = pd.DataFrame(outputs_test_original_scale, columns=[f"Pred_{i}" for i in range(outputs_test_original_scale.shape[1])])
    for i in range(targets_test_original_scale.shape[1]):
        df_results[f"True_{i}"] = targets_test_original_scale[:, i]
    # 計算每個數據點的誤差
    errors = np.abs(outputs_test_original_scale - targets_test_original_scale)
    mean_errors = np.mean(errors, axis=0)
    print('mean_errors = ',mean_errors)
    # 保存 scaler
    joblib.dump(scaler_features, input_scale_file_pattern)
    joblib.dump(scaler_targets, output_scale_file_pattern)
