# v7.4
loss的beta參數調整為0.75

##  安裝所需的套件：
所有需要套件都列在 `requirements.txt` 文件中。要安裝，執行：
    ```bash
    pip install -r requirements.txt
    ```

## 運行
    執行主程式：
    ```bash
    python systemUI.py
    ```
執行後，應用程式會自動在瀏覽器中打開，或者手動訪問 http://127.0.0.1:5000/。
## 項目結構和主要文件

- `createmodel.py`: 創建模型。
- `data_generator.py`: 生成數據。
- `engineer_rank.py`: 基於各種指標對工程師進行排名。
- `preprocessdata.py`: 數據預處理。
- `project_estimation.py`: 估算項目時間表和成效。
- `systemUI.py`: 主程式。
- `data/`: 包含像 `engineer.csv` 數據文件的目錄。

## 功能說明
首頁:

在首頁，可輸入專案的詳細資訊，如預算、工時、難度等，並獲得推薦的工程師列表。

## 工程師更新:

更新工程師的資料。
## 模型更新

要更新模型，按照以下步驟操作：

1. 收集新數據並放在 `data/` 目錄中。
2. 使用模型更新功能進行更新，在CMD會有訓練的過程和結果顯示。

---
