// 選擇所有的範圍輸入滑塊 (range input sliders)
const sliders = document.querySelectorAll('input[type="range"]');

// 選擇建立模型的按鈕
const updateButton = document.querySelector('.generate.submit-button');

// 選擇模型名稱的輸入框
const modelNameInput = document.querySelector('input[name="name"]');

// 選擇模型文件的輸入框
const modelFileInput = document.querySelector('input[name="engineerFile"]');

// 選擇上傳文件名的元素
const uploadedFileNameSpan = document.getElementById('uploadedFileName');

// 為每一個滑塊添加事件監聽器，以更新對應的顯示值
sliders.forEach(slider => {
    const valueSpan = document.getElementById(`${slider.name}Value`);

    slider.addEventListener('input', (event) => {
        valueSpan.textContent = event.target.value; // 更新滑塊旁邊的數值
    });
});

// 選擇“加入生成資料”的複選框
const showSlidersCheckbox = document.getElementById('showSlidersCheckbox');

// 選擇滑塊容器
const sliderContainer = document.getElementById('sliderContainer');

// 根據複選框的初始狀態設置滑塊容器的顯示
sliderContainer.style.display = showSlidersCheckbox.checked ? 'block' : 'none';

// 當複選框的狀態改變時，更新滑塊容器的顯示
showSlidersCheckbox.addEventListener('change', (event) => {
    sliderContainer.style.display = event.target.checked ? 'block' : 'none';
});

// 選擇模型文件夾的函數
function selectFolder(folderName) {
    // 更改界面
    let blocks = document.querySelectorAll(".folder-block span");
    const selectedColor = getComputedStyle(document.documentElement).getPropertyValue('--selected-folder-color').trim();

    blocks.forEach(block => {
        if (block.textContent == folderName) {
            block.parentNode.style.backgroundColor = selectedColor;
        } else {
            block.parentNode.style.backgroundColor = "";
        }
    });

    // 進行伺服器呼叫
    fetch(`/select_model`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ selectedModel: folderName })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
        //如果伺服器回應表示操作成功
    })
    .catch(error => {
        // Step 4: 如果伺服器回應表示操作失敗，撤銷前端的更改並顯示一個錯誤消息
        console.error("Error:", error);
        alert("操作失敗，請再試一次。");

        // 撤銷更改（如果需要的話）
        blocks.forEach(block => {
            block.parentNode.style.backgroundColor = ""; 
        });
    });
}


// 編輯模型文件夾名稱的函數
function editFolderName(oldName) {
    let newName = prompt("Enter new name for the folder:", oldName);
    if (newName && newName !== oldName) {
        fetch(`/edit_folder?old_name=${oldName}&new_name=${newName}`, {
            method: 'GET'
        })
        .then(response => response.text())
        .then(data => {
            console.log(data);
            location.reload();  // 重新載入頁面以看到更新後的名稱
        });
    }
}

// 刪除模型文件夾的函數
function deleteFolder(folderName) {
    const confirmation = confirm(`確定要刪除資料夾 "${folderName}" 嗎?`);
    if (confirmation) {
        fetch(`/delete_folder?folder_name=${folderName}`, {
            method: 'GET'
        })
        .then(response => location.reload());
    }
}

// 使用事件代理進行模型資料夾選擇
// 新增模型列表中每個項目的編輯和刪除功能
function createModelListItem(modelName, modelFile) {
    const listItem = document.createElement('tr');

    const modelNameCell = document.createElement('td');
    modelNameCell.textContent = modelName;

    const modelFileCell = document.createElement('td');
    modelFileCell.textContent = modelFile.name;

    const editButton = document.createElement('button');
    editButton.textContent = '編輯';
    editButton.classList.add('edit-button');
    editButton.addEventListener('click', () => {
        modelNameCell.contentEditable = true;
        modelNameCell.focus();
        editButton.disabled = true;
    });

    const deleteButton = document.createElement('button');
    deleteButton.textContent = '刪除';
    deleteButton.classList.add('delete-button');
    deleteButton.addEventListener('click', () => {
        listItem.remove();
    });

    const buttonsCell = document.createElement('td');
    buttonsCell.appendChild(editButton);
    buttonsCell.appendChild(deleteButton);

    listItem.appendChild(modelNameCell);
    listItem.appendChild(modelFileCell);
    listItem.appendChild(buttonsCell);

    addListItemButtons(listItem, modelNameCell, modelFile);

    return listItem;
}

// 建立新模型的事件監聽器
updateButton.addEventListener('click', () => {
    const modelName = modelNameInput.value.trim();

    if (modelName === '') {
        alert('模型名稱不能為空白！');
        return;
    }

    const modelFile = modelFileInput.files[0];

    if (modelFile) {
        const listItem = document.createElement('tr'); // 創建新的表格行

        const modelNameCell = document.createElement('td');
        modelNameCell.textContent = modelName;

        const editButton = document.createElement('button');
        editButton.textContent = '編輯';
        editButton.classList.add('edit-button');
        editButton.addEventListener('click', () => {
            modelNameCell.contentEditable = true;
            modelNameCell.focus();
            editButton.disabled = true;

            // 監視單元格失去焦點事件，當編輯完成時恢復按鈕狀態
            modelNameCell.addEventListener('blur', () => {
                modelNameCell.contentEditable = false;
                editButton.disabled = false;
            });
        });

        const deleteButton = document.createElement('button');
        deleteButton.textContent = '刪除';
        deleteButton.classList.add('delete-button');
        deleteButton.addEventListener('click', () => {
            listItem.remove();
        });

        const buttonsCell = document.createElement('td');
        buttonsCell.appendChild(editButton);
        buttonsCell.appendChild(deleteButton);

        listItem.appendChild(modelNameCell);
        listItem.appendChild(buttonsCell);

        modelList.appendChild(listItem); // 將新的表格行添加到模型列表中

        modelNameInput.value = '';
        modelFileInput.value = '';
    } else {
        alert('請上傳檔案！');
    }
});

// 當文檔加載完成時，從Flask路由獲取文件列表
document.addEventListener('DOMContentLoaded', function() {
    // 從 Flask 路徑中獲取文件列表
    fetch('/list-data-files')
    .then(response => response.json())
    .then(files => {
        // 生成文件列表供用戶選擇
        const fileList = document.getElementById('fileList');
        files.forEach(file => {
            const listItem = document.createElement('li');
            listItem.textContent = file;
            listItem.onclick = function() {
                // 處理文件選擇
                // 顯示帶有文件名的警告
                alert('Selected file: ' + file);
            };
            fileList.appendChild(listItem);
        });
    });
});

// 當文檔加載完成時，為每個滑塊的“預設”複選框添加事件監聽器
document.addEventListener('DOMContentLoaded', function() {
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const defaultCheckbox = slider.nextElementSibling.nextElementSibling;  // "預設"複選框
        defaultCheckbox.addEventListener('change', function() {
            if (this.checked) {
                slider.disabled = true;
                slider.classList.add('disabled-slider');
            } else {
                slider.disabled = false;
                slider.classList.remove('disabled-slider');
            }
        });
    });
});

// 當文檔加載完成時，為“生成”按鈕添加事件監聽器
document.addEventListener("DOMContentLoaded", function() {
    document.querySelector('.generate').addEventListener('click', function(event) {
        event.preventDefault();
        console.log("activate");
        
        let modelName = document.querySelector('input[name="modelName"]').value;
        console.log("model", modelName);
        
        let generateData = document.getElementById('showSlidersCheckbox').checked;

        let data = {
            modelName: modelName,
            generateData: generateData
        };
        // 根據每個滑塊的狀態添加資料
        if (!document.querySelector('input[name="fundingScale"]').nextElementSibling.checked) {
            data.fundingScale = document.getElementById('fundingScaleSlider').value;
        }
        if (!document.querySelector('input[name="estimatedHours"]').nextElementSibling.checked) {
            data.estimatedHours = document.getElementById('estimatedHoursSlider').value;
        }
        if (!document.querySelector('input[name="difficulty"]').nextElementSibling.checked) {
            data.difficulty = document.getElementById('difficultySlider').value;
        }
        if (!document.querySelector('input[name="collaboration"]').nextElementSibling.checked) {
            data.collaboration = document.getElementById('collaborationSlider').value;
        }
        if (!document.querySelector('input[name="seniority"]').nextElementSibling.checked) {
            data.seniority = document.getElementById('senioritySlider').value;
        }
       // 用於捕獲checkbox的選擇狀態並將其加入到data對象中的代碼片段
        let checkbox_names = ["fundingScale", "estimatedHours", "difficulty", "seniority", "collaboration"];

        for (let checkbox of checkbox_names) {
            data[`${checkbox}_checked`] = document.querySelector(`input[name="${checkbox}"] + span + input[type="checkbox"]`).checked;
        }
        // 發送數據到Flask路由
        fetch('/create-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            location.reload(); 
        });
    });
});

// 打開文件選擇模態框的函數
function openFileModal() {
    document.getElementById('fileModal').style.display='block';
    fetch('/list-data-files')
    .then(response => response.json())
    .then(files => {
        const fileList = document.getElementById('fileListModal');
        fileList.innerHTML = '';  // 清除以前的列表
        files.forEach(file => {
            const listItem = document.createElement('li');
            listItem.textContent = file;
            listItem.onclick = function() {
                // 處理文件選擇
                alert('Selected file: ' + file);
                document.getElementById('fileModal').style.display='none';
            };
            fileList.appendChild(listItem);
        });
    });
}

document.addEventListener("DOMContentLoaded", function() {
    // 找到 "建立模型" 按鈕
    const generateButton = document.querySelector(".generate");
  
    // 當按鈕被點擊時
    generateButton.addEventListener("click", function() {
      // 顯示加載畫面
      document.getElementById("loadingScreen").style.display = "flex"; 
    });
  });
  