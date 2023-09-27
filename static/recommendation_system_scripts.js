document.addEventListener("DOMContentLoaded", function () {

    // 拖放功能
    let list = document.querySelector('.list');
    let currentLi;

    // 恢复保存的排序
    let savedOrderedList = document.getElementById('orderedList').value.split(',');
    if (savedOrderedList.length > 0) {
        savedOrderedList.forEach(item => {
            let liItem = Array.from(list.childNodes).find(li => li.textContent.trim() === item);
            if (liItem) {
                list.appendChild(liItem);
            }
        });
    }

    list.addEventListener('dragstart', (e) => {
        e.dataTransfer.effectAllowed = 'move';
        currentLi = e.target;
        setTimeout(() => {
            currentLi.classList.add('moving');
        });
    });

    list.addEventListener('dragenter', (e) => {
        e.preventDefault();
        if (e.target === currentLi || e.target === list) {
            return;
        }
        let liArray = Array.from(list.childNodes);
        let currentIndex = liArray.indexOf(currentLi);
        let targetindex = liArray.indexOf(e.target);

        if (currentIndex < targetindex) {
            list.insertBefore(currentLi, e.target.nextElementSibling);
        } else {
            list.insertBefore(currentLi, e.target);
        }
    });

    list.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    list.addEventListener('dragend', (e) => {
        currentLi.classList.remove('moving');

        // Update the hidden field value with the current order of the list (without empty values)
        let orderedListInput = document.getElementById('orderedList');
        let liArray = Array.from(list.childNodes);
        let orderedValues = liArray.map(li => li.textContent.trim()).filter(value => value !== "");
        orderedListInput.value = orderedValues.join(',');
    });

    // 滑塊更新功能
    function updateSliderValue(sliderId, valueId) {
        const slider = document.getElementById(sliderId);
        const valueElement = document.getElementById(valueId);
        slider.addEventListener("input", () => {
            valueElement.textContent = slider.value;
        });
    }

    updateSliderValue("experienceSlider", "experienceValue");
    updateSliderValue("expertiseSlider", "expertiseValue");
    updateSliderValue("preferenceSlider", "preferenceValue");
});

document.addEventListener("DOMContentLoaded", function() {
    // 找到表單元素
    const form = document.getElementById("orderForm");
  
    // 當表單提交時
    form.addEventListener("submit", function() {
      // 顯示加載畫面
      document.getElementById("loadingScreen").style.display = "flex";
    });
  });
$(document).ready(function() {
    // 遍歷每個表格的每個單元格
    $('.scrollable-table table tbody td').each(function() {
        var text = $(this).text().trim();
        
        // 檢查是否為數字
        if(!isNaN(text) && text.indexOf('.') !== -1) {
            $(this).text(parseFloat(text).toFixed(1)); // 將數字修改為只有一位小數
        }
    });
});
