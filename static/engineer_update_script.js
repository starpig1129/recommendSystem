fetch('/data/engineer.csv')
    .then(response => response.text())
    .then(data => populateTable(data));

    function populateTable(data) {
        const rows = data.split('\n');
        const table = document.getElementById('csvTable');
        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');

        // Handle header row
        const headerCells = rows[0].split(',').map(cell => `<th>${cell.trim()}</th>`).join('');
        thead.innerHTML = `<tr>${headerCells}<th>Actions</th></tr>`;

        // Handle data rows
        for (let i = 1; i < rows.length; i++) {
            if(rows[i].trim() === '') continue;  // 忽略空的資料行
            const cells = rows[i].split(',').map(cell => `<td>${cell.trim()}</td>`).join('');
            const row = document.createElement('tr');
            row.innerHTML = cells + '<td><button onclick="editRow(this)">Edit</button> <button onclick="deleteRow(this)">Delete</button></td>';
            tbody.appendChild(row);
        }
    }


    function saveToCSV() {
        let csvContent = '';
        const table = document.getElementById('csvTable');
        Array.from(table.rows).forEach(row => {
            const rowData = Array.from(row.children).slice(0, -1).map(cell => {
                const inputElem = cell.querySelector('input');
                if (inputElem) {
                    return inputElem.value;
                }
                return cell.innerText;
            }).join(',');
            if(rowData.replace(/,/g, '').trim() !== '') { // 確保不加入空行
                csvContent += rowData + '\n';
            }
        });

        fetch('/update_csv', {
            method: 'POST',
            body: csvContent,
            headers: {
                'Content-Type': 'text/csv'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            return fetch('/data/engineer.csv');  // Fetch the updated CSV
        })
        .then(response => response.text())
        .then(data => {
            clearTable();  // 清空現有的表格
            populateTable(data);  // 使用更新的CSV重新填充表格
        });
    }

    function clearTable() {
        const table = document.getElementById('csvTable');
        const tbody = table.querySelector('tbody');
        while (tbody.firstChild) {
            tbody.removeChild(tbody.firstChild);
        }
    }

    function editRow(button) {
        const row = button.parentElement.parentElement;
        const cells = Array.from(row.children);
        cells.forEach((cell, index) => {
            if(index !== cells.length - 1) {
                const text = cell.innerText;
                cell.innerHTML = `<input type="text" value="${text}" />`;
            }
        });
        button.innerText = "Save";
        button.onclick = function() {
            saveRow(button);
            saveToCSV();  // Save changes immediately
        };
    }

    function saveRow(button) {
        const row = button.parentElement.parentElement;
        const cells = Array.from(row.children);
        cells.forEach((cell, index) => {
            if(index !== cells.length - 1) {
                const inputElem = cell.querySelector('input');
                if (inputElem) {
                    const text = inputElem.value;
                    cell.innerText = text;
                }
            }
        });
        button.innerText = "Edit";
        button.onclick = function() {
            editRow(button);
        };
        saveToCSV();  // 立即保存更改
    }


    function deleteRow(button) {
        const row = button.parentElement.parentElement;
        row.remove();
        saveToCSV();  // Save changes immediately
    }

    function addDataRow() {
        const table = document.getElementById('csvTable');
        const tbody = table.querySelector('tbody');
        const newRow = document.createElement('tr');

        const numberOfCells = table.querySelector('thead').querySelectorAll('th').length;
        for (let i = 0; i < numberOfCells - 1; i++) {
            const newCell = document.createElement('td');
            const inputField = document.createElement('input');
            inputField.type = "text";
            newCell.appendChild(inputField);
            newRow.appendChild(newCell);
        }

        const actionCell = document.createElement('td');
        actionCell.innerHTML = '<button onclick="saveRow(this)">Save</button> <button onclick="deleteRow(this)">Delete</button>';
        newRow.appendChild(actionCell);

        tbody.appendChild(newRow);
    }