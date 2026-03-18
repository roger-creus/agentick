// Sortable leaderboard table

function createSortableTable(tableId) {
    const table = document.getElementById(tableId);
    const headers = table.querySelectorAll('th');

    headers.forEach((header, index) => {
        header.addEventListener('click', () => {
            sortTable(table, index);
        });
        header.style.cursor = 'pointer';
    });
}

function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    rows.sort((a, b) => {
        const aValue = a.children[columnIndex].textContent;
        const bValue = b.children[columnIndex].textContent;

        // Try numeric sort
        const aNum = parseFloat(aValue);
        const bNum = parseFloat(bValue);

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return bNum - aNum; // Descending
        }

        // String sort
        return aValue.localeCompare(bValue);
    });

    // Reappend rows
    rows.forEach(row => tbody.appendChild(row));
}
