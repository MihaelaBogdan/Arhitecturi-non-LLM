function switchTab(tabId, element) {
    document.querySelectorAll('.tab-pane').forEach(tab => tab.style.display = 'none');
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    
    document.getElementById(tabId).style.display = 'block';
    element.classList.add('active');

    // Trigger resizes for canvas/board
    if (tabId === 'chess-tab' && typeof chessBoard !== 'undefined' && chessBoard) {
        chessBoard.resize();
    }
}
