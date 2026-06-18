const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const BLOCK_SIZE = 20;

// Chart setup
const chartCtx = document.getElementById('chartCanvas').getContext('2d');
const learningChart = new Chart(chartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Score',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true
            },
            {
                label: 'Mean Score',
                data: [],
                borderColor: '#10b981',
                borderWidth: 2,
                pointRadius: 0,
                borderDash: [5, 5]
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#94a3b8' }
            }
        },
        plugins: {
            legend: {
                labels: { color: '#f8fafc' }
            }
        }
    }
});

// WebSocket connection
const ws = new WebSocket(`ws://${window.location.host}/ws`);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Update DOM stats
    document.getElementById('score').innerText = data.score;
    document.getElementById('record').innerText = data.record;
    document.getElementById('games').innerText = data.games;

    // Draw game
    drawGame(data.snake, data.food);

    // Update Chart
    if (data.scores.length > learningChart.data.labels.length) {
        learningChart.data.labels = Array.from({length: data.scores.length}, (_, i) => i + 1);
        learningChart.data.datasets[0].data = data.scores;
        learningChart.data.datasets[1].data = data.mean_scores;
        learningChart.update();
    }
};

function drawGame(snake, food) {
    // clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // draw snake
    snake.forEach((pt, index) => {
        ctx.fillStyle = index === 0 ? '#60a5fa' : '#3b82f6'; // Head is lighter blue
        ctx.fillRect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE);
        
        // inner style
        ctx.fillStyle = index === 0 ? '#93c5fd' : '#2563eb';
        ctx.fillRect(pt.x + 4, pt.y + 4, 12, 12);
    });

    // draw food
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE);
}

// --- Tab Logic ---
function openTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    // Remove active class from buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabId).style.display = 'flex';
    // Add active class to clicked button
    event.currentTarget.classList.add('active');

    // Fix chessboard rendering bug when hidden
    if (tabId === 'chessTab' && chessBoard) {
        chessBoard.resize();
    }
}

// --- Chess Competitive Logic ---
let chessBoard = null;
const chessGame = new Chess();

function updateChessStatus(isAIThinking) {
    const statusEl = document.getElementById('chessStatus');
    if (!statusEl) return;
    
    if (chessGame.in_checkmate()) {
        statusEl.innerText = `Checkmate! Winner is ${chessGame.turn() === 'w' ? 'Black (AI)' : 'White (You)'}.`;
        statusEl.style.color = 'var(--danger)';
    } else if (chessGame.in_draw() || chessGame.in_stalemate() || chessGame.in_threefold_repetition()) {
        statusEl.innerText = 'Draw!';
        statusEl.style.color = 'var(--text-muted)';
    } else {
        if (isAIThinking) {
            statusEl.innerText = 'AI is thinking...';
            statusEl.style.color = 'var(--primary)';
        } else {
            statusEl.innerText = 'Your Turn (White)';
            if (chessGame.in_check()) {
                statusEl.innerText += ' - CHECK!';
                statusEl.style.color = 'var(--danger)';
            } else {
                statusEl.style.color = 'var(--accent)';
            }
        }
    }
}

function makeAIMove() {
    updateChessStatus(true);
    
    fetch('/api/chess/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), depth: 3 })
    })
    .then(response => response.json())
    .then(data => {
        if (data.move) {
            chessGame.move(data.move, { sloppy: true });
            chessBoard.position(chessGame.fen());
        }
        updateChessStatus(false);
    })
    .catch(err => {
        console.error(err);
        document.getElementById('chessStatus').innerText = 'Eroare la AI.';
    });
}

function onDragStart(source, piece, position, orientation) {
    // block drag if game over or if it's black's turn (AI)
    if (chessGame.game_over()) return false;
    if (piece.search(/^b/) !== -1) return false;
}

function onDrop(source, target) {
    // validate move
    const move = chessGame.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    // illegal move
    if (move === null) return 'snapback';

    updateChessStatus(false);

    // trigger AI
    window.setTimeout(makeAIMove, 250);
}

function onSnapEnd() {
    chessBoard.position(chessGame.fen());
}

$(document).ready(function() {
    const config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    };
    chessBoard = Chessboard('board', config);
    updateChessStatus(false);
});

function resetChessBoard() {
    chessGame.reset();
    chessBoard.start();
    updateChessStatus(false);
}

// --- Manual Snake Control ---
let isManual = false;
function toggleSnakeMode() {
    isManual = !isManual;
    const btn = document.getElementById('snakeModeBtn');
    if (isManual) {
        btn.innerText = "🎮 Mode: Manual (Play!)";
        btn.style.backgroundColor = "#ef4444";
    } else {
        btn.innerText = "🤖 Mod: AI (Automat)";
        btn.style.backgroundColor = "var(--primary)";
    }
    ws.send(JSON.stringify({type: "mode", manual: isManual}));
}

document.addEventListener('keydown', (e) => {
    if (isManual && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        ws.send(JSON.stringify({type: "action", key: e.key}));
    }
});
