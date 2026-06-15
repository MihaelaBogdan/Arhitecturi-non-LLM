// --- Snake Logic ---
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const BLOCK_SIZE = 20;

let chartInstance = null;

function initChart() {
    const ctxChart = document.getElementById('chartCanvas').getContext('2d');
    chartInstance = new Chart(ctxChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Scor per Episod', data: [], borderColor: '#3b82f6', tension: 0.1, fill: false },
                { label: 'Media (ultimele 100)', data: [], borderColor: '#10b981', tension: 0.1, fill: false }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true } },
            animation: false,
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            }
        }
    });
}

function updateChart(scores, mean_scores) {
    if (!chartInstance) return;
    chartInstance.data.labels = Array.from({length: scores.length}, (_, i) => i + 1);
    chartInstance.data.datasets[0].data = scores;
    chartInstance.data.datasets[1].data = mean_scores;
    chartInstance.update();
}

function drawGame(data) {
    const snake = data.snake;
    const food = data.food;
    
    // clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // draw snake
    snake.forEach((pt, index) => {
        ctx.fillStyle = index === 0 ? '#3b82f6' : '#60a5fa';
        ctx.fillRect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE);
        ctx.strokeStyle = '#0f172a';
        ctx.strokeRect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE);
    });

    // draw food
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE);
}

// WebSocket connection
const ws = new WebSocket(`ws://${window.location.host}/ws`);

ws.onopen = function() { console.log("WebSocket connected."); initChart(); };

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    document.getElementById('score').innerText = data.score;
    document.getElementById('record').innerText = data.record;
    document.getElementById('games').innerText = data.games;
    
    drawGame(data);
    updateChart(data.plot_scores, data.plot_mean_scores);
};

// --- Manual Mode ---
let isManual = false;
function toggleSnakeMode() {
    isManual = !isManual;
    const btn = document.getElementById('snakeModeBtn');
    if (isManual) {
        btn.innerText = "🎮 Mod: Manual (Joacă tu!)";
        btn.classList.remove('secondary-btn');
        btn.style.backgroundColor = "var(--danger)";
    } else {
        btn.innerText = "🤖 Auto Mode (AI)";
        btn.classList.add('secondary-btn');
        btn.style.backgroundColor = "";
    }
    ws.send(JSON.stringify({type: "mode", manual: isManual}));
}

document.addEventListener('keydown', (e) => {
    if (isManual && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        ws.send(JSON.stringify({type: "action", key: e.key}));
    }
});
