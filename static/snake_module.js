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
                { label: 'Media (toate)', data: [], borderColor: '#10b981', tension: 0.1, fill: false }
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

let lastScoresLength = 0;

function updateChart(scores, mean_scores) {
    if (!chartInstance || !scores) return;
    if (scores.length === lastScoresLength && chartInstance.data.labels.length > 0) return;
    lastScoresLength = scores.length;
    
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

function updateRiskBar(barId, labelId, proba) {
    const bar = document.getElementById(barId);
    const label = document.getElementById(labelId);
    if (!bar || !label) return;
    
    const pct = Math.round(proba * 100);
    label.innerText = pct + '%';
    bar.style.width = pct + '%';
    
    if (pct > 70) {
        bar.style.backgroundColor = 'var(--danger)';
        label.style.color = 'var(--danger)';
    } else if (pct > 30) {
        bar.style.backgroundColor = '#f59e0b'; // orange/yellow
        label.style.color = '#f59e0b';
    } else {
        bar.style.backgroundColor = 'var(--accent)';
        label.style.color = 'var(--accent)';
    }
}

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    document.getElementById('score').innerText = data.score;
    document.getElementById('record').innerText = data.record;
    document.getElementById('games').innerText = data.games;
    
    drawGame(data);
    updateChart(data.scores || data.plot_scores, data.mean_scores || data.plot_mean_scores);
    
    // Update Decision Tree Rule
    if (data.tree_rule) {
        document.getElementById('snakeActiveRule').innerText = data.tree_rule;
    }
    
    // Update Naive Bayes Risks
    if (data.bayes_risks) {
        updateRiskBar('riskBarStraight', 'riskLabelStraight', data.bayes_risks.straight);
        updateRiskBar('riskBarRight', 'riskLabelRight', data.bayes_risks.right);
        updateRiskBar('riskBarLeft', 'riskLabelLeft', data.bayes_risks.left);
    }
    
    // Update RNN Predicted Score
    if (data.predicted_next_score !== undefined) {
        document.getElementById('snakePredictedScore').innerText = data.predicted_next_score > 0 ? data.predicted_next_score + ' puncte' : '—';
    }
};

// --- AI Mode Selection for Snake ---
let currentSnakeAIMode = 'dqn';

function setSnakeAIMode(mode) {
    currentSnakeAIMode = mode;
    document.getElementById('btnSnakeDQN').classList.toggle('active', mode === 'dqn');
    document.getElementById('btnSnakeTree').classList.toggle('active', mode === 'tree');
    document.getElementById('btnSnakeBayes').classList.toggle('active', mode === 'bayes');
    
    const desc = document.getElementById('snakeAiModeDesc');
    if (mode === 'dqn') {
        desc.textContent = 'DQN folosește o rețea neurală profundă pentru a aproxima calitatea deciziilor (Q-Values).';
    } else if (mode === 'tree') {
        desc.textContent = 'Decision Tree ia decizii pe baza unui arbore logic extras din experiențele anterioare ale DQN-ului.';
    } else if (mode === 'bayes') {
        desc.textContent = 'Naive Bayes folosește probabilități condiționate pentru a măsura și evita riscul de coliziune pe fiecare cale.';
    }
    
    ws.send(JSON.stringify({type: "ai_mode", ai_mode: mode}));
}

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
