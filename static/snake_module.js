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
                {
                    label: 'Scor per Episod',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.08)',
                    borderWidth: 2.5,
                    tension: 0.25,
                    fill: true,
                    pointRadius: 1,
                    pointBackgroundColor: '#3b82f6'
                },
                {
                    label: 'Media (toate)',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.04)',
                    borderWidth: 3,
                    tension: 0.2,
                    fill: true,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 10 }, maxTicksLimit: 12 }
                }
            },
            animation: false,
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { size: 11 } } }
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

function isCollision(x, y, snake) {
    // Check out of bounds
    if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) return true;
    // Check hit body (excluding the head)
    for (let i = 1; i < snake.length; i++) {
        if (snake[i].x === x && snake[i].y === y) return true;
    }
    return false;
}

function drawGame(data) {
    const snake = data.snake;
    const food = data.food;
    
    // Clear board
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw food
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE);
    // Glow effect on food
    ctx.shadowColor = '#ef4444';
    ctx.shadowBlur = 8;
    ctx.fillRect(food.x + 2, food.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4);
    ctx.shadowBlur = 0; // reset
    
    // Draw snake body and head
    snake.forEach((pt, index) => {
        if (index === 0) {
            // Head color represents active model
            if (isManual) {
                ctx.fillStyle = '#ef4444'; // Red for Manual
            } else if (data.ai_mode === 'tree') {
                ctx.fillStyle = '#ffb703'; // Gold for Decision Tree
            } else if (data.ai_mode === 'bayes') {
                ctx.fillStyle = '#fb8500'; // Orange for Naive Bayes
            } else {
                ctx.fillStyle = '#3b82f6'; // Blue for DQN
            }
        } else {
            ctx.fillStyle = isManual ? '#f87171' : '#60a5fa';
        }
        
        ctx.beginPath();
        ctx.roundRect(pt.x + 1, pt.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2, index === 0 ? 5 : 3);
        ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.stroke();
        
        // Draw eyes on snake head
        if (index === 0) {
            ctx.fillStyle = '#ffffff';
            let eyeX1, eyeY1, eyeX2, eyeY2;
            if (data.direction === 'up') {
                eyeX1 = pt.x + 5; eyeY1 = pt.y + 5;
                eyeX2 = pt.x + 15; eyeY2 = pt.y + 5;
            } else if (data.direction === 'down') {
                eyeX1 = pt.x + 5; eyeY1 = pt.y + 15;
                eyeX2 = pt.x + 15; eyeY2 = pt.y + 15;
            } else if (data.direction === 'left') {
                eyeX1 = pt.x + 5; eyeY1 = pt.y + 5;
                eyeX2 = pt.x + 5; eyeY2 = pt.y + 15;
            } else {
                eyeX1 = pt.x + 15; eyeY1 = pt.y + 5;
                eyeX2 = pt.x + 15; eyeY2 = pt.y + 15;
            }
            ctx.beginPath();
            ctx.arc(eyeX1, eyeY1, 2, 0, Math.PI * 2);
            ctx.arc(eyeX2, eyeY2, 2, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw pupil
            ctx.fillStyle = '#000000';
            ctx.beginPath();
            ctx.arc(eyeX1, eyeY1, 0.8, 0, Math.PI * 2);
            ctx.arc(eyeX2, eyeY2, 0.8, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    // Draw Diagnostics (only if direction and Q-values exist)
    if (snake.length > 0 && data.direction && data.dqn_q_values) {
        const head = snake[0];
        let dx = 0, dy = 0;
        if (data.direction === 'up') { dx = 0; dy = -1; }
        else if (data.direction === 'down') { dx = 0; dy = 1; }
        else if (data.direction === 'left') { dx = -1; dy = 0; }
        else if (data.direction === 'right') { dx = 1; dy = 0; }

        // Compute adjacent positions
        const sx = head.x + dx * BLOCK_SIZE;
        const sy = head.y + dy * BLOCK_SIZE;

        const rx = -dy, ry = dx;
        const rx_c = head.x + rx * BLOCK_SIZE;
        const ry_c = head.y + ry * BLOCK_SIZE;

        const lx = dy, ly = -dx;
        const lx_c = head.x + lx * BLOCK_SIZE;
        const ly_c = head.y + ly * BLOCK_SIZE;

        const adjacents = [
            { name: "ÎNAINTE", x: sx, y: sy, idx: 0, q: data.dqn_q_values[0], r: data.bayes_risks ? data.bayes_risks.straight : 0.0 },
            { name: "DREAPTA", x: rx_c, y: ry_c, idx: 1, q: data.dqn_q_values[1], r: data.bayes_risks ? data.bayes_risks.right : 0.0 },
            { name: "STÂNGA", x: lx_c, y: ly_c, idx: 2, q: data.dqn_q_values[2], r: data.bayes_risks ? data.bayes_risks.left : 0.0 }
        ];

        // Draw overlays on board
        adjacents.forEach(adj => {
            // Check if this adjacent cell is a collision (wall or body hit)
            const isColl = isCollision(adj.x, adj.y, snake);
            
            if (adj.x < 0 || adj.x >= canvas.width || adj.y < 0 || adj.y >= canvas.height) return;
            
            const isTree = (data.tree_action === adj.idx);
            const isBayes = (data.bayes_action === adj.idx);
            const isChosen = (data.chosen_action === adj.idx);
            
            ctx.save();
            
            if (isColl) {
                // If it is a collision block, draw danger overlay with a red cross (no choices here!)
                ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
                ctx.fillRect(adj.x, adj.y, BLOCK_SIZE, BLOCK_SIZE);
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
                ctx.strokeRect(adj.x, adj.y, BLOCK_SIZE, BLOCK_SIZE);
                
                ctx.font = 'bold 9px sans-serif';
                ctx.fillStyle = '#ef4444';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText("✕", adj.x + BLOCK_SIZE/2, adj.y + BLOCK_SIZE/2);
            } else {
                if (isChosen) {
                    // Highlight action chosen by active mode
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.12)';
                    ctx.fillRect(adj.x, adj.y, BLOCK_SIZE, BLOCK_SIZE);
                    ctx.strokeStyle = isManual ? '#ef4444' : (data.ai_mode === 'tree' ? '#ffb703' : (data.ai_mode === 'bayes' ? '#fb8500' : '#3b82f6'));
                    ctx.lineWidth = 2.5;
                    ctx.strokeRect(adj.x + 1, adj.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2);
                } else {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
                    ctx.strokeRect(adj.x, adj.y, BLOCK_SIZE, BLOCK_SIZE);
                }
                
                // Draw model votes (colored dots inside the grid cells)
                let dotX = adj.x + 5;
                if (isTree) {
                    ctx.fillStyle = '#ffb703'; // Tree - Gold dot
                    ctx.beginPath(); ctx.arc(dotX, adj.y + 15, 2.5, 0, Math.PI * 2); ctx.fill();
                    dotX += 6;
                }
                if (isBayes) {
                    ctx.fillStyle = '#fb8500'; // Bayes - Orange dot
                    ctx.beginPath(); ctx.arc(dotX, adj.y + 15, 2.5, 0, Math.PI * 2); ctx.fill();
                }
            }
            ctx.restore();
        });

        // 2. Render HUD Panel (now in a clean HTML container in the sidebar to the right of the board)
        updateHTMLHUD(data, adjacents, isManual);
    }
}

// WebSocket connection
let ws;
function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = function() { 
        console.log("WebSocket connected."); 
        initChart(); 
    };
    
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

        // Update model averages comparison
        if (data.avg_scores) {
            updateAverageBar('snakeAvgBarDQN', 'snakeAvgLabelDQN', data.avg_scores.dqn);
            updateAverageBar('snakeAvgBarTree', 'snakeAvgLabelTree', data.avg_scores.tree);
            updateAverageBar('snakeAvgBarBayes', 'snakeAvgLabelBayes', data.avg_scores.bayes);
        }
    };

    ws.onclose = function(e) {
        console.log("WebSocket connection closed. Reconnecting in 2 seconds...");
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = function(err) {
        console.error("WebSocket error:", err);
        ws.close();
    };
}

connectWebSocket();

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

function updateAverageBar(barId, labelId, avg) {
    const bar = document.getElementById(barId);
    const label = document.getElementById(labelId);
    if (!bar || !label) return;
    
    label.innerText = avg.toFixed(1) + ' pct';
    
    // Scale bar width (max expected average score of 25)
    const maxVal = 25;
    const pct = Math.min(100, Math.max(0, (avg / maxVal) * 100));
    bar.style.width = pct + '%';
}

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
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: "ai_mode", ai_mode: mode}));
    }
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
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: "mode", manual: isManual}));
    }
}

document.addEventListener('keydown', (e) => {
    if (isManual && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: "action", key: e.key}));
        }
    }
});

function updateHTMLHUD(data, adjacents, isManual) {
    let modeName = "DQN (REȚEA)";
    let modeColor = "#3b82f6";
    if (isManual) { modeName = "MANUAL"; modeColor = "#ef4444"; }
    else if (data.ai_mode === 'tree') { modeName = "DECISION TREE"; modeColor = "#ffb703"; }
    else if (data.ai_mode === 'bayes') { modeName = "NAIVE BAYES"; modeColor = "#fb8500"; }
    
    const modeSpan = document.getElementById("snakeHudActiveMode");
    if (modeSpan) {
        modeSpan.innerText = modeName;
        modeSpan.style.color = modeColor;
    }

    adjacents.forEach((adj) => {
        const row = document.getElementById(`snakeHudRow${adj.idx}`);
        const qEl = document.getElementById(`snakeHudQ${adj.idx}`);
        const rEl = document.getElementById(`snakeHudR${adj.idx}`);
        const tEl = document.getElementById(`snakeHudT${adj.idx}`);
        if (!row || !qEl || !rEl || !tEl) return;

        const isChosen = (data.chosen_action === adj.idx);
        row.style.background = isChosen ? 'rgba(255,255,255,0.04)' : 'transparent';
        row.style.borderLeft = isChosen ? `3px solid ${modeColor}` : 'none';
        row.style.paddingLeft = isChosen ? `8px` : '0px';

        // Q-DQN value
        const isDqnActive = (!isManual && data.ai_mode === 'dqn');
        qEl.innerText = adj.q.toFixed(2);
        qEl.style.color = isChosen && isDqnActive ? '#60a5fa' : ((adj.q === Math.max(...data.dqn_q_values)) ? '#10b981' : '#94a3b8');
        qEl.style.fontWeight = isChosen ? 'bold' : 'normal';

        // Bayes risk pct
        const isBayesActive = (!isManual && data.ai_mode === 'bayes');
        const riskPct = Math.round(adj.r * 100);
        rEl.innerText = riskPct + "%";
        rEl.style.color = isChosen && isBayesActive ? '#fb8500' : (riskPct > 60 ? '#ef4444' : (riskPct > 20 ? '#f59e0b' : '#94a3b8'));
        rEl.style.fontWeight = isChosen ? 'bold' : 'normal';

        // Tree vote
        const isTreeActive = (!isManual && data.ai_mode === 'tree');
        if (data.tree_action === adj.idx) {
            tEl.innerText = "★ VOT";
            tEl.style.color = '#ffb703';
            tEl.style.fontWeight = 'bold';
        } else {
            tEl.innerText = "—";
            tEl.style.color = '#475569';
            tEl.style.fontWeight = 'normal';
        }
    });
}
