// ============================================================
//  Tetris KNN & Manual Mode — Professional Edition
// ============================================================

const tetrisCanvas = document.getElementById('tetrisCanvas');
const tctx = tetrisCanvas.getContext('2d');
const nextCanvas = document.getElementById('tetrisNextCanvas');
const nctx = nextCanvas ? nextCanvas.getContext('2d') : null;

const ROWS = 20;
const COLS = 10;
const BLOCK = 32;

// Resize canvas to proper dimensions
tetrisCanvas.width = COLS * BLOCK;
tetrisCanvas.height = ROWS * BLOCK;

let board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let score = 0;
let linesCleared = 0;
let level = 1;
let combo = 0;
let bestScore = parseInt(localStorage.getItem('tetrisBest') || '0');
let aiInterval = null;
let manualInterval = null;
let isManualMode = false;

// Online Learning and Model State
let tetrisAIMode = 'knn';
let tetrisHistory = [];
let linesClearedThisTurn = 0;

// Particle system
let particles = [];

// Flash animation state
let flashLines = [];
let flashAlpha = 0;

const SHAPES = [
    [[1,1,1,1]],           // I
    [[1,1],[1,1]],          // O
    [[0,1,0],[1,1,1]],      // T
    [[1,0,0],[1,1,1]],      // L
    [[0,0,1],[1,1,1]],      // J
    [[0,1,1],[1,1,0]],      // S
    [[1,1,0],[0,1,1]]       // Z
];

// Rich color palette with gradient pairs [base, highlight, shadow]
const PIECE_STYLES = [
    null, // 0 = empty
    { base: '#00d4ff', hi: '#7bf5ff', shadow: '#0090bb', name: 'I' }, // I - cyan
    { base: '#ffd600', hi: '#ffe97a', shadow: '#b59900', name: 'O' }, // O - yellow
    { base: '#a259ff', hi: '#cf9fff', shadow: '#6b2fbf', name: 'T' }, // T - purple
    { base: '#ff8c00', hi: '#ffba5c', shadow: '#b35d00', name: 'L' }, // L - orange
    { base: '#0052ff', hi: '#5c97ff', shadow: '#0033a0', name: 'J' }, // J - blue
    { base: '#00e676', hi: '#7dffc2', shadow: '#009945', name: 'S' }, // S - green
    { base: '#ff3d5a', hi: '#ff8fa0', shadow: '#b3001e', name: 'Z' }, // Z - red
];

let currentPiece = null;
let currentPos = { x: 0, y: 0 };
let nextPiece = null;
let targetX = 0;
let targetShape = null;

// Learning curve chart variables
let tetrisScores = [];
let tetrisMeanScores = [];
let tetrisChartInstance = null;
let piecesPlaced = 0;

// ─── Utilities ────────────────────────────────────────────────

function rotateMatrix(shape) {
    const n = shape.length, m = shape[0].length;
    const rotated = Array.from({ length: m }, () => Array(n).fill(0));
    for (let r = 0; r < n; r++)
        for (let c = 0; c < m; c++)
            rotated[c][n - 1 - r] = shape[r][c];
    return rotated;
}

function collide(x, y, shape) {
    for (let r = 0; r < shape.length; r++)
        for (let c = 0; c < shape[r].length; c++)
            if (shape[r][c]) {
                const nx = x + c, ny = y + r;
                if (nx < 0 || nx >= COLS || ny >= ROWS) return true;
                if (ny >= 0 && board[ny][nx] !== 0) return true;
            }
    return false;
}

function getGhostY(x, y, shape) {
    let gy = y;
    while (!collide(x, gy + 1, shape)) gy++;
    return gy;
}

// ─── Model Toggle ─────────────────────────────────────────────

function setTetrisAIMode(mode) {
    tetrisAIMode = mode;
    
    // Toggle active class on buttons
    document.querySelectorAll('.ai-mode-toggle button[id^="btnTetris"]').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const modeBtnId = 'btnTetris' + mode.charAt(0).toUpperCase() + mode.slice(1);
    const modeBtn = document.getElementById(modeBtnId);
    if (modeBtn) modeBtn.classList.add('active');
    
    // Description text update
    const descEl = document.getElementById('tetrisAiModeDesc');
    if (descEl) {
        if (mode === 'knn') {
            descEl.innerText = "KNN evaluează starea tablei comparând-o cu cele mai similare table din istoric (TD-Learning).";
        } else if (mode === 'tree') {
            descEl.innerText = "Arborele de decizie estimează calitatea pe baza metricilor tablei, antrenându-se online.";
        } else if (mode === 'mlp') {
            descEl.innerText = "Rețeaua neuronală MLP (Multi-Layer Perceptron) învață online prin optimizarea erorii TD.";
        } else if (mode === 'genetic') {
            descEl.innerText = "Strategia Evolutivă (ES) optimizează ponderile euristice online prin feedback direct.";
        }
    }
    
    // Weights panel display
    const weightsCard = document.getElementById('tetrisGeneticWeightsCard');
    if (weightsCard) {
        weightsCard.style.display = (mode === 'genetic') ? 'block' : 'none';
    }
    
    // Refresh next move recommendation instantly
    requestNextMove();
}

function updateWeightsDisplay(weights) {
    if (!weights) return;
    const wHeight = document.getElementById('weightHeight');
    const wLines = document.getElementById('weightLines');
    const wHoles = document.getElementById('weightHoles');
    const wBump = document.getElementById('weightBumpiness');
    
    if (wHeight) wHeight.innerText = parseFloat(weights.height).toFixed(3);
    if (wLines) wLines.innerText = parseFloat(weights.lines).toFixed(3);
    if (wHoles) wHoles.innerText = parseFloat(weights.holes).toFixed(3);
    if (wBump) wBump.innerText = parseFloat(weights.bumpiness).toFixed(3);
}

// ─── Drawing ──────────────────────────────────────────────────

function drawBlock(ctx, x, y, colorId, alpha = 1.0, blockSize = BLOCK) {
    if (!colorId || colorId === 0) return;
    const style = PIECE_STYLES[colorId];
    if (!style) return;

    const px = x * blockSize;
    const py = y * blockSize;
    const s = blockSize;

    ctx.save();
    ctx.globalAlpha = alpha;

    // Main gradient fill
    const grad = ctx.createLinearGradient(px, py, px + s * 0.5, py + s * 0.5);
    grad.addColorStop(0, style.hi);
    grad.addColorStop(0.5, style.base);
    grad.addColorStop(1, style.shadow);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(px + 1, py + 1, s - 2, s - 2, 3);
    ctx.fill();

    // Inner highlight (top-left bevel)
    ctx.fillStyle = 'rgba(255,255,255,0.28)';
    ctx.beginPath();
    ctx.roundRect(px + 3, py + 3, s * 0.55, s * 0.22, 2);
    ctx.fill();

    // Border
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(px + 1, py + 1, s - 2, s - 2, 3);
    ctx.stroke();

    ctx.restore();
}

function drawGhostBlock(ctx, x, y, colorId, blockSize = BLOCK) {
    if (!colorId || colorId === 0) return;
    const style = PIECE_STYLES[colorId];
    if (!style) return;

    const px = x * blockSize;
    const py = y * blockSize;
    const s = blockSize;

    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = style.base;
    ctx.beginPath();
    ctx.roundRect(px + 1, py + 1, s - 2, s - 2, 3);
    ctx.fill();
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = style.base;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.roundRect(px + 1, py + 1, s - 2, s - 2, 3);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
}

function drawGrid() {
    tctx.fillStyle = '#0a0e1a';
    tctx.fillRect(0, 0, tetrisCanvas.width, tetrisCanvas.height);

    // Subtle grid lines
    tctx.strokeStyle = 'rgba(255,255,255,0.04)';
    tctx.lineWidth = 0.5;
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            tctx.strokeRect(c * BLOCK, r * BLOCK, BLOCK, BLOCK);
        }
    }
}

function drawTetris() {
    drawGrid();

    // Flash effect for line clear
    if (flashAlpha > 0 && flashLines.length > 0) {
        tctx.save();
        tctx.globalAlpha = flashAlpha;
        tctx.fillStyle = '#ffffff';
        for (const row of flashLines) {
            tctx.fillRect(0, row * BLOCK, COLS * BLOCK, BLOCK);
        }
        tctx.restore();
        flashAlpha -= 0.08;
        if (flashAlpha < 0) flashAlpha = 0;
    }

    // Draw placed blocks
    for (let r = 0; r < ROWS; r++)
        for (let c = 0; c < COLS; c++)
            if (board[r][c] !== 0)
                drawBlock(tctx, c, r, board[r][c]);

    // Draw ghost piece (always visible when there's a current piece)
    if (currentPiece) {
        const ghostY = getGhostY(currentPos.x, currentPos.y, currentPiece.shape);
        if (ghostY !== currentPos.y) {
            for (let r = 0; r < currentPiece.shape.length; r++)
                for (let c = 0; c < currentPiece.shape[r].length; c++)
                    if (currentPiece.shape[r][c])
                        drawGhostBlock(tctx, currentPos.x + c, ghostY + r, currentPiece.id);
        }
    }

    // Draw AI recommendation ghost (gold overlay) in manual mode
    if (isManualMode && targetShape && targetX >= -2 && targetX < COLS) {
        let landingY = 0;
        while (!collide(targetX, landingY + 1, targetShape)) landingY++;

        tctx.save();
        for (let r = 0; r < targetShape.length; r++) {
            for (let c = 0; c < targetShape[r].length; c++) {
                if (targetShape[r][c]) {
                    const px = (targetX + c) * BLOCK;
                    const py = (landingY + r) * BLOCK;
                    tctx.globalAlpha = 0.22;
                    tctx.fillStyle = '#fbbf24';
                    tctx.beginPath();
                    tctx.roundRect(px + 1, py + 1, BLOCK - 2, BLOCK - 2, 3);
                    tctx.fill();
                    tctx.globalAlpha = 0.7;
                    tctx.strokeStyle = '#fbbf24';
                    tctx.lineWidth = 2;
                    tctx.setLineDash([5, 3]);
                    tctx.beginPath();
                    tctx.roundRect(px + 1, py + 1, BLOCK - 2, BLOCK - 2, 3);
                    tctx.stroke();
                    tctx.setLineDash([]);
                }
            }
        }
        tctx.restore();
    }

    // Draw active falling piece
    if (currentPiece) {
        for (let r = 0; r < currentPiece.shape.length; r++)
            for (let c = 0; c < currentPiece.shape[r].length; c++)
                if (currentPiece.shape[r][c])
                    drawBlock(tctx, currentPos.x + c, currentPos.y + r, currentPiece.id);
    }

    // Draw particles
    drawParticles();
}

function drawNextPiece() {
    if (!nctx || !nextPiece) return;

    const NB = 24; // next block size
    const NW = nextCanvas.width;
    const NH = nextCanvas.height;

    nctx.fillStyle = '#0a0e1a';
    nctx.fillRect(0, 0, NW, NH);

    const rows = nextPiece.shape.length;
    const cols = nextPiece.shape[0].length;
    const startX = Math.floor((NW / NB - cols) / 2);
    const startY = Math.floor((NH / NB - rows) / 2);

    for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
            if (nextPiece.shape[r][c])
                drawBlock(nctx, startX + c, startY + r, nextPiece.id, 1.0, NB);
}

// ─── Particles ────────────────────────────────────────────────

function spawnParticles(row, colorId) {
    const style = PIECE_STYLES[colorId] || PIECE_STYLES[1];
    for (let c = 0; c < COLS; c++) {
        for (let i = 0; i < 4; i++) {
            particles.push({
                x: (c + 0.5) * BLOCK,
                y: (row + 0.5) * BLOCK,
                vx: (Math.random() - 0.5) * 5,
                vy: (Math.random() - 0.8) * 5,
                life: 1.0,
                decay: 0.05 + Math.random() * 0.04,
                size: 3 + Math.random() * 5,
                color: [style.base, style.hi, '#ffffff'][Math.floor(Math.random() * 3)]
            });
        }
    }
}

function drawParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.15; // gravity
        p.life -= p.decay;
        if (p.life <= 0) {
            particles.splice(i, 1);
            continue;
        }
        tctx.save();
        tctx.globalAlpha = p.life;
        tctx.fillStyle = p.color;
        tctx.beginPath();
        tctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
        tctx.fill();
        tctx.restore();
    }
}

// ─── Chart ────────────────────────────────────────────────────

function initTetrisChart() {
    const ctxChart = document.getElementById('tetrisChartCanvas').getContext('2d');
    tetrisChartInstance = new Chart(ctxChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Scor Joc',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59,130,246,0.08)',
                    tension: 0.3,
                    fill: true,
                    pointRadius: 3,
                    pointBackgroundColor: '#3b82f6'
                },
                {
                    label: 'Media',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16,185,129,0.06)',
                    tension: 0.3,
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
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', maxTicksLimit: 8 }
                }
            },
            animation: false,
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { size: 11 } } }
            }
        }
    });
}

function updateTetrisChart() {
    if (!tetrisChartInstance) return;

    let scoresToPlot = [...tetrisScores];
    let meansToPlot = [...tetrisMeanScores];

    const gameActive = aiInterval !== null || manualInterval !== null;
    if (gameActive && piecesPlaced > 0) {
        scoresToPlot.push(score);
        const totalSum = tetrisScores.reduce((a, b) => a + b, 0) + score;
        meansToPlot.push(totalSum / (tetrisScores.length + 1));
    }

    const labels = scoresToPlot.map((_, i) =>
        i < tetrisScores.length ? `Joc ${i + 1}` : `🔴 Live`
    );

    tetrisChartInstance.data.labels = labels;
    tetrisChartInstance.data.datasets[0].data = scoresToPlot;
    tetrisChartInstance.data.datasets[1].data = meansToPlot;
    tetrisChartInstance.update('none');
}

// ─── Speed helpers ────────────────────────────────────────────

function getAISpeedDelay() {
    const speed = document.getElementById('tetrisSpeed').value;
    return { slow: 250, normal: 100, fast: 40, turbo: 12 }[speed] || 100;
}

function getManualSpeedDelay() {
    const speed = document.getElementById('tetrisSpeed').value;
    return { slow: 1200, normal: 700, fast: 350, turbo: 120 }[speed] || 700;
}

function updateTetrisSpeed() {
    if (aiInterval) { clearInterval(aiInterval); aiInterval = setInterval(aiStep, getAISpeedDelay()); }
    if (manualInterval) { clearInterval(manualInterval); manualInterval = setInterval(manualStep, getManualSpeedDelay()); }
}

// ─── API Calls ────────────────────────────────────────────────

async function requestNextMove() {
    if (!currentPiece) return;
    try {
        const res = await fetch('/api/tetris/next-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board, shape: currentPiece.shape, model_type: tetrisAIMode })
        });
        const data = await res.json();
        targetX = data.x;
        targetShape = data.shape;
        updateMetrics(data.metrics);
        if (data.weights) updateWeightsDisplay(data.weights);
        drawTetris();
    } catch (err) {
        console.error("Failed to fetch next Tetris move:", err);
    }
}

function updateMetrics(metrics) {
    if (!metrics) return;
    document.getElementById('tetrisMetricHeight').innerText = metrics.height;
    document.getElementById('tetrisMetricHoles').innerText = metrics.holes;
    document.getElementById('tetrisMetricBumpiness').innerText = metrics.bumpiness;
    document.getElementById('tetrisMetricRowTransitions').innerText = metrics.row_transitions;
    document.getElementById('tetrisMetricColTransitions').innerText = metrics.col_transitions;
    document.getElementById('tetrisMetricWells').innerText = metrics.wells;
}

// ─── Piece Management ─────────────────────────────────────────

function spawnPiece() {
    const id = Math.floor(Math.random() * SHAPES.length) + 1;
    return {
        shape: SHAPES[id - 1],
        id,
        originalShape: JSON.parse(JSON.stringify(SHAPES[id - 1]))
    };
}

async function newPiece() {
    currentPiece = nextPiece || spawnPiece();
    nextPiece = spawnPiece();
    currentPos = { x: Math.floor((COLS - currentPiece.shape[0].length) / 2), y: 0 };

    drawNextPiece();
    await requestNextMove();
}

// ─── Merge & Line Clear ───────────────────────────────────────

function merge() {
    for (let r = 0; r < currentPiece.shape.length; r++)
        for (let c = 0; c < currentPiece.shape[r].length; c++)
            if (currentPiece.shape[r][c])
                board[currentPos.y + r][currentPos.x + c] = currentPiece.id;

    // Find and clear completed lines
    const clearedRows = [];
    for (let r = ROWS - 1; r >= 0; r--) {
        if (board[r].every(cell => cell !== 0)) {
            clearedRows.push(r);
        }
    }

    linesClearedThisTurn = clearedRows.length;

    if (clearedRows.length > 0) {
        for (const row of clearedRows) {
            for (let c = 0; c < COLS; c++) {
                if (board[row][c]) spawnParticles(row, board[row][c]);
            }
        }

        flashLines = [...clearedRows];
        flashAlpha = 0.9;

        for (const r of clearedRows) {
            board.splice(r, 1);
            board.unshift(Array(COLS).fill(0));
        }

        const count = clearedRows.length;
        linesCleared += count;
        level = Math.floor(linesCleared / 10) + 1;
        combo++;

        const lineScores = [0, 100, 300, 500, 800];
        const baseScore = (lineScores[count] || 1200) * level;
        const comboBonus = combo > 1 ? (combo - 1) * 50 * level : 0;
        score += baseScore + comboBonus;

        if (combo > 1) {
            const comboEl = document.getElementById('tetrisComboDisplay');
            if (comboEl) {
                comboEl.innerText = `🔥 x${combo} COMBO!`;
                comboEl.style.opacity = '1';
                clearTimeout(window._comboTimeout);
                window._comboTimeout = setTimeout(() => { comboEl.style.opacity = '0'; }, 1500);
            }
        }

        if (score > bestScore) {
            bestScore = score;
            localStorage.setItem('tetrisBest', bestScore);
        }
    } else {
        combo = 0;
    }

    document.getElementById('tetris-score').innerText = score;
    document.getElementById('tetrisLinesDisplay').innerText = linesCleared;
    document.getElementById('tetrisLevelDisplay').innerText = level;
    const bestEl = document.getElementById('tetrisBestDisplay');
    if (bestEl) bestEl.innerText = bestScore;
}

// ─── Move Evaluation ──────────────────────────────────────────

async function evaluateMove(boardBefore, originalShape, chosenX, chosenShape) {
    try {
        const res = await fetch('/api/tetris/evaluate-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: boardBefore,
                shape: originalShape,
                chosen_x: chosenX,
                chosen_shape: chosenShape,
                model_type: tetrisAIMode
            })
        });
        const data = await res.json();

        if (data.metrics) updateMetrics(data.metrics);

        const badge = document.getElementById('tetrisMoveBadge');
        badge.className = 'move-badge';
        const classMap = {
            'Excelentă': ['badge-excellent', '🌟'],
            'Bună':      ['badge-good',      '✅'],
            'Medie':     ['badge-okay',       '⚠️'],
            'Greșeală':  ['badge-blunder',    '❌']
        };
        const [cls, emoji] = classMap[data.classification] || ['badge-neutral', ''];
        badge.classList.add(cls);
        badge.innerText = `${data.classification} ${emoji}`;
        document.getElementById('tetrisMoveExplanation').innerText = data.explanation;
    } catch (err) {
        console.error("Failed to evaluate move:", err);
    }
}

// ─── Game Over Reporter ───────────────────────────────────────

async function sendGameOver(finalScore, history) {
    try {
        const res = await fetch('/api/tetris/game-over', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ score: finalScore, history: history })
        });
        const data = await res.json();
        console.log("Online training response:", data);
        if (data.weights) {
            updateWeightsDisplay(data.weights);
        }
    } catch (err) {
        console.error("Failed to report game over to server:", err);
    }
}

// ─── Core Game Loop ───────────────────────────────────────────

async function handleMergeAndNext() {
    if (currentPiece) {
        const boardBeforeMerge = JSON.parse(JSON.stringify(board));
        const originalShape = currentPiece.originalShape;
        const chosenShape = currentPiece.shape;
        const chosenX = currentPos.x;

        // Evaluate move on active model
        evaluateMove(boardBeforeMerge, originalShape, chosenX, chosenShape);
        merge();
        
        // Log state transition for online learning
        const gridFlat = [].concat(...board).map(cell => cell > 0 ? 1 : 0);
        const height = parseInt(document.getElementById('tetrisMetricHeight').innerText) || 0;
        const holes = parseInt(document.getElementById('tetrisMetricHoles').innerText) || 0;
        const bump = parseInt(document.getElementById('tetrisMetricBumpiness').innerText) || 0;
        
        tetrisHistory.push({
            features: [height, linesClearedThisTurn, holes, bump],
            grid: gridFlat,
            reward: linesClearedThisTurn * 100.0
        });
        
        piecesPlaced++;
    }

    updateTetrisChart();
    currentPiece = null;
    await newPiece();

    // Game Over Check
    if (collide(currentPos.x, currentPos.y, currentPiece.shape)) {
        // Send logs to trigger online learning
        sendGameOver(score, tetrisHistory);
        tetrisHistory = [];

        tetrisScores.push(score);
        tetrisMeanScores.push(tetrisScores.reduce((a, b) => a + b, 0) / tetrisScores.length);
        updateTetrisChart();

        board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        score = 0;
        linesCleared = 0;
        level = 1;
        combo = 0;
        piecesPlaced = 0;
        particles = [];
        document.getElementById('tetris-score').innerText = 0;
        document.getElementById('tetrisLinesDisplay').innerText = 0;
        document.getElementById('tetrisLevelDisplay').innerText = 1;

        const badge = document.getElementById('tetrisMoveBadge');
        badge.className = 'move-badge badge-neutral';
        badge.innerText = '💀 Game Over!';
        document.getElementById('tetrisMoveExplanation').innerText = 'Jocul s-a terminat. Pornește din nou!';

        if (isManualMode && manualInterval) {
            clearInterval(manualInterval); manualInterval = null;
            const btn = document.getElementById('btnToggleTetrisManual');
            btn.innerText = '🎮 Joacă tu (Mod Manual)';
            btn.style.background = '';
        }
        if (!isManualMode && aiInterval) {
            clearInterval(aiInterval); aiInterval = null;
            const btn = document.getElementById('btnStartTetrisAI');
            btn.innerText = '🤖 Pornire AI';
            btn.style.background = '';
        }
    }

    updateTetrisChart();
    drawTetris();
}

async function manualStep() {
    if (!currentPiece) { await newPiece(); drawTetris(); return; }
    if (!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) {
        currentPos.y++;
    } else {
        await handleMergeAndNext();
    }
    drawTetris();
}

// ─── AI Step (Instant Drop) ───────────────────────────────────

async function aiStep() {
    if (!currentPiece) { await newPiece(); drawTetris(); return; }

    // 1. Rotate to target shape
    if (targetShape) {
        const targetStr = JSON.stringify(targetShape);
        let rotations = 0;
        while (JSON.stringify(currentPiece.shape) !== targetStr && rotations < 4) {
            const rotated = rotateMatrix(currentPiece.shape);
            if (!collide(currentPos.x, currentPos.y, rotated)) {
                currentPiece.shape = rotated;
            }
            rotations++;
        }
    }

    // 2. Teleport horizontally to target X
    if (targetX !== undefined) {
        let tx = targetX;
        if (!collide(tx, currentPos.y, currentPiece.shape)) {
            currentPos.x = tx;
        } else {
            for (let d = 1; d <= 5; d++) {
                if (!collide(tx + d, currentPos.y, currentPiece.shape)) { currentPos.x = tx + d; break; }
                if (!collide(tx - d, currentPos.y, currentPiece.shape)) { currentPos.x = tx - d; break; }
            }
        }
    }

    // 3. Hard drop
    while (!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) {
        currentPos.y++;
    }

    drawTetris();
    await handleMergeAndNext();
}

// ─── Start/Stop Controls ──────────────────────────────────────

function startTetrisAI() {
    if (manualInterval) {
        clearInterval(manualInterval); manualInterval = null;
        document.getElementById('btnToggleTetrisManual').innerText = '🎮 Joacă tu (Mod Manual)';
        document.getElementById('btnToggleTetrisManual').style.background = '';
    }
    isManualMode = false;

    const btn = document.getElementById('btnStartTetrisAI');
    if (aiInterval) {
        clearInterval(aiInterval); aiInterval = null;
        btn.innerText = '🤖 Pornire AI';
        btn.style.background = '';
    } else {
        resetGame();
        newPiece().then(() => {
            aiInterval = setInterval(aiStep, getAISpeedDelay());
            btn.innerText = '⏹️ Oprire AI';
            btn.style.background = 'var(--danger)';
            updateTetrisChart();
        });
    }
}

function toggleTetrisManualMode() {
    if (aiInterval) {
        clearInterval(aiInterval); aiInterval = null;
        document.getElementById('btnStartTetrisAI').innerText = '🤖 Pornire AI';
        document.getElementById('btnStartTetrisAI').style.background = '';
    }

    const btn = document.getElementById('btnToggleTetrisManual');
    if (manualInterval) {
        clearInterval(manualInterval); manualInterval = null;
        btn.innerText = '🎮 Joacă tu (Mod Manual)';
        btn.style.background = '';
    } else {
        isManualMode = true;
        resetGame();
        newPiece().then(() => {
            manualInterval = setInterval(manualStep, getManualSpeedDelay());
            btn.innerText = '⏹️ Oprire Mod Manual';
            btn.style.background = 'var(--danger)';
            updateTetrisChart();
        });
    }
}

function resetGame() {
    board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
    score = 0;
    linesCleared = 0;
    level = 1;
    combo = 0;
    piecesPlaced = 0;
    particles = [];
    nextPiece = null;
    flashLines = [];
    flashAlpha = 0;
    tetrisHistory = [];
    document.getElementById('tetris-score').innerText = 0;
    document.getElementById('tetrisLinesDisplay').innerText = 0;
    document.getElementById('tetrisLevelDisplay').innerText = 1;

    const badge = document.getElementById('tetrisMoveBadge');
    badge.className = 'move-badge badge-neutral';
    badge.innerText = 'Se așteaptă mutarea...';
    document.getElementById('tetrisMoveExplanation').innerText =
        'Plasează o piesă sau pornește AI-ul pentru a vedea analiza în timp real.';
}

// ─── Keyboard Controls ────────────────────────────────────────

window.addEventListener('keydown', function(e) {
    if (!isManualMode || !currentPiece || !manualInterval) return;
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) e.preventDefault();

    switch (e.key) {
        case 'ArrowLeft':
            if (!collide(currentPos.x - 1, currentPos.y, currentPiece.shape)) {
                currentPos.x--;
                drawTetris();
            }
            break;
        case 'ArrowRight':
            if (!collide(currentPos.x + 1, currentPos.y, currentPiece.shape)) {
                currentPos.x++;
                drawTetris();
            }
            break;
        case 'ArrowDown':
            if (!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) {
                currentPos.y++;
                drawTetris();
            }
            break;
        case 'ArrowUp': {
            const rotated = rotateMatrix(currentPiece.shape);
            for (const dx of [0, -1, 1, -2, 2]) {
                if (!collide(currentPos.x + dx, currentPos.y, rotated)) {
                    currentPos.x += dx;
                    currentPiece.shape = rotated;
                    break;
                }
            }
            drawTetris();
            break;
        }
        case ' ':
            while (!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) currentPos.y++;
            drawTetris();
            handleMergeAndNext();
            break;
    }
});

// ─── Init ─────────────────────────────────────────────────────
const bestEl = document.getElementById('tetrisBestDisplay');
if (bestEl) bestEl.innerText = bestScore;
drawTetris();
initTetrisChart();
