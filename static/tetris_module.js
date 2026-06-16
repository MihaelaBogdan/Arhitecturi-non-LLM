// --- Tetris KNN & Manual Mode Logic ---
const tetrisCanvas = document.getElementById('tetrisCanvas');
const tctx = tetrisCanvas.getContext('2d');
const ROWS = 20;
const COLS = 10;
const BLOCK = 30;

let board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let score = 0;
let aiInterval = null;
let manualInterval = null;
let isManualMode = false;

const SHAPES = [
    [[1,1,1,1]], // I
    [[1,1],[1,1]], // O
    [[0,1,0],[1,1,1]], // T
    [[1,0,0],[1,1,1]], // L
    [[0,0,1],[1,1,1]], // J
    [[0,1,1],[1,1,0]], // S
    [[1,1,0],[0,1,1]]  // Z
];

const COLORS = ['#0f172a', '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899'];

let currentPiece = null;
let currentPos = {x:0, y:0};

let targetX = 0;
let targetShape = null;

// Learning curve chart variables
let tetrisScores = [];
let tetrisMeanScores = [];
let tetrisChartInstance = null;

function initTetrisChart() {
    const ctxChart = document.getElementById('tetrisChartCanvas').getContext('2d');
    tetrisChartInstance = new Chart(ctxChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Scor Joc', data: [], borderColor: '#3b82f6', tension: 0.1, fill: false },
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

function updateTetrisChart() {
    if (!tetrisChartInstance) return;
    
    // We append the current active score dynamically to show real-time progress
    let scoresToPlot = [...tetrisScores];
    let meansToPlot = [...tetrisMeanScores];
    
    if (currentPiece && (aiInterval || manualInterval)) {
        scoresToPlot.push(score);
        const totalSum = tetrisScores.reduce((a, b) => a + b, 0) + score;
        const runningMean = totalSum / (tetrisScores.length + 1);
        meansToPlot.push(runningMean);
    }
    
    tetrisChartInstance.data.labels = Array.from({length: scoresToPlot.length}, (_, i) => i + 1);
    tetrisChartInstance.data.datasets[0].data = scoresToPlot;
    tetrisChartInstance.data.datasets[1].data = meansToPlot;
    tetrisChartInstance.update();
}

function rotateMatrix(shape) {
    const n = shape.length;
    const m = shape[0].length;
    const rotated = Array.from({ length: m }, () => Array(n).fill(0));
    for (let r = 0; r < n; r++) {
        for (let c = 0; c < m; c++) {
            rotated[c][n - 1 - r] = shape[r][c];
        }
    }
    return rotated;
}

function getAISpeedDelay() {
    const speed = document.getElementById('tetrisSpeed').value;
    switch (speed) {
        case 'slow': return 300;
        case 'fast': return 80;
        case 'turbo': return 30;
        case 'normal':
        default: return 150;
    }
}

function getManualSpeedDelay() {
    const speed = document.getElementById('tetrisSpeed').value;
    switch (speed) {
        case 'slow': return 1500;
        case 'fast': return 400;
        case 'turbo': return 150;
        case 'normal':
        default: return 800;
    }
}

function updateTetrisSpeed() {
    if (aiInterval) {
        clearInterval(aiInterval);
        aiInterval = setInterval(aiStep, getAISpeedDelay());
    }
    if (manualInterval) {
        clearInterval(manualInterval);
        manualInterval = setInterval(manualStep, getManualSpeedDelay());
    }
}

async function requestNextMove() {
    if (!currentPiece) return;
    try {
        const res = await fetch('/api/tetris/next-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: board,
                shape: currentPiece.shape,
                model_type: 'knn'
            })
        });
        const data = await res.json();
        
        targetX = data.x;
        targetShape = data.shape;
        
        // Update UI metrics
        document.getElementById('tetrisMetricHeight').innerText = data.metrics.height;
        document.getElementById('tetrisMetricHoles').innerText = data.metrics.holes;
        document.getElementById('tetrisMetricBumpiness').innerText = data.metrics.bumpiness;
        document.getElementById('tetrisMetricRowTransitions').innerText = data.metrics.row_transitions;
        document.getElementById('tetrisMetricColTransitions').innerText = data.metrics.col_transitions;
        document.getElementById('tetrisMetricWells').innerText = data.metrics.wells;
        
        // Redraw to show the new recommendation instantly
        drawTetris();
        
    } catch (err) {
        console.error("Failed to fetch next Tetris move:", err);
    }
}

async function newPiece() {
    const id = Math.floor(Math.random() * SHAPES.length) + 1;
    const shape = SHAPES[id - 1];
    currentPiece = { 
        shape, 
        id, 
        originalShape: JSON.parse(JSON.stringify(shape)) 
    };
    currentPos = { x: Math.floor((COLS - shape[0].length)/2), y: 0 };
    
    // Request recommended placement for both manual and AI mode
    await requestNextMove();
}

function drawTetris() {
    tctx.fillStyle = '#0f172a';
    tctx.fillRect(0, 0, tetrisCanvas.width, tetrisCanvas.height);
    
    // Draw grid board
    for(let r=0; r<ROWS; r++){
        for(let c=0; c<COLS; c++){
            if(board[r][c] !== 0) {
                tctx.fillStyle = COLORS[board[r][c]];
                tctx.fillRect(c*BLOCK, r*BLOCK, BLOCK, BLOCK);
                tctx.strokeStyle = '#1e293b';
                tctx.strokeRect(c*BLOCK, r*BLOCK, BLOCK, BLOCK);
            }
        }
    }

    // Draw recommendation ghost piece (in manual mode)
    if (isManualMode && targetShape) {
        let landingY = 0;
        if (targetX >= -2 && targetX < COLS) {
            while (!collide(targetX, landingY + 1, targetShape)) {
                landingY++;
            }
            
            // Draw ghost piece (gold outline with semi-transparent fill)
            tctx.fillStyle = 'rgba(251, 191, 36, 0.2)';
            for (let r = 0; r < targetShape.length; r++) {
                for (let c = 0; c < targetShape[r].length; c++) {
                    if (targetShape[r][c]) {
                        tctx.fillRect((targetX + c) * BLOCK, (landingY + r) * BLOCK, BLOCK, BLOCK);
                        tctx.strokeStyle = 'rgba(251, 191, 36, 0.5)';
                        tctx.lineWidth = 2;
                        tctx.strokeRect((targetX + c) * BLOCK, (landingY + r) * BLOCK, BLOCK, BLOCK);
                    }
                }
            }
            tctx.lineWidth = 1;
        }
    }

    // Draw active falling piece
    if (currentPiece) {
        tctx.fillStyle = COLORS[currentPiece.id];
        for(let r=0; r<currentPiece.shape.length; r++){
            for(let c=0; c<currentPiece.shape[r].length; c++){
                if(currentPiece.shape[r][c]) {
                    tctx.fillRect((currentPos.x + c)*BLOCK, (currentPos.y + r)*BLOCK, BLOCK, BLOCK);
                    tctx.strokeStyle = '#1e293b';
                    tctx.strokeRect((currentPos.x + c)*BLOCK, (currentPos.y + r)*BLOCK, BLOCK, BLOCK);
                }
            }
        }
    }
}

function collide(x, y, shape) {
    for(let r=0; r<shape.length; r++){
        for(let c=0; c<shape[r].length; c++){
            if(shape[r][c]) {
                let nx = x + c;
                let ny = y + r;
                if(nx < 0 || nx >= COLS || ny >= ROWS || (ny >= 0 && board[ny][nx] !== 0)) {
                    return true;
                }
            }
        }
    }
    return false;
}

function merge() {
    for(let r=0; r<currentPiece.shape.length; r++){
        for(let c=0; c<currentPiece.shape[r].length; c++){
            if(currentPiece.shape[r][c]) {
                board[currentPos.y + r][currentPos.x + c] = currentPiece.id;
            }
        }
    }
    // Check lines
    let linesClearedThisTurn = 0;
    for(let r=ROWS-1; r>=0; r--){
        let full = true;
        for(let c=0; c<COLS; c++) {
            if(board[r][c] === 0) full = false;
        }
        if(full) {
            board.splice(r, 1);
            board.unshift(Array(COLS).fill(0));
            score += 100;
            linesClearedThisTurn++;
            document.getElementById('tetris-score').innerText = score;
            r++;
        }
    }
    if (linesClearedThisTurn > 0) {
        updateTetrisChart();
    }
}

async function evaluateMove(boardBefore, originalShape, chosenX, chosenShape) {
    try {
        const res = await fetch('/api/tetris/evaluate-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: boardBefore,
                shape: originalShape,
                chosen_x: chosenX,
                chosen_shape: chosenShape
            })
        });
        const data = await res.json();
        
        // Update live metrics from evaluation
        if (data.metrics) {
            document.getElementById('tetrisMetricHeight').innerText = data.metrics.height;
            document.getElementById('tetrisMetricHoles').innerText = data.metrics.holes;
            document.getElementById('tetrisMetricBumpiness').innerText = data.metrics.bumpiness;
            document.getElementById('tetrisMetricRowTransitions').innerText = data.metrics.row_transitions;
            document.getElementById('tetrisMetricColTransitions').innerText = data.metrics.col_transitions;
            document.getElementById('tetrisMetricWells').innerText = data.metrics.wells;
        }
        
        // Update review card
        const badge = document.getElementById('tetrisMoveBadge');
        badge.innerText = data.classification;
        
        badge.className = "move-badge";
        if (data.classification === "Excelentă") {
            badge.classList.add('badge-excellent');
            badge.innerText = "Excelentă 🌟";
        } else if (data.classification === "Bună") {
            badge.classList.add('badge-good');
            badge.innerText = "Bună ✅";
        } else if (data.classification === "Medie") {
            badge.classList.add('badge-okay');
            badge.innerText = "Medie ⚠️";
        } else if (data.classification === "Greșeală") {
            badge.classList.add('badge-blunder');
            badge.innerText = "Greșeală ❌";
        } else {
            badge.classList.add('badge-neutral');
        }
        
        document.getElementById('tetrisMoveExplanation').innerText = data.explanation;
        
    } catch (err) {
        console.error("Failed to evaluate move:", err);
    }
}

async function handleMergeAndNext() {
    if (currentPiece) {
        const boardBeforeMerge = JSON.parse(JSON.stringify(board));
        const originalShape = currentPiece.originalShape;
        const chosenShape = currentPiece.shape;
        const chosenX = currentPos.x;
        
        await evaluateMove(boardBeforeMerge, originalShape, chosenX, chosenShape);
        merge();
    }
    
    currentPiece = null;
    await newPiece();
    
    // Game Over Check
    if (collide(currentPos.x, currentPos.y, currentPiece.shape)) {
        // Record score
        tetrisScores.push(score);
        const avg = tetrisScores.reduce((a, b) => a + b, 0) / tetrisScores.length;
        tetrisMeanScores.push(avg);
        updateTetrisChart();
        
        board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        score = 0;
        document.getElementById('tetris-score').innerText = score;
        
        // Reset move evaluation UI
        const badge = document.getElementById('tetrisMoveBadge');
        badge.className = "move-badge badge-neutral";
        badge.innerText = "Game Over!";
        document.getElementById('tetrisMoveExplanation').innerText = "Jocul s-a terminat. Pornește din nou pentru a juca!";
        
        if (isManualMode && manualInterval) {
            clearInterval(manualInterval);
            manualInterval = null;
            const btnManual = document.getElementById('btnToggleTetrisManual');
            btnManual.innerText = "🎮 Joacă tu (Mod Manual)";
            btnManual.style.background = "";
        }
        
        if (!isManualMode && aiInterval) {
            clearInterval(aiInterval);
            aiInterval = null;
            const btnAI = document.getElementById('btnStartTetrisAI');
            btnAI.innerText = "🤖 Pornire KNN AI";
            btnAI.style.background = "";
        }
    }
    
    updateTetrisChart();
    drawTetris();
}

async function manualStep() {
    if (!currentPiece) {
        await newPiece();
        drawTetris();
        return;
    }
    
    if (!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) {
        currentPos.y++;
    } else {
        await handleMergeAndNext();
    }
    drawTetris();
}

async function aiStep() {
    if(!currentPiece) {
        await newPiece();
        drawTetris();
        return;
    }
    
    // 1. Rotate piece to match target rotation shape
    if (targetShape && JSON.stringify(currentPiece.shape) !== JSON.stringify(targetShape)) {
        const rotated = rotateMatrix(currentPiece.shape);
        if (!collide(currentPos.x, currentPos.y, rotated)) {
            currentPiece.shape = rotated;
        }
    }
    
    // 2. Slide horizontally to target column
    if (currentPos.x < targetX) {
        if (!collide(currentPos.x + 1, currentPos.y, currentPiece.shape)) {
            currentPos.x++;
        }
    } else if (currentPos.x > targetX) {
        if (!collide(currentPos.x - 1, currentPos.y, currentPiece.shape)) {
            currentPos.x--;
        }
    }
    
    // 3. Move down
    if(!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) {
        currentPos.y++;
    } else {
        await handleMergeAndNext();
    }
    drawTetris();
}

function startTetrisAI() {
    if (manualInterval) {
        clearInterval(manualInterval);
        manualInterval = null;
        const btnManual = document.getElementById('btnToggleTetrisManual');
        btnManual.innerText = "🎮 Joacă tu (Mod Manual)";
        btnManual.style.background = "";
    }
    
    isManualMode = false;
    
    const btnAI = document.getElementById('btnStartTetrisAI');
    if(aiInterval) {
        clearInterval(aiInterval);
        aiInterval = null;
        btnAI.innerText = "🤖 Pornire KNN AI";
        btnAI.style.background = "";
    } else {
        board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        score = 0;
        document.getElementById('tetris-score').innerText = score;
        
        const badge = document.getElementById('tetrisMoveBadge');
        badge.className = "move-badge badge-neutral";
        badge.innerText = "Se așteaptă mutarea...";
        document.getElementById('tetrisMoveExplanation').innerText = "Plasează o piesă sau pornește AI-ul pentru a vedea analiza în timp real a mutărilor.";
        
        newPiece().then(() => {
            aiInterval = setInterval(aiStep, getAISpeedDelay());
            btnAI.innerText = "⏹️ Oprire AI";
            btnAI.style.background = "var(--danger)";
            updateTetrisChart();
        });
    }
}

function toggleTetrisManualMode() {
    if (aiInterval) {
        clearInterval(aiInterval);
        aiInterval = null;
        const btnAI = document.getElementById('btnStartTetrisAI');
        btnAI.innerText = "🤖 Pornire KNN AI";
        btnAI.style.background = "";
    }
    
    const btnManual = document.getElementById('btnToggleTetrisManual');
    if (manualInterval) {
        clearInterval(manualInterval);
        manualInterval = null;
        btnManual.innerText = "🎮 Joacă tu (Mod Manual)";
        btnManual.style.background = "";
    } else {
        isManualMode = true;
        board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        score = 0;
        document.getElementById('tetris-score').innerText = score;
        
        const badge = document.getElementById('tetrisMoveBadge');
        badge.className = "move-badge badge-neutral";
        badge.innerText = "Se așteaptă mutarea...";
        document.getElementById('tetrisMoveExplanation').innerText = "Folosește săgețile pentru a deplasa/roti piesa și Space pentru drop instant.";
        
        newPiece().then(() => {
            manualInterval = setInterval(manualStep, getManualSpeedDelay());
            btnManual.innerText = "⏹️ Oprire Mod Manual";
            btnManual.style.background = "var(--danger)";
            updateTetrisChart();
        });
    }
}

// Keyboard controls
window.addEventListener('keydown', function(e) {
    if (!isManualMode || !currentPiece || !manualInterval) return;
    
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        e.preventDefault();
    }
    
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
        case 'ArrowUp':
            const rotated = rotateMatrix(currentPiece.shape);
            if (!collide(currentPos.x, currentPos.y, rotated)) {
                currentPiece.shape = rotated;
                drawTetris();
            }
            break;
        case ' ':
            while (!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) {
                currentPos.y++;
            }
            drawTetris();
            handleMergeAndNext();
            break;
    }
});

// Initialization
drawTetris();
initTetrisChart();
