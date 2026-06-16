// --- Basic Tetris AI Logic ---
const tetrisCanvas = document.getElementById('tetrisCanvas');
const tctx = tetrisCanvas.getContext('2d');
const ROWS = 20;
const COLS = 10;
const BLOCK = 30;

let board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
let score = 0;
let aiInterval = null;

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

let currentTetrisModel = 'mlp';
let targetX = 0;
let targetShape = null;

function setTetrisModel(model) {
    currentTetrisModel = model;
    document.getElementById('btnTetrisMLP').classList.toggle('active', model === 'mlp');
    document.getElementById('btnTetrisTree').classList.toggle('active', model === 'tree');
    document.getElementById('btnTetrisKNN').classList.toggle('active', model === 'knn');
    
    document.getElementById('tetrisRulesCard').style.display = model === 'tree' ? 'block' : 'none';
    
    const desc = document.getElementById('tetrisModelDesc');
    if (model === 'mlp') {
        desc.textContent = 'Rețeaua MLP (Multi-Layer Perceptron - Slide 28) prezice calitatea unei plasări pe baza caracteristicilor tablei.';
    } else if (model === 'tree') {
        desc.textContent = 'Decision Tree Regressor (Slide 12) prezice calitatea plasării, afișând regulile logice active (criteriul MSE).';
    } else if (model === 'knn') {
        desc.textContent = 'KNN Regressor (Slide 15) prezice calitatea plasării făcând media euristică a celor mai apropiate 5 table din istoric.';
    }
    
    // Recalculate target for current piece with new model if game is active
    if (currentPiece && aiInterval) {
        requestNextMove();
    }
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

async function requestNextMove() {
    if (!currentPiece) return;
    try {
        const res = await fetch('/api/tetris/next-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: board,
                shape: currentPiece.shape,
                model_type: currentTetrisModel
            })
        });
        const data = await res.json();
        
        targetX = data.x;
        targetShape = data.shape;
        
        // Update UI metrics
        document.getElementById('tetrisMetricHeight').innerText = data.metrics.height;
        document.getElementById('tetrisMetricHoles').innerText = data.metrics.holes;
        document.getElementById('tetrisMetricBumpiness').innerText = data.metrics.bumpiness;
        
        if (data.explanation) {
            document.getElementById('tetrisActiveRule').innerText = data.explanation;
        }
    } catch (err) {
        console.error("Failed to fetch next Tetris move:", err);
    }
}

async function newPiece() {
    const id = Math.floor(Math.random() * SHAPES.length) + 1;
    const shape = SHAPES[id - 1];
    currentPiece = { shape, id };
    currentPos = { x: Math.floor((COLS - shape[0].length)/2), y: 0 };
    
    // Instantly request the optimal path from backend models
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
    for(let r=ROWS-1; r>=0; r--){
        let full = true;
        for(let c=0; c<COLS; c++) {
            if(board[r][c] === 0) full = false;
        }
        if(full) {
            board.splice(r, 1);
            board.unshift(Array(COLS).fill(0));
            score += 100;
            document.getElementById('tetris-score').innerText = score;
            r++;
        }
    }
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
        merge();
        currentPiece = null;
        await newPiece();
        
        // Game Over Check
        if(collide(currentPos.x, currentPos.y, currentPiece.shape)) {
            board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
            score = 0;
            document.getElementById('tetris-score').innerText = score;
            await requestNextMove();
        }
    }
    drawTetris();
}

function startTetrisAI() {
    const btn = document.getElementById('btnStartTetrisAI');
    if(aiInterval) {
        clearInterval(aiInterval);
        aiInterval = null;
        btn.innerText = "▶️ Pornire AI";
        btn.style.background = "";
    } else {
        board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        score = 0;
        document.getElementById('tetris-score').innerText = score;
        newPiece().then(() => {
            aiInterval = setInterval(aiStep, 150);
            btn.innerText = "⏹️ Oprire AI";
            btn.style.background = "var(--danger)";
        });
    }
}

drawTetris();
