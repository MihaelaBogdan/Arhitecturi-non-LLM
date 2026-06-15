// --- Basic Tetris AI Logic ---
// For presentation purposes, this creates a simple visual Tetris AI
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

function newPiece() {
    const id = Math.floor(Math.random() * SHAPES.length) + 1;
    const shape = SHAPES[id - 1];
    currentPiece = { shape, id };
    currentPos = { x: Math.floor((COLS - shape[0].length)/2), y: 0 };
}

function drawTetris() {
    tctx.fillStyle = '#0f172a';
    tctx.fillRect(0, 0, tetrisCanvas.width, tetrisCanvas.height);
    
    // Draw board
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

    // Draw piece
    if (currentPiece) {
        tctx.fillStyle = COLORS[currentPiece.id];
        for(let r=0; r<currentPiece.shape.length; r++){
            for(let c=0; c<currentPiece.shape[r].length; c++){
                if(currentPiece.shape[r][c]) {
                    tctx.fillRect((currentPos.x + c)*BLOCK, (currentPos.y + r)*BLOCK, BLOCK, BLOCK);
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

// Very simple AI heuristics
function aiStep() {
    if(!currentPiece) newPiece();
    
    // Move down
    if(!collide(currentPos.x, currentPos.y + 1, currentPiece.shape)) {
        currentPos.y++;
    } else {
        merge();
        newPiece();
        // Check game over
        if(collide(currentPos.x, currentPos.y, currentPiece.shape)) {
            board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
            score = 0;
            document.getElementById('tetris-score').innerText = score;
        }
    }

    // Since it's a presentation demo, just randomly move left/right to look active
    // A real heuristic searches all rotations and drops.
    if(Math.random() > 0.5) {
        let dir = Math.random() > 0.5 ? 1 : -1;
        if(!collide(currentPos.x + dir, currentPos.y, currentPiece.shape)) {
            currentPos.x += dir;
        }
    }
    drawTetris();
}

function startTetrisAI() {
    if(aiInterval) {
        clearInterval(aiInterval);
        aiInterval = null;
        event.target.innerText = "Start AI";
        event.target.classList.remove("danger");
    } else {
        board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
        score = 0;
        document.getElementById('tetris-score').innerText = score;
        aiInterval = setInterval(aiStep, 100);
        event.target.innerText = "Stop AI";
        event.target.style.background = "var(--danger)";
    }
}

drawTetris();
