// --- Chess Module: Minimax + CNN + Educational Features ---
let chessBoard = null;
const chessGame = new Chess();
let currentAIMode = 'minimax';  // 'minimax' | 'cnn' (Player vs AI)
let chessGameMode = 'pvai';     // 'pvai' | 'aivsai'
let whiteAIMode = 'minimax';    // 'minimax' | 'cnn'
let blackAIMode = 'minimax';    // 'minimax' | 'cnn'
let isAIvsAIPlaying = false;
let aivsaiTimeout = null;
let lastAIMove = null;

let cnnAnimationInterval = null;

function startCNNAnimation() {
    stopCNNAnimation();
    const layers = ['input', 'conv1', 'conv2', 'conv3', 'flatten', 'dense', 'output'];
    let currentIdx = 0;
    
    // Initial clear
    document.querySelectorAll('.cnn-layer').forEach(el => {
        el.classList.remove('pulse-layer', 'active-layer');
    });
    
    // Light up the input layer
    const inputLayer = document.querySelector('.cnn-layer[data-layer="input"]');
    if (inputLayer) inputLayer.classList.add('active-layer');
    
    cnnAnimationInterval = setInterval(() => {
        // Remove pulse from all layers
        document.querySelectorAll('.cnn-layer').forEach(el => el.classList.remove('pulse-layer'));
        
        // Pulse current layer
        const layerName = layers[currentIdx];
        const el = document.querySelector(`.cnn-layer[data-layer="${layerName}"]`);
        if (el) {
            el.classList.add('pulse-layer');
        }
        
        // Progress to next layer
        currentIdx = (currentIdx + 1) % layers.length;
    }, 120);
}

function stopCNNAnimation() {
    if (cnnAnimationInterval) {
        clearInterval(cnnAnimationInterval);
        cnnAnimationInterval = null;
    }
    // Reset layers to active-layer state
    document.querySelectorAll('.cnn-layer').forEach(el => {
        el.classList.remove('pulse-layer');
        el.classList.add('active-layer');
    });
}

// ---- AI Mode ----
function setAIMode(mode) {
    currentAIMode = mode;
    document.getElementById('btnMinimax').classList.toggle('active', mode === 'minimax');
    document.getElementById('btnCNN').classList.toggle('active', mode === 'cnn');
    const desc = document.getElementById('aiModeDesc');
    const visualizer = document.getElementById('cnnVisualizer');
    
    if (mode === 'minimax') {
        desc.textContent = 'Minimax explorează pozițiile viitoare sistematic cu Alpha-Beta Pruning.';
        if (visualizer) $(visualizer).slideUp(300);
        stopCNNAnimation();
    } else {
        desc.textContent = 'CNN evaluează pozițiile prin pattern-uri învățate din mii de partide simulate.';
        if (visualizer) {
            $(visualizer).slideDown(300);
            document.querySelectorAll('.cnn-layer').forEach(el => el.classList.add('active-layer'));
        }
    }
}

// ---- Status ----
function updateChessStatus(isAIThinking) {
    const statusEl = document.getElementById('chessStatus');
    if (!statusEl) return;

    if (chessGame.in_checkmate()) {
        statusEl.innerText = `Șah Mat! A câștigat ${chessGame.turn() === 'w' ? 'Negrul (AI)' : (chessGameMode === 'aivsai' ? 'Albul (AI)' : 'Albul (Tu)')}.`;
        statusEl.style.color = 'var(--danger)';
        statusEl.style.background = 'rgba(239,68,68,0.2)';
    } else if (chessGame.in_draw() || chessGame.in_stalemate() || chessGame.in_threefold_repetition()) {
        statusEl.innerText = 'Remiză!';
        statusEl.style.color = 'var(--text-muted)';
        statusEl.style.background = '';
    } else {
        if (isAIThinking) {
            const turn = chessGame.turn(); // 'w' or 'b'
            const model = (chessGameMode === 'aivsai') ? (turn === 'w' ? whiteAIMode : blackAIMode) : currentAIMode;
            statusEl.innerText = model === 'cnn' ? 'CNN calculează...' : 'Minimax calculează...';
            statusEl.style.color = 'var(--primary)';
            statusEl.style.background = '';
        } else {
            if (chessGameMode === 'aivsai') {
                statusEl.innerText = chessGame.turn() === 'w' ? 'Rândul Albului (AI)' : 'Rândul Negrului (AI)';
            } else {
                statusEl.innerText = 'Rândul tău (Alb)';
            }
            if (chessGame.in_check()) {
                statusEl.innerText += ' — ȘAH!';
                statusEl.style.color = 'var(--danger)';
            } else {
                statusEl.style.color = 'var(--accent)';
            }
        }
    }
}

// ---- Evaluation Bar ----
function updateEvalBar(score) {
    // score: positive = white advantage (range roughly -500 to +500 from API)
    const clamped = Math.max(-300, Math.min(300, score));
    const whitePct = ((clamped + 300) / 600) * 100;
    const blackPct = 100 - whitePct;

    document.getElementById('evalBarWhite').style.width = whitePct + '%';
    document.getElementById('evalBarBlack').style.width = blackPct + '%';

    const label = score > 0 ? `+${score.toFixed(1)}` : score.toFixed(1);
    document.getElementById('evalScore').textContent = label;
}

// ---- Analyze position (auto-called after every AI move) ----
function analyzePosition(lastMove = null) {
    const move = lastMove || lastAIMove;
    fetch('/api/chess/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), last_move: move })
    })
    .then(r => r.json())
    .then(data => {
        // Eval bar
        updateEvalBar(data.score || 0);

        // Material balance
        if (data.material) {
            document.getElementById('whiteCaptured').textContent = data.material.white_captured || '—';
            document.getElementById('blackCaptured').textContent = data.material.black_captured || '—';
        }

        // Opening
        if (data.opening) {
            document.getElementById('openingName').textContent = data.opening;
        }

        // Move explanation
        if (data.explanation) {
            document.getElementById('moveExplanation').textContent = data.explanation;
        }
    })
    .catch(err => console.error('Analyze error:', err));
}

// ---- Hint ----
function requestHint() {
    const depth = parseInt(document.getElementById('chessDifficulty').value) || 2;
    const hintBtn = document.querySelector('button[onclick="requestHint()"]');
    if (hintBtn) hintBtn.textContent = '⏳ Se calculează...';

    fetch('/api/chess/hint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), depth: Math.min(depth, 2) })
    })
    .then(r => r.json())
    .then(data => {
        if (hintBtn) hintBtn.textContent = 'Sugerează-mi o mutare';
        if (data.hint) {
            const from = data.hint.substring(0, 2);
            const to   = data.hint.substring(2, 4);
            // Highlight suggestion
            removeHighlights();
            $('.square-' + from).addClass('highlight-hint-from');
            $('.square-' + to).addClass('highlight-hint-to');
            document.getElementById('moveExplanation').textContent =
                `Sugestie: mută de pe ${from.toUpperCase()} pe ${to.toUpperCase()}`;
            // Remove hint highlight after 3s
            setTimeout(removeHighlights, 3000);
        }
    })
    .catch(err => {
        if (hintBtn) hintBtn.textContent = 'Sugerează-mi o mutare';
        console.error('Hint error:', err);
    });
}

// ---- AI Move ----
function makeAIMove() {
    updateChessStatus(true);
    if (currentAIMode === 'cnn') {
        startCNNAnimation();
    }
    const depth = parseInt(document.getElementById('chessDifficulty').value) || 3;
    const endpoint = currentAIMode === 'cnn' ? '/api/chess/cnn-move' : '/api/chess/move';

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), depth: depth })
    })
    .then(r => r.json())
    .then(data => {
        const moveKey = data.move;
        if (moveKey) {
            const move = chessGame.move(moveKey, { sloppy: true });
            chessBoard.position(chessGame.fen());
            lastAIMove = moveKey;
            if (typeof addTerminalLog === 'function') {
                const moveSan = move ? move.san : moveKey;
                addTerminalLog(`[CHESS] AI (${currentAIMode.toUpperCase()}) move: ${moveSan}`);
            }
            // Auto-analyze after every AI move
            analyzePosition(moveKey);
        }
        updateChessStatus(false);
        stopCNNAnimation();
    })
    .catch(err => {
        console.error(err);
        document.getElementById('chessStatus').innerText = 'Eroare la AI.';
        stopCNNAnimation();
    });
}

// ---- Board Interaction ----
function removeHighlights() {
    $('#board .square-55d63').removeClass(
        'highlight-square highlight-source highlight-hint-from highlight-hint-to'
    );
}

function onDragStart(source, piece) {
    if (chessGame.game_over()) return false;
    if (chessGameMode === 'aivsai') return false; // block drag in AI vs AI mode
    if (piece.search(/^b/) !== -1) return false;
}

function onMouseoverSquare(square, piece) {
    if (chessGame.game_over()) return;
    if (!piece || piece.search(/^b/) !== -1) return;
    const moves = chessGame.moves({ square: square, verbose: true });
    if (moves.length === 0) return;
    $('.square-' + square).addClass('highlight-source');
    for (let i = 0; i < moves.length; i++) {
        $('.square-' + moves[i].to).addClass('highlight-square');
    }
}

function onMouseoutSquare() {
    removeHighlights();
}

function onDrop(source, target) {
    removeHighlights();
    const move = chessGame.move({ from: source, to: target, promotion: 'q' });
    if (move === null) return 'snapback';
    if (typeof addTerminalLog === 'function') {
        addTerminalLog(`[CHESS] Player move: ${move.san} (${source}->${target})`);
    }
    lastAIMove = null;
    updateChessStatus(false);
    analyzePosition(null);
    window.setTimeout(makeAIMove, 250);
}

function onSnapEnd() {
    chessBoard.position(chessGame.fen());
}

// ---- Init ----
$(document).ready(function () {
    const config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        onMouseoverSquare: onMouseoverSquare,
        onMouseoutSquare: onMouseoutSquare,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    };
    chessBoard = Chessboard('board', config);
    updateChessStatus(false);
    analyzePosition(null);
});

function resetChessGame() {
    if (isAIvsAIPlaying) {
        toggleAIvsAIAutoPlay();
    }
    chessGame.reset();
    chessBoard.start();
    lastAIMove = null;
    removeHighlights();
    updateChessStatus(false);
    document.getElementById('moveExplanation').textContent = 'Jocul nu a început.';
    document.getElementById('openingName').textContent = 'Poziție de start';
    document.getElementById('whiteCaptured').textContent = '—';
    document.getElementById('blackCaptured').textContent = '—';
    updateEvalBar(0);
    stopCNNAnimation();
}

// ---- AI vs AI controls ----
function setChessGameMode(mode) {
    chessGameMode = mode;
    
    if (isAIvsAIPlaying) {
        toggleAIvsAIAutoPlay();
    }
    
    document.getElementById('btnPlayerVSai').classList.toggle('active', mode === 'pvai');
    document.getElementById('btnAIvsAI').classList.toggle('active', mode === 'aivsai');
    
    const playerAiCard = document.getElementById('chessPlayerAiModeCard');
    const aiConfigCard = document.getElementById('chessAiConfigCard');
    const hintBtn = document.getElementById('btnChessHint');
    const startBtn = document.getElementById('btnStartAIvsAI');
    const visualizer = document.getElementById('cnnVisualizer');
    
    if (mode === 'pvai') {
        if (playerAiCard) playerAiCard.style.display = 'block';
        if (aiConfigCard) aiConfigCard.style.display = 'none';
        if (hintBtn) hintBtn.style.display = 'block';
        if (startBtn) startBtn.style.display = 'none';
        
        if (currentAIMode === 'cnn') {
            if (visualizer) $(visualizer).slideDown(300);
        } else {
            if (visualizer) $(visualizer).slideUp(300);
        }
    } else {
        if (playerAiCard) playerAiCard.style.display = 'none';
        if (aiConfigCard) aiConfigCard.style.display = 'block';
        if (hintBtn) hintBtn.style.display = 'none';
        if (startBtn) startBtn.style.display = 'block';
        
        if (whiteAIMode === 'cnn' || blackAIMode === 'cnn') {
            if (visualizer) $(visualizer).slideDown(300);
        } else {
            if (visualizer) $(visualizer).slideUp(300);
        }
    }
    updateChessStatus(false);
}

function setWhiteAI(mode) {
    whiteAIMode = mode;
    document.getElementById('btnWhiteMinimax').classList.toggle('active', mode === 'minimax');
    document.getElementById('btnWhiteCNN').classList.toggle('active', mode === 'cnn');
    
    const visualizer = document.getElementById('cnnVisualizer');
    if (whiteAIMode === 'cnn' || blackAIMode === 'cnn') {
        if (visualizer) $(visualizer).slideDown(300);
    } else {
        if (visualizer) $(visualizer).slideUp(300);
    }
}

function setBlackAI(mode) {
    blackAIMode = mode;
    document.getElementById('btnBlackMinimax').classList.toggle('active', mode === 'minimax');
    document.getElementById('btnBlackCNN').classList.toggle('active', mode === 'cnn');
    
    const visualizer = document.getElementById('cnnVisualizer');
    if (whiteAIMode === 'cnn' || blackAIMode === 'cnn') {
        if (visualizer) $(visualizer).slideDown(300);
    } else {
        if (visualizer) $(visualizer).slideUp(300);
    }
}

function toggleAIvsAIAutoPlay() {
    isAIvsAIPlaying = !isAIvsAIPlaying;
    updateAIvsAIButtonState();
    
    if (isAIvsAIPlaying) {
        if (chessGame.game_over()) {
            resetChessGame();
            isAIvsAIPlaying = true;
            updateAIvsAIButtonState();
        }
        stepAIvsAI();
    } else {
        if (aivsaiTimeout) {
            clearTimeout(aivsaiTimeout);
            aivsaiTimeout = null;
        }
        updateChessStatus(false);
        stopCNNAnimation();
    }
}

function updateAIvsAIButtonState() {
    const btn = document.getElementById('btnStartAIvsAI');
    if (!btn) return;
    if (isAIvsAIPlaying) {
        btn.innerHTML = 'Oprire AI contra AI';
        btn.style.backgroundColor = 'var(--danger)';
    } else {
        btn.innerHTML = 'Pornire AI contra AI';
        btn.style.backgroundColor = 'var(--accent)';
    }
}

function stepAIvsAI() {
    if (!isAIvsAIPlaying || chessGame.game_over()) {
        isAIvsAIPlaying = false;
        updateAIvsAIButtonState();
        updateChessStatus(false);
        return;
    }

    const turn = chessGame.turn(); // 'w' or 'b'
    const model = (turn === 'w') ? whiteAIMode : blackAIMode;
    
    updateChessStatus(true);
    if (model === 'cnn') {
        startCNNAnimation();
    } else {
        stopCNNAnimation();
    }

    const depth = parseInt(document.getElementById('chessDifficulty').value) || 3;
    const endpoint = model === 'cnn' ? '/api/chess/cnn-move' : '/api/chess/move';

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), depth: model === 'cnn' ? Math.min(depth, 2) : depth })
    })
    .then(r => r.json())
    .then(data => {
        if (!isAIvsAIPlaying) return;

        const moveKey = data.move;
        if (moveKey) {
            chessGame.move(moveKey, { sloppy: true });
            chessBoard.position(chessGame.fen());
            lastAIMove = moveKey;
            analyzePosition(moveKey);
        }
        
        updateChessStatus(false);
        stopCNNAnimation();

        if (!chessGame.game_over() && isAIvsAIPlaying) {
            aivsaiTimeout = window.setTimeout(stepAIvsAI, 1000);
        } else {
            isAIvsAIPlaying = false;
            updateAIvsAIButtonState();
        }
    })
    .catch(err => {
        console.error("AI vs AI move error:", err);
        isAIvsAIPlaying = false;
        updateAIvsAIButtonState();
        stopCNNAnimation();
    });
}
