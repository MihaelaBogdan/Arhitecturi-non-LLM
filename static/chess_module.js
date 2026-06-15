// --- Chess Module: Minimax + CNN + Educational Features ---
let chessBoard = null;
const chessGame = new Chess();
let currentAIMode = 'minimax';  // 'minimax' | 'cnn'
let lastAIMove = null;

// ---- AI Mode ----
function setAIMode(mode) {
    currentAIMode = mode;
    document.getElementById('btnMinimax').classList.toggle('active', mode === 'minimax');
    document.getElementById('btnCNN').classList.toggle('active', mode === 'cnn');
    const desc = document.getElementById('aiModeDesc');
    if (mode === 'minimax') {
        desc.textContent = 'Minimax explorează pozițiile viitoare sistematic cu Alpha-Beta Pruning.';
    } else {
        desc.textContent = 'CNN evaluează pozițiile prin pattern-uri învățate din mii de partide simulate.';
    }
}

// ---- Status ----
function updateChessStatus(isAIThinking) {
    const statusEl = document.getElementById('chessStatus');
    if (!statusEl) return;

    if (chessGame.in_checkmate()) {
        statusEl.innerText = `Șah Mat! A câștigat ${chessGame.turn() === 'w' ? 'Negrul (AI)' : 'Albul (Tu)'}.`;
        statusEl.style.color = 'var(--danger)';
        statusEl.style.background = 'rgba(239,68,68,0.2)';
    } else if (chessGame.in_draw() || chessGame.in_stalemate() || chessGame.in_threefold_repetition()) {
        statusEl.innerText = 'Remiză!';
        statusEl.style.color = 'var(--text-muted)';
        statusEl.style.background = '';
    } else {
        if (isAIThinking) {
            statusEl.innerText = currentAIMode === 'cnn' ? '🧠 CNN calculează...' : '🌲 Minimax calculează...';
            statusEl.style.color = 'var(--primary)';
            statusEl.style.background = '';
        } else {
            statusEl.innerText = 'Rândul tău (Alb)';
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
        if (hintBtn) hintBtn.textContent = '💡 Sugerează-mi o mutare';
        if (data.hint) {
            const from = data.hint.substring(0, 2);
            const to   = data.hint.substring(2, 4);
            // Highlight suggestion
            removeHighlights();
            $('.square-' + from).addClass('highlight-hint-from');
            $('.square-' + to).addClass('highlight-hint-to');
            document.getElementById('moveExplanation').textContent =
                `💡 Sugestie: mută de pe ${from.toUpperCase()} pe ${to.toUpperCase()}`;
            // Remove hint highlight after 3s
            setTimeout(removeHighlights, 3000);
        }
    })
    .catch(err => {
        if (hintBtn) hintBtn.textContent = '💡 Sugerează-mi o mutare';
        console.error('Hint error:', err);
    });
}

// ---- AI Move ----
function makeAIMove() {
    updateChessStatus(true);
    const depth = parseInt(document.getElementById('chessDifficulty').value) || 3;
    const endpoint = currentAIMode === 'cnn' ? '/api/chess/cnn-move' : '/api/chess/move';

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), depth: depth })
    })
    .then(r => r.json())
    .then(data => {
        const moveKey = currentAIMode === 'cnn' ? data.move : data.move;
        if (moveKey) {
            chessGame.move(moveKey, { sloppy: true });
            chessBoard.position(chessGame.fen());
            lastAIMove = moveKey;
            // Auto-analyze after every AI move
            analyzePosition(moveKey);
        }
        updateChessStatus(false);
    })
    .catch(err => {
        console.error(err);
        document.getElementById('chessStatus').innerText = 'Eroare la AI.';
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
}
