// --- Chess Logic ---
let chessBoard = null;
const chessGame = new Chess();

function updateChessStatus(isAIThinking) {
    const statusEl = document.getElementById('chessStatus');
    if (!statusEl) return;
    
    if (chessGame.in_checkmate()) {
        statusEl.innerText = `Șah Mat! A câștigat ${chessGame.turn() === 'w' ? 'Negrul (AI)' : 'Albul (Tu)'}.`;
        statusEl.style.color = 'var(--danger)';
        statusEl.style.background = 'rgba(239, 68, 68, 0.2)';
    } else if (chessGame.in_draw() || chessGame.in_stalemate() || chessGame.in_threefold_repetition()) {
        statusEl.innerText = 'Remiză!';
        statusEl.style.color = 'var(--text-muted)';
    } else {
        if (isAIThinking) {
            statusEl.innerText = 'AI-ul calculează...';
            statusEl.style.color = 'var(--primary)';
        } else {
            statusEl.innerText = 'Rândul tău (Alb)';
            if (chessGame.in_check()) {
                statusEl.innerText += ' - ȘAH!';
                statusEl.style.color = 'var(--danger)';
            } else {
                statusEl.style.color = 'var(--accent)';
            }
        }
    }
}

function makeAIMove() {
    updateChessStatus(true);
    
    const depth = parseInt(document.getElementById('chessDifficulty').value) || 3;
    
    fetch('/api/chess/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), depth: depth })
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

function removeHighlights() {
    $('#board .square-55d63').removeClass('highlight-square highlight-source');
}

function onDragStart(source, piece, position, orientation) {
    if (chessGame.game_over()) return false;
    if (piece.search(/^b/) !== -1) return false;
}

function onMouseoverSquare(square, piece) {
    if (chessGame.game_over()) return;
    if (!piece || piece.search(/^b/) !== -1) return; // Only highlight White

    const moves = chessGame.moves({ square: square, verbose: true });
    if (moves.length === 0) return;

    $('.square-' + square).addClass('highlight-source');
    for (let i = 0; i < moves.length; i++) {
        $('.square-' + moves[i].to).addClass('highlight-square');
    }
}

function onMouseoutSquare(square, piece) {
    removeHighlights();
}

function onDrop(source, target) {
    removeHighlights();
    const move = chessGame.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    updateChessStatus(false);
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
        onMouseoverSquare: onMouseoverSquare,
        onMouseoutSquare: onMouseoutSquare,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    };
    chessBoard = Chessboard('board', config);
    updateChessStatus(false);
});

function resetChessGame() {
    chessGame.reset();
    chessBoard.start();
    updateChessStatus(false);
}
