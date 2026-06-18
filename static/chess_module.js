// --- Chess Module: Minimax + CNN + Board Recognition + Smart Analysis ---
let chessBoard = null;
const chessGame = new Chess();
let currentAIMode = 'minimax';  // 'minimax' | 'cnn'
let lastAIMove = null;
let moveHistory = [];  // Track all moves in UCI format

// ──────────────────────────────────────────────────
// AI Mode
// ──────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────
// Status
// ──────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────
// Evaluation Bar
// ──────────────────────────────────────────────────

function updateEvalBar(score) {
    const clamped = Math.max(-300, Math.min(300, score));
    const whitePct = ((clamped + 300) / 600) * 100;
    const blackPct = 100 - whitePct;

    document.getElementById('evalBarWhite').style.width = whitePct + '%';
    document.getElementById('evalBarBlack').style.width = blackPct + '%';

    const label = score > 0 ? `+${score.toFixed(1)}` : score.toFixed(1);
    document.getElementById('evalScore').textContent = label;
}

// ──────────────────────────────────────────────────
// Analyze position
// ──────────────────────────────────────────────────

function analyzePosition(lastMove = null) {
    const move = lastMove || lastAIMove;
    fetch('/api/chess/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), last_move: move })
    })
    .then(r => r.json())
    .then(data => {
        updateEvalBar(data.score || 0);

        if (data.material) {
            document.getElementById('whiteCaptured').textContent = data.material.white_captured || '—';
            document.getElementById('blackCaptured').textContent = data.material.black_captured || '—';
        }

        if (data.opening) {
            document.getElementById('openingName').textContent = data.opening;
        }

        if (data.explanation) {
            document.getElementById('moveExplanation').textContent = data.explanation;
        }

        // Also fetch opening details
        fetchOpeningDetails();
    })
    .catch(err => console.error('Analyze error:', err));
}

// ──────────────────────────────────────────────────
// Opening Details
// ──────────────────────────────────────────────────

function fetchOpeningDetails() {
    fetch('/api/chess/opening-info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), moves: moveHistory })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success && data.detected) {
            document.getElementById('openingName').textContent = `${data.eco} — ${data.name}`;
            
            const detailsEl = document.getElementById('openingDetails');
            detailsEl.style.display = 'block';
            
            document.getElementById('openingDescription').textContent = data.description || '';
            document.getElementById('openingStrategy').textContent = data.strategy ? `📋 Strategie: ${data.strategy}` : '';
            
            const diffBadge = document.getElementById('openingDifficulty');
            if (data.difficulty) {
                diffBadge.textContent = data.difficulty;
                diffBadge.className = `difficulty-badge diff-${data.difficulty}`;
                diffBadge.style.display = 'inline-block';
            }
        } else {
            document.getElementById('openingDetails').style.display = 'none';
        }
    })
    .catch(() => {});
}

// ──────────────────────────────────────────────────
// Hint
// ──────────────────────────────────────────────────

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
            removeHighlights();
            $('.square-' + from).addClass('highlight-hint-from');
            $('.square-' + to).addClass('highlight-hint-to');
            document.getElementById('moveExplanation').textContent =
                `💡 Sugestie: mută de pe ${from.toUpperCase()} pe ${to.toUpperCase()}`;
            setTimeout(removeHighlights, 3000);
        }
    })
    .catch(err => {
        if (hintBtn) hintBtn.textContent = '💡 Sugerează-mi o mutare';
        console.error('Hint error:', err);
    });
}

// ──────────────────────────────────────────────────
// AI Move
// ──────────────────────────────────────────────────

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
            moveHistory.push(moveKey);
            analyzePosition(moveKey);
        }
        updateChessStatus(false);
    })
    .catch(err => {
        console.error(err);
        document.getElementById('chessStatus').innerText = 'Eroare la AI.';
    });
}

// ──────────────────────────────────────────────────
// Board Interaction
// ──────────────────────────────────────────────────

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
    moveHistory.push(source + target);
    updateChessStatus(false);
    analyzePosition(null);
    window.setTimeout(makeAIMove, 250);
}

function onSnapEnd() {
    chessBoard.position(chessGame.fen());
}

// ──────────────────────────────────────────────────
// Smart Analysis (Combined AI Analysis)
// ──────────────────────────────────────────────────

function smartAnalyze() {
    const btn = document.querySelector('button[onclick="smartAnalyze()"]');
    if (btn) {
        btn.textContent = '⏳ Se analizează...';
        btn.disabled = true;
    }

    fetch('/api/chess/smart-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            fen: chessGame.fen(),
            last_move: lastAIMove,
            depth: 2
        })
    })
    .then(r => r.json())
    .then(data => {
        if (btn) {
            btn.textContent = '🧠 Analiză Completă AI';
            btn.disabled = false;
        }

        if (!data.success) {
            console.error('Smart analyze error:', data.error);
            return;
        }

        // Update evaluation
        if (data.evaluation) {
            updateEvalBar(data.evaluation.score || 0);
            const content = document.getElementById('smartAnalysisContent');
            content.innerHTML = `
                <div class="analysis-score">
                    <span class="score-value">${data.evaluation.score > 0 ? '+' : ''}${data.evaluation.score}</span>
                    <span class="score-desc">${data.evaluation.description}</span>
                </div>
            `;
        }

        // Update material
        if (data.material) {
            document.getElementById('whiteCaptured').textContent = data.material.white_captured || '—';
            document.getElementById('blackCaptured').textContent = data.material.black_captured || '—';
        }

        // Show predictions
        if (data.predictions && data.predictions.length > 0) {
            const panel = document.getElementById('predictionsPanel');
            const list = document.getElementById('predictionsList');
            panel.style.display = 'block';
            
            list.innerHTML = data.predictions.map((p, i) => `
                <div class="prediction-item" onclick="highlightPrediction('${p.move}')">
                    <span class="pred-rank">${i + 1}.</span>
                    <span class="pred-move">${p.move.toUpperCase()}</span>
                    <span class="pred-score">${p.score}%</span>
                    <span class="pred-explain">${p.explanation}</span>
                </div>
            `).join('');
        }

        // Show tactics
        if (data.tactics && data.tactics.length > 0) {
            const panel = document.getElementById('tacticsPanel');
            const list = document.getElementById('tacticsList');
            panel.style.display = 'block';
            
            list.innerHTML = data.tactics.map(t => `
                <div class="tactic-item">${t}</div>
            `).join('');
        }

        // Update opening
        if (data.opening) {
            const nameText = data.opening.eco !== '?' 
                ? `${data.opening.eco} — ${data.opening.name}`
                : data.opening.name;
            document.getElementById('openingName').textContent = nameText;
            
            if (data.opening.description) {
                const detailsEl = document.getElementById('openingDetails');
                detailsEl.style.display = 'block';
                document.getElementById('openingDescription').textContent = data.opening.description;
            }
        }

        // Show explanation
        if (data.explanation) {
            document.getElementById('moveExplanation').textContent = data.explanation;
        }
    })
    .catch(err => {
        if (btn) {
            btn.textContent = '🧠 Analiză Completă AI';
            btn.disabled = false;
        }
        console.error('Smart analyze error:', err);
    });
}

function highlightPrediction(moveUci) {
    if (moveUci.length >= 4) {
        const from = moveUci.substring(0, 2);
        const to = moveUci.substring(2, 4);
        removeHighlights();
        $('.square-' + from).addClass('highlight-hint-from');
        $('.square-' + to).addClass('highlight-hint-to');
        setTimeout(removeHighlights, 3000);
    }
}

// ──────────────────────────────────────────────────
// Opening Lesson
// ──────────────────────────────────────────────────

function showOpeningLesson() {
    fetch('/api/chess/opening-info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), moves: moveHistory })
    })
    .then(r => r.json())
    .then(data => {
        if (!data.success) return;

        const card = document.getElementById('openingLessonCard');
        const content = document.getElementById('lessonContent');
        const lesson = data.lesson || {};

        card.style.display = 'block';
        document.getElementById('lessonTitle').textContent = lesson.title || '📖 Lecție';

        let html = '';
        
        if (lesson.english_name) {
            html += `<div class="lesson-name">${lesson.english_name}</div>`;
        }
        if (lesson.theory) {
            html += `<div class="lesson-section"><strong>📝 Teorie:</strong> ${lesson.theory}</div>`;
        }
        if (lesson.strategy) {
            html += `<div class="lesson-section"><strong>📋 Strategie:</strong> ${lesson.strategy}</div>`;
        }
        if (lesson.key_ideas && lesson.key_ideas.length > 0) {
            html += `<div class="lesson-section"><strong>💡 Idei cheie:</strong><ul>`;
            lesson.key_ideas.forEach(idea => {
                html += `<li>${idea}</li>`;
            });
            html += `</ul></div>`;
        }
        if (lesson.common_mistakes && lesson.common_mistakes.length > 0) {
            html += `<div class="lesson-section"><strong>⚠️ Greșeli frecvente:</strong><ul>`;
            lesson.common_mistakes.forEach(mistake => {
                html += `<li>${mistake}</li>`;
            });
            html += `</ul></div>`;
        }
        if (lesson.recommended_for) {
            html += `<div class="lesson-section lesson-recommendation">${lesson.recommended_for}</div>`;
        }

        // Show continuations/suggestions
        if (data.suggestions && data.suggestions.length > 0) {
            html += `<div class="lesson-section"><strong>🔀 Continuări teoretice:</strong><ul>`;
            data.suggestions.forEach(s => {
                html += `<li class="continuation-item" onclick="highlightPrediction('${s.next_move}')">
                    <strong>${s.eco}</strong> ${s.name} 
                    <span class="cont-move">→ ${s.next_move.toUpperCase()}</span>
                </li>`;
            });
            html += `</ul></div>`;
        }

        content.innerHTML = html;

        // Scroll to lesson
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    })
    .catch(err => console.error('Opening lesson error:', err));
}

// ──────────────────────────────────────────────────
// Image Upload & Board Recognition
// ──────────────────────────────────────────────────

function initImageUpload() {
    const zone = document.getElementById('imageUploadZone');
    const fileInput = document.getElementById('imageFileInput');
    
    if (!zone || !fileInput) return;

    // Drag & Drop handlers
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            processUploadedImage(files[0]);
        }
    });

    // File input handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            processUploadedImage(e.target.files[0]);
        }
    });
}

function processUploadedImage(file) {
    if (!file.type.startsWith('image/')) {
        alert('Te rog selectează un fișier imagine (PNG, JPG, etc.)');
        return;
    }

    const zone = document.getElementById('imageUploadZone');
    zone.innerHTML = `
        <div class="upload-processing">
            <div class="spinner"></div>
            <p>🧠 Se analizează imaginea...</p>
        </div>
    `;

    const reader = new FileReader();
    reader.onload = function(e) {
        // Show preview
        const base64Data = e.target.result.split(',')[1];
        
        // Send to recognition API
        fetch('/api/chess/recognize-board', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_data: base64Data })
        })
        .then(r => r.json())
        .then(data => {
            if (data.success && data.fen) {
                // Set the recognized position on the board
                const fenParts = data.fen.split(' ');
                chessGame.load(data.fen);
                chessBoard.position(fenParts[0]);
                
                // Update UI
                zone.innerHTML = `
                    <div class="upload-success">
                        <div class="success-icon">✅</div>
                        <p>Poziție recunoscută!</p>
                        <p class="confidence-text">Încredere: ${data.confidence}%</p>
                        <p class="fen-text">${fenParts[0]}</p>
                        <button class="btn secondary-btn btn-sm" onclick="resetUploadZone()">📷 Altă imagine</button>
                    </div>
                `;

                // Auto-analyze the recognized position
                analyzePosition(null);
                updateChessStatus(false);
                moveHistory = [];
                
                document.getElementById('moveExplanation').textContent = 
                    `📷 Poziție recunoscută din imagine (încredere: ${data.confidence}%)`;
            } else {
                zone.innerHTML = `
                    <div class="upload-error">
                        <div class="error-icon">❌</div>
                        <p>Nu am putut recunoaște poziția</p>
                        <p class="error-text">${data.error || 'Eroare necunoscută'}</p>
                        <button class="btn secondary-btn btn-sm" onclick="resetUploadZone()">🔄 Încearcă din nou</button>
                    </div>
                `;
            }
        })
        .catch(err => {
            console.error('Recognition error:', err);
            zone.innerHTML = `
                <div class="upload-error">
                    <div class="error-icon">❌</div>
                    <p>Eroare la recunoaștere</p>
                    <button class="btn secondary-btn btn-sm" onclick="resetUploadZone()">🔄 Încearcă din nou</button>
                </div>
            `;
        });
    };
    reader.readAsDataURL(file);
}

function resetUploadZone() {
    const zone = document.getElementById('imageUploadZone');
    zone.innerHTML = `
        <div class="upload-icon">📷</div>
        <p>Trage o imagine aici sau <label for="imageFileInput" class="upload-link">alege un fișier</label></p>
        <p class="upload-hint">Screenshot de pe Chess.com, Lichess, sau orice tablă de șah</p>
        <input type="file" id="imageFileInput" accept="image/*" style="display:none;">
    `;
    // Re-attach file input handler
    document.getElementById('imageFileInput').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            processUploadedImage(e.target.files[0]);
        }
    });
}

// ──────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────

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
    initImageUpload();
});

function resetChessGame() {
    chessGame.reset();
    chessBoard.start();
    lastAIMove = null;
    moveHistory = [];
    removeHighlights();
    updateChessStatus(false);
    document.getElementById('moveExplanation').textContent = 'Jocul nu a început.';
    document.getElementById('openingName').textContent = 'Poziție de start';
    document.getElementById('openingDetails').style.display = 'none';
    document.getElementById('openingLessonCard').style.display = 'none';
    document.getElementById('predictionsPanel').style.display = 'none';
    document.getElementById('tacticsPanel').style.display = 'none';
    document.getElementById('whiteCaptured').textContent = '—';
    document.getElementById('blackCaptured').textContent = '—';
    document.getElementById('smartAnalysisContent').innerHTML = 
        '<p style="font-size:0.85rem; color:var(--text-muted);">Apasă butonul de mai jos pentru analiză completă.</p>';
    updateEvalBar(0);
    resetUploadZone();
}
