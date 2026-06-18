// --- Chess Module: Minimax + CNN + Board Recognition + Smart Analysis ---
let chessBoard = null;
const chessGame = new Chess();
let currentAIMode = 'minimax';  // 'minimax' | 'cnn' (Player vs AI)
let chessGameMode = 'pvai';     // 'pvai' | 'aivsai'
let whiteAIMode = 'minimax';    // 'minimax' | 'cnn'
let blackAIMode = 'minimax';    // 'minimax' | 'cnn'
let isAIvsAIPlaying = false;
let aivsaiTimeout = null;
let lastAIMove = null;
let moveHistory = [];  // Track all moves in UCI format

// ──────────────────────────────────────────────────
// AI Mode
// ──────────────────────────────────────────────────

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
function setAIMode(mode) {
    currentAIMode = mode;
    document.getElementById('btnMinimax').classList.toggle('active', mode === 'minimax');
    document.getElementById('btnCNN').classList.toggle('active', mode === 'cnn');
    const desc = document.getElementById('aiModeDesc');
    const visualizer = document.getElementById('cnnVisualizer');
    
    if (mode === 'minimax') {
        desc.textContent = 'Minimax systematically explores future positions with Alpha-Beta Pruning.';
        if (visualizer) $(visualizer).slideUp(300);
        stopCNNAnimation();
    } else {
        desc.textContent = 'CNN evaluates positions through patterns learned from thousands of simulated games.';
        if (visualizer) {
            $(visualizer).slideDown(300);
            document.querySelectorAll('.cnn-layer').forEach(el => el.classList.add('active-layer'));
        }
    }
}

// ──────────────────────────────────────────────────
// Status
// ──────────────────────────────────────────────────

function updateChessStatus(isAIThinking) {
    const statusEl = document.getElementById('chessStatus');
    if (!statusEl) return;

    if (chessGame.in_checkmate()) {
        statusEl.innerText = `Checkmate! Winner is ${chessGame.turn() === 'w' ? 'Black (AI)' : (chessGameMode === 'aivsai' ? 'White (AI)' : 'White (You)')}.`;
        statusEl.style.color = 'var(--danger)';
        statusEl.style.background = 'rgba(239,68,68,0.2)';
    } else if (chessGame.in_draw() || chessGame.in_stalemate() || chessGame.in_threefold_repetition()) {
        statusEl.innerText = 'Draw!';
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
                statusEl.innerText = chessGame.turn() === 'w' ? "White's Turn (AI)" : "Black's Turn (AI)";
            } else {
                statusEl.innerText = 'Your Turn (White)';
            }
            if (chessGame.in_check()) {
                statusEl.innerText += ' — CHECK!';
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
        body: JSON.stringify({ fen: chessGame.fen(), last_move: move, moves: moveHistory })
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
        if (data.success) {
            const detailsEl = document.getElementById('openingDetails');
            detailsEl.style.display = 'block';
            
            if (data.detected) {
                document.getElementById('openingName').textContent = `${data.eco} — ${data.name}`;
                document.getElementById('openingDescription').textContent = data.description || '';
                document.getElementById('openingStrategy').textContent = data.strategy ? `Strategy: ${data.strategy}` : '';
                
                const diffBadge = document.getElementById('openingDifficulty');
                if (data.difficulty) {
                    diffBadge.textContent = data.difficulty;
                    diffBadge.className = `difficulty-badge diff-${data.difficulty}`;
                    diffBadge.style.display = 'inline-block';
                } else {
                    diffBadge.style.display = 'none';
                }
            } else {
                document.getElementById('openingName').textContent = 'Custom / Open Play';
                document.getElementById('openingDescription').textContent = 'No standard opening matches this exact move order yet.';
                document.getElementById('openingStrategy').textContent = 'Strategy: Keep developing pieces, control the center, and prepare castling.';
                
                const diffBadge = document.getElementById('openingDifficulty');
                diffBadge.style.display = 'none';
            }

            // Render continuations/suggestions in the sidebar card
            const suggPanel = document.getElementById('openingSuggestions');
            const suggList = document.getElementById('openingSuggestionsList');
            if (suggPanel && suggList) {
                if (data.suggestions && data.suggestions.length > 0) {
                    suggPanel.style.display = 'block';
                    suggList.innerHTML = data.suggestions.map(s => `
                        <div class="continuation-item" style="font-size:0.75rem; padding:4px 6px; cursor:pointer;" onclick="highlightPrediction('${s.next_move}')">
                            <strong>${s.eco}</strong> ${s.name} <span class="cont-move" style="color:var(--accent);">→ ${s.next_move.toUpperCase()}</span>
                        </div>
                    `).join('');
                } else {
                    suggPanel.style.display = 'none';
                }
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
    if (hintBtn) hintBtn.textContent = 'Calculating...';

    fetch('/api/chess/hint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: chessGame.fen(), depth: Math.min(depth, 2), moves: moveHistory })
    })
    .then(r => r.json())
    .then(data => {
        if (hintBtn) hintBtn.textContent = 'Suggest a Move';
        if (data.hint) {
            const from = data.hint.substring(0, 2);
            const to   = data.hint.substring(2, 4);
            removeHighlights();
            $('.square-' + from).addClass('highlight-hint-from');
            $('.square-' + to).addClass('highlight-hint-to');
            document.getElementById('moveExplanation').textContent =
                `Suggestion: move from ${from.toUpperCase()} to ${to.toUpperCase()}`;
            // Remove hint highlight after 3s
            setTimeout(removeHighlights, 3000);
        }
    })
    .catch(err => {
        if (hintBtn) hintBtn.textContent = 'Suggest a Move';
        console.error('Hint error:', err);
    });
}

// ──────────────────────────────────────────────────
// AI Move
// ──────────────────────────────────────────────────

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
            moveHistory.push(moveKey);
            
            // Draw attention heatmap for the AI's move (Black AI)
            if (data.saliency) {
                drawSaliencyHeatmap(data.saliency, 'b');
            }

            if (typeof addTerminalLog === 'function') {
                const moveSan = move ? move.san : moveKey;
                addTerminalLog(`[CHESS] AI (${currentAIMode.toUpperCase()}) move: ${moveSan}`);
            }
            analyzePosition(moveKey);
        }
        updateChessStatus(false);
        stopCNNAnimation();
    })
    .catch(err => {
        console.error(err);
        document.getElementById('chessStatus').innerText = 'AI Error.';
        stopCNNAnimation();
    });
}

// ──────────────────────────────────────────────────
// Board Interaction
// ──────────────────────────────────────────────────

function removeHighlights() {
    $('#board .square-55d63').removeClass(
        'highlight-square highlight-source highlight-hint-from highlight-hint-to'
    );
    $('#board .square-55d63').css('box-shadow', '');
    $('#board .square-55d63').css('background-color', '');
}

function onDragStart(source, piece) {
    if (chessGame.game_over()) return false;
    if (chessGameMode === 'aivsai') return false; // block drag in AI vs AI mode
    if (piece.search(/^b/) !== -1) return false;
    
    // Clear heatmap highlights when player starts moving
    removeHighlights();
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
        btn.textContent = 'Analyzing...';
        btn.disabled = true;
    }

    fetch('/api/chess/smart-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            fen: chessGame.fen(),
            last_move: lastAIMove,
            depth: 2,
            moves: moveHistory
        })
    })
    .then(r => r.json())
    .then(data => {
        if (btn) {
            btn.textContent = 'Full AI Analysis';
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
            btn.textContent = 'Full AI Analysis';
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
        document.getElementById('lessonTitle').textContent = lesson.title || 'Lesson';

        let html = '';
        
        if (lesson.english_name) {
            html += `<div class="lesson-name">${lesson.english_name}</div>`;
        }
        if (lesson.theory) {
            html += `<div class="lesson-section"><strong>Theory:</strong> ${lesson.theory}</div>`;
        }
        if (lesson.strategy) {
            html += `<div class="lesson-section"><strong>Strategy:</strong> ${lesson.strategy}</div>`;
        }
        if (lesson.key_ideas && lesson.key_ideas.length > 0) {
            html += `<div class="lesson-section"><strong>Key ideas:</strong><ul>`;
            lesson.key_ideas.forEach(idea => {
                html += `<li>${idea}</li>`;
            });
            html += `</ul></div>`;
        }
        if (lesson.common_mistakes && lesson.common_mistakes.length > 0) {
            html += `<div class="lesson-section"><strong>Common mistakes:</strong><ul>`;
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
            html += `<div class="lesson-section"><strong>Theoretical continuations:</strong><ul>`;
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
        alert('Please select an image file (PNG, JPG, etc.)');
        return;
    }

    const zone = document.getElementById('imageUploadZone');
    zone.innerHTML = `
        <div class="upload-processing">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
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
                        <div class="success-icon">Ready</div>
                        <p>Position recognized!</p>
                        <p class="confidence-text">Confidence: ${data.confidence}%</p>
                        <p class="fen-text">${fenParts[0]}</p>
                        <button class="btn secondary-btn btn-sm" onclick="resetUploadZone()">Another image</button>
                    </div>
                `;

                // Auto-analyze the recognized position
                analyzePosition(null);
                updateChessStatus(false);
                moveHistory = [];
                
                document.getElementById('moveExplanation').textContent = 
                    `Position recognized from image (confidence: ${data.confidence}%)`;
            } else {
                zone.innerHTML = `
                    <div class="upload-error">
                        <div class="error-icon">X</div>
                        <p>Could not recognize position</p>
                        <p class="error-text">${data.error || 'Unknown error'}</p>
                        <button class="btn secondary-btn btn-sm" onclick="resetUploadZone()">Try again</button>
                    </div>
                `;
            }
        })
        .catch(err => {
            console.error('Recognition error:', err);
            zone.innerHTML = `
                <div class="upload-error">
                    <div class="error-icon">X</div>
                    <p>Recognition error</p>
                    <button class="btn secondary-btn btn-sm" onclick="resetUploadZone()">Try again</button>
                </div>
            `;
        });
    };
    reader.readAsDataURL(file);
}

function resetUploadZone() {
    const zone = document.getElementById('imageUploadZone');
    zone.innerHTML = `
        <div class="upload-icon">[IMAGE]</div>
        <p>Drag an image here or <label for="imageFileInput" class="upload-link">choose a file</label></p>
        <p class="upload-hint">Screenshot from Chess.com, Lichess, or any chess board</p>
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
    if (isAIvsAIPlaying) {
        toggleAIvsAIAutoPlay();
    }
    chessGame.reset();
    chessBoard.start();
    lastAIMove = null;
    moveHistory = [];
    removeHighlights();
    updateChessStatus(false);
    document.getElementById('moveExplanation').textContent = 'Game has not started.';
    document.getElementById('openingName').textContent = 'Starting Position';
    document.getElementById('openingDetails').style.display = 'none';
    document.getElementById('openingLessonCard').style.display = 'none';
    document.getElementById('predictionsPanel').style.display = 'none';
    document.getElementById('tacticsPanel').style.display = 'none';
    document.getElementById('whiteCaptured').textContent = '—';
    document.getElementById('blackCaptured').textContent = '—';
    document.getElementById('smartAnalysisContent').innerHTML = 
        '<p style="font-size:0.85rem; color:var(--text-muted);">Press the button below for full analysis.</p>';
    updateEvalBar(0);
    resetUploadZone();
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
    const stepBtn = document.getElementById('btnStepAIvsAI');
    const visualizer = document.getElementById('cnnVisualizer');
    
    if (mode === 'pvai') {
        if (playerAiCard) playerAiCard.style.display = 'block';
        if (aiConfigCard) aiConfigCard.style.display = 'none';
        if (hintBtn) hintBtn.style.display = 'block';
        if (startBtn) startBtn.style.display = 'none';
        if (stepBtn) stepBtn.style.display = 'none';
        
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
        if (stepBtn) stepBtn.style.display = 'block';
        
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
        btn.innerHTML = 'Stop AI vs AI';
        btn.style.backgroundColor = 'var(--danger)';
    } else {
        btn.innerHTML = 'Start AI vs AI';
        btn.style.backgroundColor = 'var(--accent)';
    }
}

function stepAIvsAI(isManualStep = false) {
    if (chessGame.game_over()) {
        updateChessStatus(false);
        return;
    }

    if (!isAIvsAIPlaying && !isManualStep) {
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
        if (!isAIvsAIPlaying && !isManualStep) {
            stopCNNAnimation();
            return;
        }

        const moveKey = data.move;
        if (moveKey) {
            chessGame.move(moveKey, { sloppy: true });
            chessBoard.position(chessGame.fen());
            lastAIMove = moveKey;
            
            // Draw attention heatmap with respective colors
            if (data.saliency) {
                drawSaliencyHeatmap(data.saliency, turn);
            }

            if (typeof addTerminalLog === 'function') {
                const turnName = turn === 'w' ? 'White' : 'Black';
                addTerminalLog(`[CHESS] AI (${model.toUpperCase()}) ${turnName} move: ${moveKey}`);
            }
            analyzePosition(moveKey);
        }
        
        updateChessStatus(false);
        stopCNNAnimation();

        if (isManualStep) {
            isAIvsAIPlaying = false;
            updateAIvsAIButtonState();
        } else {
            if (!chessGame.game_over() && isAIvsAIPlaying) {
                aivsaiTimeout = window.setTimeout(() => stepAIvsAI(false), 3000);
            } else {
                isAIvsAIPlaying = false;
                updateAIvsAIButtonState();
            }
        }
    })
    .catch(err => {
        console.error("AI vs AI move error:", err);
        isAIvsAIPlaying = false;
        updateAIvsAIButtonState();
        stopCNNAnimation();
    });
}

function drawSaliencyHeatmap(saliency, turn) {
    // Clear old heatmap styles
    $('#board .square-55d63').css('box-shadow', '');
    $('#board .square-55d63').css('background-color', '');
    
    if (!saliency || saliency.length !== 64) return;
    
    // White: Cyan (rgba(0, 243, 255, alpha))
    // Black: Magenta (rgba(255, 0, 127, alpha))
    const color = (turn === 'w') ? '0, 243, 255' : '255, 0, 127';
    
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];
    
    for (let sq = 0; sq < 64; sq++) {
        const val = saliency[sq];
        if (val > 0.05) {
            const fileIdx = sq % 8;
            const rankIdx = Math.floor(sq / 8);
            const squareName = files[fileIdx] + ranks[rankIdx];
            
            const squareEl = $('#board .square-' + squareName);
            if (squareEl.length > 0) {
                const alpha = (val * 0.45).toFixed(2);
                squareEl.css('background-color', `rgba(${color}, ${alpha})`);
                squareEl.css('box-shadow', `inset 0 0 10px rgba(${color}, ${(val * 0.75).toFixed(2)})`);
            }
        }
    }
}
