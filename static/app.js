// --- AUDIO SYNTHESIS FOR RETRO CHIP SOUNDS ---
let audioCtx = null;
function playSound(type) {
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'sustonded') {
            audioCtx.resume();
        }
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        
        const now = audioCtx.currentTime;
        if (type === 'coin') {
            osc.type = 'square';
            osc.frequency.setValueAtTime(987.77, now); // B5
            osc.frequency.setValueAtTime(1318.51, now + 0.08); // E6
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
            osc.start(now);
            osc.stop(now + 0.3);
        } else if (type === 'laser') {
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(150, now);
            osc.frequency.exponentialRampToValueAtTime(800, now + 0.25);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
            osc.start(now);
            osc.stop(now + 0.25);
        } else if (type === 'click') {
            osc.type = 'sine';
            osc.frequency.setValueAtTime(300, now);
            gain.gain.setValueAtTime(0.05, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.05);
            osc.start(now);
            osc.stop(now + 0.05);
        }
    } catch(e) {
        console.log("Audio not supported or blocked:", e);
    }
}

function insertCoin() {
    const counter = document.getElementById('creditCount');
    if (counter) {
        let credits = parseInt(counter.innerText) || 0;
        credits += 1;
        counter.innerText = credits;
        playSound('coin');
        addTerminalLog(`[COIN] Credit accepted. Total: ${credits}`);
    }
}

// --- TAB SYSTEM NAVIGATION ---
function switchTab(tabId, element) {
    playSound('click');
    document.querySelectorAll('.tab-pane').forEach(tab => tab.style.display = 'none');
    document.querySelectorAll('.os-nav-links li').forEach(li => li.classList.remove('active'));
    
    document.getElementById(tabId).style.display = 'block';
    if (element) {
        element.classList.add('active');
    } else {
        // Activate sidebar tab item corresponding to tabId
        document.querySelectorAll('.os-nav-links li').forEach(li => {
            const labelText = li.textContent.toUpperCase();
            if (tabId === 'desktop-tab' && labelText.includes('DESKTOP')) li.classList.add('active');
            else if (tabId === 'chess-tab' && labelText.includes('CHESS')) li.classList.add('active');
            else if (tabId === 'snake-tab' && labelText.includes('SNAKE')) li.classList.add('active');
            else if (tabId === 'tetris-tab' && labelText.includes('TETRIS')) li.classList.add('active');
        });
    }

    addTerminalLog(`[OS] Switch tab execution -> ${tabId.toUpperCase()}`);

    // Trigger resizes for canvas/board
    if (tabId === 'chess-tab' && typeof chessBoard !== 'undefined' && chessBoard) {
        chessBoard.resize();
    }
}

function launchApp(tabId) {
    switchTab(tabId);
}

function getActiveTab() {
    const chessTab = document.getElementById('chess-tab');
    const snakeTab = document.getElementById('snake-tab');
    const tetrisTab = document.getElementById('tetris-tab');
    const desktopTab = document.getElementById('desktop-tab');
    if (chessTab && chessTab.style.display !== 'none') return 'chess';
    if (snakeTab && snakeTab.style.display !== 'none') return 'snake';
    if (tetrisTab && tetrisTab.style.display !== 'none') return 'tetris';
    return 'desktop';
}

function triggerArcadeAction(action) {
    const activeTab = getActiveTab();
    playSound('click');
    
    if (action === 'reset') {
        playSound('laser');
        addTerminalLog(`[SYSTEM] Hot reboot command sent for active instance...`);
        if (activeTab === 'chess') {
            if (typeof resetChessGame === 'function') resetChessGame();
        } else if (activeTab === 'snake') {
            if (typeof ws !== 'undefined' && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: "reset"}));
            }
        } else if (activeTab === 'tetris') {
            if (typeof resetGame === 'function') resetGame();
        } else {
            // Desktop reboot
            window.location.reload();
        }
    } else if (action === 'auto') {
        if (activeTab === 'chess') {
            if (typeof setChessGameMode === 'function') {
                const btnPvAI = document.getElementById('btnPlayerVSai');
                if (btnPvAI && btnPvAI.classList.contains('active')) {
                    setChessGameMode('aivsai');
                } else {
                    setChessGameMode('pvai');
                }
            }
        } else if (activeTab === 'snake') {
            if (typeof toggleSnakeMode === 'function') toggleSnakeMode();
        } else if (activeTab === 'tetris') {
            if (typeof toggleTetrisManualMode === 'function') toggleTetrisManualMode();
        }
    } else if (action === 'start') {
        playSound('laser');
        if (activeTab === 'chess') {
            const btnStart = document.getElementById('btnStartAIvsAI');
            if (typeof toggleAIvsAIAutoPlay === 'function' && btnStart && btnStart.style.display !== 'none') {
                toggleAIvsAIAutoPlay();
            } else if (typeof resetChessGame === 'function') {
                resetChessGame();
            }
        } else if (activeTab === 'snake') {
            if (typeof toggleSnakeMode === 'function') toggleSnakeMode();
        } else if (activeTab === 'tetris') {
            if (typeof startTetrisAI === 'function') startTetrisAI();
        }
    } else if (action === 'hint') {
        if (activeTab === 'chess') {
            if (typeof requestHint === 'function') requestHint();
        } else if (activeTab === 'snake') {
            const modes = ['dqn', 'tree', 'bayes', 'astar', 'vs_tree'];
            let nextIdx = 0;
            if (typeof currentSnakeAIMode !== 'undefined') {
                const curIdx = modes.indexOf(currentSnakeAIMode);
                nextIdx = (curIdx + 1) % modes.length;
            }
            if (typeof setSnakeAIMode === 'function') setSnakeAIMode(modes[nextIdx]);
        } else if (activeTab === 'tetris') {
            const modes = ['knn', 'tree', 'mlp', 'genetic'];
            let nextIdx = 0;
            const activeBtn = document.querySelector('.ai-mode-toggle .mode-btn.active');
            if (activeBtn) {
                const modeText = activeBtn.id || '';
                if (modeText.includes('KNN') || modeText.includes('knn')) nextIdx = 1;
                else if (modeText.includes('Tree') || modeText.includes('tree')) nextIdx = 2;
                else if (modeText.includes('MLP') || modeText.includes('mlp')) nextIdx = 3;
                else nextIdx = 0;
            }
            if (typeof setTetrisAIMode === 'function') setTetrisAIMode(modes[nextIdx]);
        }
    }
}

// --- TERMINAL LOGGING HELPER ---
function addTerminalLog(message) {
    const term = document.getElementById('sysTerminal');
    if (!term) return;
    
    // Maintain maximum 30 lines to prevent memory bloating
    const lines = term.getElementsByClassName('term-line');
    if (lines.length > 30) {
        term.removeChild(lines[0]);
    }
    
    const timestamp = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.className = 'term-line';
    div.innerText = `[${timestamp}] > ${message}`;
    
    // Insert before cursor element if exists, else append
    term.appendChild(div);
    term.scrollTop = term.scrollHeight;
}

// --- SYSTEM HARDWARE ACTIVITIES SIMULATION ---
let monitorPoints = Array.from({length: 30}, () => 10 + Math.random() * 20);

function drawMonitorGraph() {
    const canvas = document.getElementById('monitorGraph');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw scan lines grid
    ctx.strokeStyle = 'rgba(255, 0, 127, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 15) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 15) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }

    // Shift left and add new point based on active tabs
    monitorPoints.shift();
    const activeTab = getActiveTab();
    let baseLoad = 12;
    if (activeTab === 'snake') baseLoad = 35 + Math.random() * 15;
    else if (activeTab === 'tetris') baseLoad = 40 + Math.random() * 10;
    else if (activeTab === 'chess') baseLoad = 25 + Math.random() * 20;
    
    monitorPoints.push(baseLoad);
    
    // Draw path
    ctx.strokeStyle = '#ff007f';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#ff007f';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    
    const wStep = canvas.width / (monitorPoints.length - 1);
    monitorPoints.forEach((val, idx) => {
        const x = idx * wStep;
        const y = canvas.height - (val / 100) * canvas.height;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    ctx.shadowBlur = 0; // reset
    
    // Update labels
    const cpuLabel = document.getElementById('cpuLoadText');
    const osSysLoad = document.getElementById('osSysLoad');
    if (cpuLabel) cpuLabel.innerText = `${Math.round(baseLoad)}%`;
    if (osSysLoad) osSysLoad.innerText = `${Math.round(baseLoad)}%`;
}

// Tick monitor graph & background logs at intervals
setInterval(drawMonitorGraph, 500);

const sysLogsTemplate = [
    "DQN memory consolidation optimized.",
    "FastAPI connection keep-alive validated.",
    "KNN table heuristic database matching complete.",
    "Tensorflow execution buffer flushing done.",
    "Evaluating environment state vector... OK.",
    "Pruning minimax branch paths (Alpha-Beta)."
];
setInterval(() => {
    const logIdx = Math.floor(Math.random() * sysLogsTemplate.length);
    addTerminalLog(sysLogsTemplate[logIdx]);
}, 8000);

// --- MODALS AND SPECIAL ACTIONS ---
function showHelpModal() {
    playSound('click');
    document.getElementById('helpModal').style.display = 'block';
}
function hideHelpModal() {
    playSound('click');
    document.getElementById('helpModal').style.display = 'none';
}
function triggerNeonRunner() {
    playSound('laser');
    addTerminalLog("[NEON_RUNNER] Connection refused: Retro grid sync error 404.");
}
function triggerDumpsExplorer() {
    playSound('click');
    document.getElementById('dumpsModal').style.display = 'block';
    addTerminalLog("[DUMPS] Reading local file indexes...");
}
function hideDumpsModal() {
    playSound('click');
    document.getElementById('dumpsModal').style.display = 'none';
}

function powerOffOS() {
    playSound('laser');
    addTerminalLog("[OS] Initiating power-off sequence...");
    const monitor = document.getElementById('crtMonitor');
    if (monitor) {
        monitor.classList.add('power-off');
        setTimeout(() => {
            // Re-enable monitor after 4 seconds to allow user to continue
            alert("ARCADE_OS is shutting down. Press OK to reboot.");
            monitor.classList.remove('power-off');
            window.location.reload();
        }, 1500);
    }
}

// --- Keyboard listeners for joystick animation ---
document.addEventListener('keydown', (e) => {
    const joystick = document.getElementById('arcadeJoystick');
    if (!joystick) return;
    
    if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') {
        joystick.className = 'joystick-shaft tilt-up';
    } else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') {
        joystick.className = 'joystick-shaft tilt-down';
    } else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        joystick.className = 'joystick-shaft tilt-left';
    } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        joystick.className = 'joystick-shaft tilt-right';
    }
});

document.addEventListener('keyup', (e) => {
    const joystick = document.getElementById('arcadeJoystick');
    if (!joystick) return;
    
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 's', 'a', 'd', 'W', 'S', 'A', 'D'].includes(e.key)) {
        joystick.className = 'joystick-shaft';
    }
});
