import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from game_logic import SnakeGameHeadless, MultiAgentSnakeGame, get_astar_path, Direction, Point
from agent import Agent
from pydantic import BaseModel
import chess
from chess_ai import get_best_move, evaluate_board, detect_opening, explain_move, get_material_balance
from chess_cnn import get_best_move_cnn, cnn_evaluate
from next_move_predictor import predict_next_move
from opening_coach import get_opening_coach
from board_recognition_cnn import recognize_board, recognize_board_with_confidence
import json
import base64
import io
import tempfile
from collections import deque
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

app = FastAPI()

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

class ChessMoveRequest(BaseModel):
    fen: str
    depth: int = 3
    moves: list[str] = []

class ChessAnalyzeRequest(BaseModel):
    fen: str
    last_move: str | None = None
    moves: list[str] = []

class OpeningRequest(BaseModel):
    fen: str
    moves: list[str] | None = None

class OpeningSearchRequest(BaseModel):
    query: str | None = None
    difficulty: str | None = None
    style: str | None = None

class ImageRecognizeRequest(BaseModel):
    image_data: str  # base64 encoded image

class SmartAnalyzeRequest(BaseModel):
    fen: str
    last_move: str | None = None
    depth: int = 2

@app.post("/api/chess/move")
async def post_chess_move(req: ChessMoveRequest):
    board = chess.Board(req.fen)
    best_move = await asyncio.to_thread(get_best_move, board, req.depth)
    return {"move": best_move}

@app.post("/api/chess/cnn-move")
async def post_chess_cnn_move(req: ChessMoveRequest):
    board = chess.Board(req.fen)
    best_move = await asyncio.to_thread(get_best_move_cnn, board, min(req.depth, 2))
    return {"move": best_move}

@app.post("/api/chess/analyze")
async def post_chess_analyze(req: ChessAnalyzeRequest):
    board = chess.Board(req.fen)
    score = await asyncio.to_thread(cnn_evaluate, board)
    material = get_material_balance(board)
    opening = detect_opening(board)
    explanation = ""
    if req.last_move:
        try:
            move = chess.Move.from_uci(req.last_move)
            board_before = chess.Board(req.fen)
            # Reconstruct board before last move
            board_before.push(move)
            board_before2 = chess.Board(req.fen)
            explanation = explain_move(board_before2, move)
        except Exception:
            explanation = ""
    return {
        "score": round(score * 100, 2),
        "material": material,
        "opening": opening,
        "explanation": explanation,
    }

@app.post("/api/chess/hint")
async def post_chess_hint(req: ChessMoveRequest):
    if req.moves:
        board = chess.Board()
        for move_uci in req.moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
            except ValueError:
                pass
    else:
        board = chess.Board(req.fen)
    # Use the new predict_next_move which has Opening Book and Hybrid CNN logic!
    results = await asyncio.to_thread(predict_next_move, board, 1)
    best_move = results[0]["move"] if results else None
    return {"hint": best_move}


# ──────────────────────────────────────────────────
# NEW: Board Recognition Endpoint
# ──────────────────────────────────────────────────

@app.post("/api/chess/recognize-board")
async def post_recognize_board(req: ImageRecognizeRequest):
    """Recognize chess position from an uploaded board image."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_data)
        from PIL import Image
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Run recognition
        result = await asyncio.to_thread(recognize_board_with_confidence, img)
        
        # Validate FEN: ensure both kings are present
        board = chess.Board(result["fen"])
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            return {
                "success": False,
                "error": "The model did not detect both kings on the board. Make sure the image contains the entire chessboard and is cropped correctly to its outline.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            }
        
        return {
            "success": True,
            "fen": result["fen"],
            "confidence": round(result["avg_confidence"] * 100, 1),
            "min_confidence": round(result["min_confidence"] * 100, 1),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        }


@app.post("/api/chess/recognize-board-upload")
async def post_recognize_board_upload(file: UploadFile = File(...)):
    """Recognize chess position from a file upload."""
    try:
        contents = await file.read()
        from PIL import Image
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result = await asyncio.to_thread(recognize_board_with_confidence, img)
        
        # Validate FEN: ensure both kings are present
        board = chess.Board(result["fen"])
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            return {
                "success": False,
                "error": "The model did not detect both kings on the board. Make sure the image contains the entire chessboard and is cropped correctly to its outline.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            }
        
        return {
            "success": True,
            "fen": result["fen"],
            "confidence": round(result["avg_confidence"] * 100, 1),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        }


# ──────────────────────────────────────────────────
# NEW: Next-Move Prediction Endpoint
# ──────────────────────────────────────────────────

@app.post("/api/chess/next-move-prediction")
async def post_next_move_prediction(req: ChessMoveRequest):
    """Predict next best moves using the trained policy network."""
    try:
        if req.moves:
            board = chess.Board()
            for move_uci in req.moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                except ValueError:
                    pass
        else:
            board = chess.Board(req.fen)
        predictions = await asyncio.to_thread(predict_next_move, board, 5)
        return {
            "success": True,
            "predictions": predictions,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "predictions": [],
        }


# ──────────────────────────────────────────────────
# NEW: Opening Coach Endpoints
# ──────────────────────────────────────────────────

@app.post("/api/chess/opening-info")
async def post_opening_info(req: OpeningRequest):
    """Get detailed opening information for the current position."""
    try:
        board = chess.Board(req.fen)
        
        # Replay moves if provided
        if req.moves:
            board = chess.Board()
            for move_uci in req.moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                except (ValueError, chess.InvalidMoveError):
                    break
        
        coach = get_opening_coach()
        info = await asyncio.to_thread(coach.get_opening_info, board)
        return {"success": True, **info}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/chess/opening-search")
async def post_opening_search(req: OpeningSearchRequest):
    """Search openings by name, difficulty, or style."""
    try:
        coach = get_opening_coach()
        
        if req.style:
            results = coach.suggest_opening_for_style(req.style)
        elif req.query:
            results = coach.search_openings(req.query)
        elif req.difficulty:
            results = coach.get_all_openings(req.difficulty)
        else:
            results = coach.get_all_openings()
        
        return {"success": True, "openings": results}
    except Exception as e:
        return {"success": False, "error": str(e), "openings": []}


@app.get("/api/chess/openings")
async def get_all_openings():
    """Get all openings in the database."""
    try:
        coach = get_opening_coach()
        return {"success": True, "openings": coach.get_all_openings()}
    except Exception as e:
        return {"success": False, "error": str(e), "openings": []}


# ──────────────────────────────────────────────────
# NEW: Smart Analysis (Combined endpoint)
# ──────────────────────────────────────────────────

@app.post("/api/chess/smart-analyze")
async def post_smart_analyze(req: SmartAnalyzeRequest):
    """
    Comprehensive analysis combining:
    - CNN position evaluation
    - Next-move predictions
    - Opening information
    - Material balance
    - Move explanation
    """
    try:
        board = chess.Board(req.fen)
        
        # CNN evaluation
        score = await asyncio.to_thread(cnn_evaluate, board)
        
        # Material balance
        material = get_material_balance(board)
        
        # Next-move predictions
        try:
            predictions = await asyncio.to_thread(predict_next_move, board, 3)
        except Exception:
            predictions = []
        
        # Opening info
        coach = get_opening_coach()
        opening_info = coach.get_opening_info(board)
        
        # Move explanation
        explanation = ""
        if req.last_move:
            try:
                move = chess.Move.from_uci(req.last_move)
                board_before = chess.Board(req.fen)
                explanation = explain_move(board_before, move)
            except Exception:
                pass
        
        # Tactical analysis
        tactics = _detect_tactics(board)
        
        return {
            "success": True,
            "evaluation": {
                "score": round(score * 100, 2),
                "description": _score_description(score),
            },
            "material": material,
            "predictions": predictions,
            "opening": {
                "name": opening_info.get("name", "Necunoscută"),
                "eco": opening_info.get("eco", "?"),
                "description": opening_info.get("description", ""),
                "suggestions": opening_info.get("suggestions", [])[:3],
            },
            "explanation": explanation,
            "tactics": tactics,
            "lesson": opening_info.get("lesson", {}),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _score_description(score: float) -> str:
    """Convert a score to a human-readable description."""
    s = score * 100
    if s > 200:
        return "⬜ Albul câștigă decisiv"
    elif s > 100:
        return "⬜ Avantaj mare pentru Alb"
    elif s > 30:
        return "⬜ Slight advantage for White"
    elif s > -30:
        return "= Poziție egală"
    elif s > -100:
        return "⬛ Slight advantage for Black"
    elif s > -200:
        return "⬛ Avantaj mare pentru Negru"
    else:
        return "⬛ Negrul câștigă decisiv"


def _detect_tactics(board: chess.Board) -> list[str]:
    """Detect basic tactical patterns in the position."""
    tactics = []
    
    if board.is_check():
        tactics.append("♚ King este în ȘAH!")
    
    # Check for possible forks, pins, etc.
    for move in board.legal_moves:
        board.push(move)
        
        if board.is_checkmate():
            board.pop()
            piece = board.piece_at(move.from_square)
            sq = chess.square_name(move.to_square).upper()
            tactics.append(f"💀 MAT posibil: {sq}!")
            continue
        
        if board.is_check():
            # Check if this also attacks another piece (fork/discovered attack)
            board.pop()
            if board.is_capture(move):
                tactics.append(f"⚡ Șah cu captură posibil!")
                continue
        
        board.pop()
    
    # Check for hanging pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color != board.turn:
            if board.is_attacked_by(board.turn, sq):
                if not board.is_attacked_by(not board.turn, sq):
                    piece_name = {
                        chess.PAWN: "Pawn", chess.KNIGHT: "Knight",
                        chess.BISHOP: "Bishop", chess.ROOK: "Rook",
                        chess.QUEEN: "Queen", chess.KING: "King"
                    }.get(piece.piece_type, "Piece")
                    sq_name = chess.square_name(sq).upper()
                    tactics.append(f"🎯 Undefended {piece_name} on {sq_name}!")
    
    if not tactics:
        tactics.append("🔍 No immediate tactics detected")
    
    return tactics[:5]


# ──────────────────────────────────────────────────
# WebSocket (Snake game - existing)
# ──────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

server_state = {
    "is_manual": False,
    "manual_key": None,
    "ai_mode": "dqn",  # "dqn" | "tree" | "bayes"
    "reset_requested": False
}

# --- Snake AI Models & Data Buffers ---
MAX_BUFFER = 5_000
training_data_x = deque(maxlen=MAX_BUFFER)
training_data_y = deque(maxlen=MAX_BUFFER)

collision_data_x = deque(maxlen=MAX_BUFFER)
collision_data_y = deque(maxlen=MAX_BUFFER)

scores_history = []
snake_scores_by_mode = {
    "dqn": [],
    "tree": [],
    "bayes": []
}

tree_model = None
bayes_model = None
score_predictor_model = None
predicted_next_score = 0.0

class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.fc(x)

def explain_tree_decision(model, x):
    if model is None:
        return "Tree is training... Wait for the first game to end."
    
    tree_ = model.tree_
    node = 0
    conditions = []
    
    action_names = ["FORWARD", "RIGHT", "LEFT"]
    feature_names = [
        "Pericol Forward",
        "Pericol Right",
        "Pericol Left",
        "Direcție Left",
        "Direcție Right",
        "Direcție Sus",
        "Direcție Jos",
        "Mâncare Left",
        "Mâncare Right",
        "Mâncare Sus",
        "Mâncare Jos"
    ]
    
    while tree_.children_left[node] != -1:
        feat = tree_.feature[node]
        val = x[feat]
        thresh = tree_.threshold[node]
        
        if val <= thresh:
            conditions.append(f"FĂRĂ {feature_names[feat]}")
            node = tree_.children_left[node]
        else:
            conditions.append(feature_names[feat])
            node = tree_.children_right[node]
            
    val_at_leaf = tree_.value[node][0]
    action_idx = int(np.argmax(val_at_leaf))
    action_name = action_names[action_idx]
    
    explanation = "Dacă " + " ȘI ".join(conditions) + " ➜ " + action_name
    return explanation

def pretrain_models_at_startup():
    global tree_model, bayes_model, score_predictor_model, predicted_next_score
    print("Pre-training Snake models at startup...")
    try:
        agent = Agent()
        game = SnakeGameHeadless(w=400, h=400)
        
        # Simulate play using the loaded model weights (so we get real data)
        # Disable randomness for high-quality data
        agent.epsilon = 0
        
        # Simulate 2,000 steps to fill buffer
        for _ in range(2500):
            state_old = agent.get_state(game)
            state0 = torch.tensor(state_old, dtype=torch.float)
            with torch.no_grad():
                prediction = agent.model(state0)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1
            
            reward, done, score = game.play_step(final_move)
            
            action_idx = move
            training_data_x.append(state_old)
            training_data_y.append(action_idx)
            
            collision_data_x.append(np.append(state_old, action_idx))
            collision_data_y.append(1 if done else 0)
            
            if done:
                scores_history.append(score)
                game.reset()
                
        # Fit Decision Tree
        if len(training_data_x) >= 100:
            X_t = np.array(list(training_data_x))
            y_t = np.array(list(training_data_y))
            tree_model = DecisionTreeClassifier(max_depth=6, random_state=42)
            tree_model.fit(X_t, y_t)
            
        # Fit Naive Bayes
        if len(collision_data_x) >= 100:
            X_c = np.array(list(collision_data_x))
            y_c = np.array(list(collision_data_y))
            bayes_model = GaussianNB()
            bayes_model.fit(X_c, y_c)
            
        # Fit Score Predictor (RNN)
        if len(scores_history) >= 10:
            X_s = []
            y_s = []
            for i in range(len(scores_history) - 5):
                X_s.append(scores_history[i:i+5])
                y_s.append(scores_history[i+5])
            X_s = torch.tensor(X_s, dtype=torch.float32)
            y_s = torch.tensor(y_s, dtype=torch.float32).unsqueeze(1)
            
            score_predictor_model = ScorePredictor()
            opt = optim.Adam(score_predictor_model.parameters(), lr=0.01)
            crit = nn.MSELoss()
            score_predictor_model.train()
            for _ in range(15):
                opt.zero_grad()
                loss = crit(score_predictor_model(X_s), y_s)
                loss.backward()
                opt.step()
                
            score_predictor_model.eval()
            with torch.no_grad():
                last_5 = torch.tensor([scores_history[-5:]], dtype=torch.float32)
                predicted_next_score = round(max(0.0, score_predictor_model(last_5).item()), 1)
        print("Pre-training complete! Snake models are ready.")
    except Exception as e:
        print(f"Failed to pre-train Snake models at startup: {e}")

class SnakeGameProxy:
    def __init__(self, multi_game, is_dqn=True):
        self.multi_game = multi_game
        self.is_dqn = is_dqn
        
    @property
    def snake(self):
        return self.multi_game.snake_dqn if self.is_dqn else self.multi_game.snake_tree
        
    @property
    def direction(self):
        return self.multi_game.direction_dqn if self.is_dqn else self.multi_game.direction_tree
        
    @property
    def food(self):
        return self.multi_game.food
        
    @property
    def head(self):
        return self.multi_game.head_dqn if self.is_dqn else self.multi_game.head_tree
        
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        return self.multi_game.is_collision(
            pt, 
            self.multi_game.snake_dqn if self.is_dqn else self.multi_game.snake_tree,
            self.multi_game.snake_tree if self.is_dqn else self.multi_game.snake_dqn,
            is_dqn=self.is_dqn
        )


async def training_loop():
    global server_state, tree_model, bayes_model, score_predictor_model, predicted_next_score
    agent = Agent()
    game = SnakeGameHeadless(w=400, h=400)
    multi_game = MultiAgentSnakeGame(w=400, h=400)
    
    # Initialize plot scores from startup pre-training to populate the chart immediately
    plot_scores = list(scores_history)
    total_score = sum(plot_scores)
    plot_mean_scores = []
    for i in range(1, len(plot_scores) + 1):
        plot_mean_scores.append(sum(plot_scores[:i]) / i)
        
    agent.n_games = len(plot_scores)
    record = max(plot_scores) if plot_scores else 0
    prev_ai_mode = server_state.get("ai_mode", "dqn")
    
    while True:
        if not manager.active_connections:
            await asyncio.sleep(0.5)
            continue

        ai_mode = server_state.get("ai_mode", "dqn")
        if ai_mode != prev_ai_mode:
            if ai_mode == "vs_tree":
                multi_game.reset()
            else:
                game.reset()
            prev_ai_mode = ai_mode
            
        if server_state.get("reset_requested", False):
            game.reset()
            multi_game.reset()
            server_state["reset_requested"] = False
        
        # Determine current state representation
        if ai_mode == "vs_tree":
            proxy_dqn = SnakeGameProxy(multi_game, is_dqn=True)
            state_old = agent.get_state(proxy_dqn)
        else:
            state_old = agent.get_state(game)
        
        # Calculate Q-values for current state (always calculated for visual comparison)
        state_tensor = torch.tensor(state_old, dtype=torch.float)
        with torch.no_grad():
            pred = agent.model(state_tensor)
        dqn_q_values = [float(q) for q in pred.tolist()]
        greedy_action_idx = int(torch.argmax(pred).item())

        # Determine Decision Tree prediction
        tree_action_idx = greedy_action_idx
        if tree_model is not None:
            try:
                tree_action_idx = int(tree_model.predict(state_old.reshape(1, -1))[0])
            except:
                pass

        # Calculate Naive Bayes risks and decision
        current_bayes_risks = {"straight": 0.0, "right": 0.0, "left": 0.0}
        bayes_action_idx = greedy_action_idx
        if bayes_model is not None:
            try:
                risks = []
                for a in range(3):
                    sample = np.append(state_old, a).reshape(1, -1)
                    proba = bayes_model.predict_proba(sample)[0][1]
                    risks.append(float(proba))
                current_bayes_risks = {"straight": risks[0], "right": risks[1], "left": risks[2]}
                
                # Bayes decision logic: if DQN chosen action is risky (> 0.4), override with safest
                if risks[bayes_action_idx] > 0.4:
                    bayes_action_idx = int(np.argmin(risks))
            except:
                pass

        astar_path_data = []
        action_chosen_idx = 0

        if server_state["is_manual"]:
            # Convert manual key to relative action
            action = [1, 0, 0] # default straight
            if server_state["manual_key"]:
                k = server_state["manual_key"]
                d = game.direction
                if k == "ArrowUp" and d != Direction.DOWN:
                    if d == Direction.LEFT: action = [0, 1, 0] # turn right
                    elif d == Direction.RIGHT: action = [0, 0, 1] # turn left
                elif k == "ArrowDown" and d != Direction.UP:
                    if d == Direction.RIGHT: action = [0, 1, 0]
                    elif d == Direction.LEFT: action = [0, 0, 1]
                elif k == "ArrowLeft" and d != Direction.RIGHT:
                    if d == Direction.UP: action = [0, 0, 1]
                    elif d == Direction.DOWN: action = [0, 1, 0]
                elif k == "ArrowRight" and d != Direction.LEFT:
                    if d == Direction.UP: action = [0, 1, 0]
                    elif d == Direction.DOWN: action = [0, 0, 1]
                server_state["manual_key"] = None # consume key
            
            reward, done, score = game.play_step(action)
            action_chosen_idx = int(np.argmax(action))
        else:
            # AI Control
            if ai_mode == "vs_tree":
                # DQN move prediction
                final_move_dqn = [1, 0, 0]
                if not multi_game.dead_dqn:
                    proxy_dqn = SnakeGameProxy(multi_game, is_dqn=True)
                    st_dqn = agent.get_state(proxy_dqn)
                    st_tensor = torch.tensor(st_dqn, dtype=torch.float)
                    with torch.no_grad():
                        pred_dqn = agent.model(st_tensor)
                    move_dqn = torch.argmax(pred_dqn).item()
                    final_move_dqn = [0, 0, 0]
                    final_move_dqn[move_dqn] = 1
                
                # Tree move prediction
                final_move_tree = [1, 0, 0]
                if not multi_game.dead_tree:
                    proxy_tree = SnakeGameProxy(multi_game, is_dqn=False)
                    st_tree = agent.get_state(proxy_tree)
                    if tree_model is not None:
                        try:
                            move_tree = int(tree_model.predict(st_tree.reshape(1, -1))[0])
                        except:
                            move_tree = 0
                    else:
                        move_tree = 0
                    final_move_tree = [0, 0, 0]
                    final_move_tree[move_tree] = 1
                    
                done, score_dqn, score_tree = multi_game.play_step(final_move_dqn, final_move_tree)
                score = max(score_dqn, score_tree) # overall score for statistics
            elif ai_mode == "astar":
                # Compute A* path
                path = get_astar_path(game.snake, game.head, game.food)
                if path and len(path) > 0:
                    astar_path_data = [{"x": pt.x, "y": pt.y} for pt in path]
                    next_pt = path[0]
                    if next_pt.x > game.head.x:
                        next_dir = Direction.RIGHT
                    elif next_pt.x < game.head.x:
                        next_dir = Direction.LEFT
                    elif next_pt.y > game.head.y:
                        next_dir = Direction.DOWN
                    else:
                        next_dir = Direction.UP
                    
                    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                    idx = clock_wise.index(game.direction)
                    if next_dir == clock_wise[idx]:
                        action_idx = 0
                    elif next_dir == clock_wise[(idx + 1) % 4]:
                        action_idx = 1
                    else:
                        action_idx = 2
                    final_move = [0, 0, 0]
                    final_move[action_idx] = 1
                else:
                    # Fallback to local collision-free heuristics
                    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                    idx = clock_wise.index(game.direction)
                    best_action_idx = 0
                    min_dist = float('inf')
                    for a_idx in range(3):
                        if a_idx == 0:
                            new_dir = clock_wise[idx]
                        elif a_idx == 1:
                            new_dir = clock_wise[(idx + 1) % 4]
                        else:
                            new_dir = clock_wise[(idx - 1) % 4]
                        x = game.head.x
                        y = game.head.y
                        if new_dir == Direction.RIGHT: x += 20
                        elif new_dir == Direction.LEFT: x -= 20
                        elif new_dir == Direction.DOWN: y += 20
                        elif new_dir == Direction.UP: y -= 20
                        next_pt = Point(x, y)
                        if not game.is_collision(next_pt):
                            dist = abs(next_pt.x - game.food.x) + abs(next_pt.y - game.food.y)
                            if dist < min_dist:
                                min_dist = dist
                                best_action_idx = a_idx
                    final_move = [0, 0, 0]
                    final_move[best_action_idx] = 1
                
                reward, done, score = game.play_step(final_move)
                action_chosen_idx = int(np.argmax(final_move))
            else:
                # Other single AI modes
                if ai_mode == "tree" and tree_model is not None:
                    action_idx = tree_action_idx
                    final_move = [0, 0, 0]
                    final_move[action_idx] = 1
                elif ai_mode == "bayes":
                    action_idx = bayes_action_idx
                    final_move = [0, 0, 0]
                    final_move[action_idx] = 1
                else:
                    final_move = agent.get_action(state_old)
                
                reward, done, score = game.play_step(final_move)
                action_chosen_idx = int(np.argmax(final_move))
                
                # Data collection for tree and bayes (collect whenever AI plays)
                training_data_x.append(state_old)
                training_data_y.append(greedy_action_idx)
                collision_data_x.append(np.append(state_old, action_chosen_idx))
                collision_data_y.append(1 if done else 0)
                
                # DQN updates (only if in DQN mode)
                if ai_mode == "dqn":
                    state_new = agent.get_state(game)
                    agent.train_short_memory(state_old, final_move, reward, state_new, done)
                    agent.remember(state_old, final_move, reward, state_new, done)

        # Calculate average score tor mode for live bar comparison
        avg_scores = {
            "dqn": round(sum(snake_scores_by_mode["dqn"]) / len(snake_scores_by_mode["dqn"]), 1) if snake_scores_by_mode["dqn"] else 0.0,
            "tree": round(sum(snake_scores_by_mode["tree"]) / len(snake_scores_by_mode["tree"]), 1) if snake_scores_by_mode["tree"] else 0.0,
            "bayes": round(sum(snake_scores_by_mode["bayes"]) / len(snake_scores_by_mode["bayes"]), 1) if snake_scores_by_mode["bayes"] else 0.0
        }

        # Broadcast state
        if ai_mode == "vs_tree" and not server_state["is_manual"]:
            state_msg = {
                "multi_agent": True,
                "dqn_snake": [{"x": pt.x, "y": pt.y} for pt in multi_game.snake_dqn] if not multi_game.dead_dqn else [],
                "tree_snake": [{"x": pt.x, "y": pt.y} for pt in multi_game.snake_tree] if not multi_game.dead_tree else [],
                "food": {"x": multi_game.food.x, "y": multi_game.food.y},
                "dqn_score": score_dqn,
                "tree_score": score_tree,
                "dead_dqn": multi_game.dead_dqn,
                "dead_tree": multi_game.dead_tree,
                "ai_mode": "vs_tree",
                "direction_dqn": multi_game.direction_dqn.name.lower(),
                "direction_tree": multi_game.direction_tree.name.lower(),
                "games": agent.n_games,
                "record": record,
                "scores": plot_scores,
                "mean_scores": plot_mean_scores,
                "avg_scores": avg_scores
            }
        else:
            state_msg = {
                "snake": [{"x": pt.x, "y": pt.y} for pt in game.snake],
                "food": {"x": game.food.x, "y": game.food.y},
                "score": score,
                "games": agent.n_games,
                "record": record,
                "scores": plot_scores,
                "mean_scores": plot_mean_scores,
                "ai_mode": server_state["ai_mode"],
                "tree_rule": explain_tree_decision(tree_model, state_old),
                "bayes_risks": current_bayes_risks,
                "predicted_next_score": predicted_next_score,
                "direction": game.direction.name.lower(),
                "dqn_q_values": dqn_q_values,
                "tree_action": tree_action_idx,
                "bayes_action": bayes_action_idx,
                "chosen_action": action_chosen_idx,
                "avg_scores": avg_scores
            }
            if ai_mode == "astar":
                state_msg["astar_path"] = astar_path_data
                
        await manager.broadcast(state_msg)
        if server_state.get("is_manual", False):
            await asyncio.sleep(0.12) # ~8 FPS for human play
        else:
            await asyncio.sleep(0.04) # 25 FPS for AI

        if done:
            if ai_mode == "vs_tree" and not server_state["is_manual"]:
                snake_scores_by_mode["dqn"].append(score_dqn)
                if len(snake_scores_by_mode["dqn"]) > 50:
                    snake_scores_by_mode["dqn"].pop(0)
                snake_scores_by_mode["tree"].append(score_tree)
                if len(snake_scores_by_mode["tree"]) > 50:
                    snake_scores_by_mode["tree"].pop(0)
                    
                multi_game.reset()
                agent.n_games += 1
                
                if score > record:
                    record = score
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / max(1, agent.n_games)
                plot_mean_scores.append(mean_score)
                scores_history.append(score)
            else:
                game.reset()
                current_mode = "manual" if server_state["is_manual"] else server_state.get("ai_mode", "dqn")
                if current_mode in snake_scores_by_mode:
                    snake_scores_by_mode[current_mode].append(score)
                    if len(snake_scores_by_mode[current_mode]) > 50:
                        snake_scores_by_mode[current_mode].pop(0)

                if not server_state["is_manual"]:
                    agent.n_games += 1
                    if server_state["ai_mode"] == "dqn":
                        agent.train_long_memory()

                if score > record:
                    record = score
                    if not server_state["is_manual"] and server_state["ai_mode"] == "dqn":
                        agent.model.save()
                
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / max(1, agent.n_games)
                plot_mean_scores.append(mean_score)
                scores_history.append(score)
            if len(scores_history) > 100:
                scores_history.pop(0)

            # --- Retrain Models ---
            # Retrain only every 5 games to avoid blocking the event loop frequently
            if agent.n_games % 5 == 0:
                # 1. Train Decision Tree
                if len(training_data_x) >= 100:
                    try:
                        X_t = np.array(list(training_data_x))
                        y_t = np.array(list(training_data_y))
                        tree_model = DecisionTreeClassifier(max_depth=6, random_state=42)
                        tree_model.fit(X_t, y_t)
                    except Exception as e:
                        print(f"Error training Decision Tree: {e}")

                # 2. Train Naive Bayes
                if len(collision_data_x) >= 100:
                    try:
                        X_c = np.array(list(collision_data_x))
                        y_c = np.array(list(collision_data_y))
                        bayes_model = GaussianNB()
                        bayes_model.fit(X_c, y_c)
                    except Exception as e:
                        print(f"Error training Naive Bayes: {e}")

                # 3. Train Score Predictor (RNN/MLP)
                if len(scores_history) >= 10:
                    try:
                        X_s = []
                        y_s = []
                        for i in range(len(scores_history) - 5):
                            X_s.append(scores_history[i:i+5])
                            y_s.append(scores_history[i+5])
                        X_s = torch.tensor(X_s, dtype=torch.float32)
                        y_s = torch.tensor(y_s, dtype=torch.float32).unsqueeze(1)
                        
                        if score_predictor_model is None:
                            score_predictor_model = ScorePredictor()
                        
                        opt = optim.Adam(score_predictor_model.parameters(), lr=0.01)
                        crit = nn.MSELoss()
                        score_predictor_model.train()
                        for _ in range(15):
                            opt.zero_grad()
                            loss = crit(score_predictor_model(X_s), y_s)
                            loss.backward()
                            opt.step()
                            
                        score_predictor_model.eval()
                        with torch.no_grad():
                            last_5 = torch.tensor([scores_history[-5:]], dtype=torch.float32)
                            predicted_next_score = round(max(0.0, score_predictor_model(last_5).item()), 1)
                    except Exception as e:
                        print(f"Error training Score Predictor: {e}")

# --- Tetris AI Models & Heltor Functions ---
import random

class TetrisMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.fc(x)

tetris_mlp_model = None
tetris_tree_model = None
tetris_knn_model = None
tetris_replay_buffer = deque(maxlen=2000)

# Base Heuristic Weights
tetris_base_weights = {
    "height": -0.510066,
    "lines": 0.760666,
    "holes": -0.35663,
    "bumpiness": -0.184483
}

# Active weights (perturbed during current game for search)
tetris_active_weights = dict(tetris_base_weights)
tetris_perturbed = False
tetris_perturbation = {
    "height": 0.0,
    "lines": 0.0,
    "holes": 0.0,
    "bumpiness": 0.0
}
tetris_avg_score = 0.0
tetris_game_count = 0

def perturb_weights():
    global tetris_active_weights, tetris_perturbed, tetris_perturbation
    # Normal distribution mutations
    tetris_perturbation = {
        "height": random.normalvariate(0, 0.08),
        "lines": random.normalvariate(0, 0.08),
        "holes": random.normalvariate(0, 0.08),
        "bumpiness": random.normalvariate(0, 0.08)
    }
    # Perturb
    tetris_active_weights = {
        k: tetris_base_weights[k] + tetris_perturbation[k]
        for k in tetris_base_weights
    }
    # Clamp signs to keep intuitive meaning (penalties must be negative, line clears positive)
    tetris_active_weights["height"] = min(-0.02, tetris_active_weights["height"])
    tetris_active_weights["holes"] = min(-0.02, tetris_active_weights["holes"])
    tetris_active_weights["bumpiness"] = min(-0.02, tetris_active_weights["bumpiness"])
    tetris_active_weights["lines"] = max(0.02, tetris_active_weights["lines"])
    tetris_perturbed = True

def update_weights(game_score: float):
    global tetris_base_weights, tetris_avg_score, tetris_game_count, tetris_perturbed
    tetris_game_count += 1
    
    if tetris_game_count == 1:
        tetris_avg_score = float(game_score)
    else:
        # Running average of scores with momentum
        tetris_avg_score = 0.85 * tetris_avg_score + 0.15 * game_score
        
    if tetris_perturbed:
        score_diff = game_score - tetris_avg_score
        # Calculate learning rate based on relative score improvement
        # Higher improvement = larger weight update in that direction
        lr = 0.12 * float(np.tanh(score_diff / 500.0)) if score_diff != 0 else 0.0
        
        for k in tetris_base_weights:
            tetris_base_weights[k] += lr * tetris_perturbation[k]
            
        # Keep signs consistent
        tetris_base_weights["height"] = min(-0.02, tetris_base_weights["height"])
        tetris_base_weights["holes"] = min(-0.02, tetris_base_weights["holes"])
        tetris_base_weights["bumpiness"] = min(-0.02, tetris_base_weights["bumpiness"])
        tetris_base_weights["lines"] = max(0.02, tetris_base_weights["lines"])
        
    # Prepare active weights for next game
    perturb_weights()

class TetrisMoveRequest(BaseModel):
    board: list[list[int]]
    shape: list[list[int]]
    model_type: str # "mlp" | "tree" | "knn" | "genetic"


def rotate_matrix(shape):
    return [list(x) for x in zip(*shape[::-1])]

def calculate_tetris_features(board):
    ROWS = 20
    COLS = 10
    
    # 1. Lines cleared
    lines_cleared = 0
    temp_board = []
    for r in range(ROWS):
        if all(board[r][c] != 0 for c in range(COLS)):
            lines_cleared += 1
        else:
            temp_board.append(board[r])
            
    while len(temp_board) < ROWS:
        temp_board.insert(0, [0] * COLS)
        
    # 2. Heights
    column_heights = [0] * COLS
    for c in range(COLS):
        for r in range(ROWS):
            if temp_board[r][c] != 0:
                column_heights[c] = ROWS - r
                break
    aggregate_height = sum(column_heights)
    
    # 3. Holes
    holes = 0
    for c in range(COLS):
        block_found = False
        for r in range(ROWS):
            if temp_board[r][c] != 0:
                block_found = True
            elif block_found and temp_board[r][c] == 0:
                holes += 1
                
    # 4. Bumpiness
    bumpiness = 0
    for c in range(COLS - 1):
        bumpiness += abs(column_heights[c] - column_heights[c+1])
        
    # 5. Row transitions
    row_transitions = 0
    for r in range(ROWS):
        if temp_board[r][0] == 0:
            row_transitions += 1
        for c in range(COLS - 1):
            if (temp_board[r][c] == 0) != (temp_board[r][c+1] == 0):
                row_transitions += 1
        if temp_board[r][COLS - 1] == 0:
            row_transitions += 1
            
    # 6. Column transitions
    col_transitions = 0
    for c in range(COLS):
        if temp_board[0][c] != 0:
            col_transitions += 1
        for r in range(ROWS - 1):
            if (temp_board[r][c] == 0) != (temp_board[r+1][c] == 0):
                col_transitions += 1
        if temp_board[ROWS - 1][c] == 0:
            col_transitions += 1
            
    # 7. Wells
    wells = 0
    for c in range(COLS):
        left_h = ROWS if c == 0 else column_heights[c - 1]
        right_h = ROWS if c == COLS - 1 else column_heights[c + 1]
        target_h = min(left_h, right_h)
        if target_h > column_heights[c]:
            depth = target_h - column_heights[c]
            wells += sum(range(1, depth + 1))
        
    return aggregate_height, lines_cleared, holes, bumpiness, row_transitions, col_transitions, wells


def explain_tetris_tree_decision(model, x):
    if model is None:
        return "Arborele se antrenează..."
    
    tree_ = model.tree_
    node = 0
    conditions = []
    feature_names = ["Înălțime", "Linii Curățate", "Goluri", "Denivelare"]
    
    while tree_.children_left[node] != -1:
        feat = tree_.feature[node]
        val = x[feat]
        thresh = tree_.threshold[node]
        
        if val <= thresh:
            conditions.append(f"{feature_names[feat]} <= {round(thresh, 1)}")
            node = tree_.children_left[node]
        else:
            conditions.append(f"{feature_names[feat]} > {round(thresh, 1)}")
            node = tree_.children_right[node]
            
    val_at_leaf = float(tree_.value[node][0][0])
    explanation = "Dacă " + " ȘI ".join(conditions) + " ➜ Scor Estimator: " + f"{round(val_at_leaf, 2)}"
    return explanation

def pretrain_tetris_models():
    global tetris_mlp_model, tetris_tree_model, tetris_knn_model
    print("Pre-training Tetris models...")
    try:
        X_features = []
        X_grids = []
        y = []
        
        # We generate 3,000 synthetic states and evaluate them
        for _ in range(3000):
            board = [[0]*10 for _ in range(20)]
            for c in range(10):
                h = np.random.randint(0, 8)
                for r in range(20 - h, 20):
                    board[r][c] = 1 if np.random.rand() > 0.15 else 0
                    
            h_agg, lines, holes, bump, _, _, _ = calculate_tetris_features(board)
            score = -0.510066 * h_agg + 0.760666 * lines - 0.35663 * holes - 0.184483 * bump
            
            X_features.append([float(h_agg), float(lines), float(holes), float(bump)])
            X_grids.append(np.array(board, dtype=float).flatten())
            y.append(float(score))
            
        X_features = np.array(X_features)
        X_grids = np.array(X_grids)
        y = np.array(y)
        
        # 1. Fit Decision Tree Regressor
        tetris_tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
        tetris_tree_model.fit(X_features, y)
        
        # 2. Fit KNN Regressor (Slide 15) on 200 raw grid features (like Conlan dataset)
        tetris_knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        tetris_knn_model.fit(X_grids, y)
        
        # 3. Fit MLP Regressor
        tetris_mlp_model = TetrisMLP()
        opt = optim.Adam(tetris_mlp_model.parameters(), lr=0.01)
        crit = nn.MSELoss()
        
        X_t = torch.tensor(X_features, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        tetris_mlp_model.train()
        for _ in range(20):
            opt.zero_grad()
            loss = crit(tetris_mlp_model(X_t), y_t)
            loss.backward()
            opt.step()
            
        tetris_mlp_model.eval()
        print("Tetris models pre-training complete!")
    except Exception as e:
        print(f"Failed to pre-train Tetris models: {e}")

def check_collision(board, x, y, shape):
    ROWS = 20
    COLS = 10
    for r in range(len(shape)):
        for c in range(len(shape[r])):
            if shape[r][c]:
                nx = x + c
                ny = y + r
                if nx < 0 or nx >= COLS or ny >= ROWS:
                    return True
                if ny >= 0 and board[ny][nx] != 0:
                    return True
    return False

@app.post("/api/tetris/next-move")
async def post_tetris_next_move(req: TetrisMoveRequest):
    board = req.board
    original_shape = req.shape
    model_type = req.model_type
    
    best_score = -999999.0
    best_x = 0
    best_shape = original_shape
    best_features = [0.0, 0.0, 0.0, 0.0]
    best_extra = [0, 0, 0]
    
    # Try all 4 rotations
    current_shape = original_shape
    for rot in range(4):
        # Try all possible columns
        for x in range(-2, 12):
            if not check_collision(board, x, 0, current_shape):
                # Find landing y
                y = 0
                while not check_collision(board, x, y + 1, current_shape):
                    y += 1
                    
                # Simulate placement
                temp_board = [row[:] for row in board]
                for r in range(len(current_shape)):
                    for c in range(len(current_shape[r])):
                        if current_shape[r][c]:
                            ny = y + r
                            nx = x + c
                            if 0 <= ny < 20 and 0 <= nx < 10:
                                temp_board[ny][nx] = 1
                                
                # Calculate features
                h_agg, lines, holes, bump, row_trans, col_trans, wells = calculate_tetris_features(temp_board)
                features = [float(h_agg), float(lines), float(holes), float(bump)]
                
                # Predict score using chosen model
                if model_type == "mlp" and tetris_mlp_model is not None:
                    feat_tensor = torch.tensor([features], dtype=torch.float32)
                    with torch.no_grad():
                        score = float(tetris_mlp_model(feat_tensor).item())
                elif model_type == "tree" and tetris_tree_model is not None:
                    score = float(tetris_tree_model.predict([features])[0])
                elif model_type == "knn" and tetris_knn_model is not None:
                    grid_flat = np.array(temp_board, dtype=float).flatten()
                    score = float(tetris_knn_model.predict([grid_flat])[0])
                elif model_type == "genetic":
                    score = (
                        tetris_active_weights["height"] * h_agg
                        + tetris_active_weights["lines"] * lines
                        + tetris_active_weights["holes"] * holes
                        + tetris_active_weights["bumpiness"] * bump
                    )
                else:
                    score = -0.510066 * h_agg + 0.760666 * lines - 0.35663 * holes - 0.184483 * bump
                    
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_shape = current_shape
                    best_features = features
                    best_extra = [row_trans, col_trans, wells]
                    
        current_shape = rotate_matrix(current_shape)
        
    explanation = ""
    if model_type == "tree" and tetris_tree_model is not None:
        explanation = explain_tetris_tree_decision(tetris_tree_model, best_features)
    elif model_type == "knn" and tetris_knn_model is not None:
        explanation = f"Evaluare KNN: Calitate prezisă din cele mai similare 5 grile din istoric (Scor = {round(best_score, 2)})."
    elif model_type == "mlp" and tetris_mlp_model is not None:
        explanation = f"Evaluare MLP: Scor plasare prezis = {round(best_score, 2)} pe baza metricilor tablei."
    elif model_type == "genetic":
        explanation = f"Evaluare Genetică (ES): Scor = {round(best_score, 2)} pe baza ponderilor active."
    else:
        explanation = f"Evaluare Euristică: Scor = {round(best_score, 2)}."
        
    return {
        "x": best_x,
        "shape": best_shape,
        "explanation": explanation,
        "weights": tetris_active_weights,
        "metrics": {
            "height": int(best_features[0]),
            "lines": int(best_features[1]),
            "holes": int(best_features[2]),
            "bumpiness": int(best_features[3]),
            "row_transitions": int(best_extra[0]),
            "col_transitions": int(best_extra[1]),
            "wells": int(best_extra[2])
        }
    }


class TetrisEvaluateRequest(BaseModel):
    board: list[list[int]]
    shape: list[list[int]]
    chosen_x: int
    chosen_shape: list[list[int]]
    model_type: str # "mlp" | "tree" | "knn" | "genetic"

@app.post("/api/tetris/evaluate-move")
async def post_tetris_evaluate_move(req: TetrisEvaluateRequest):
    board = req.board
    original_shape = req.shape
    chosen_x = req.chosen_x
    chosen_shape = req.chosen_shape
    model_type = req.model_type
    
    # 1. Simulate all possible placements and evaluate them using chosen model
    placements = []
    current_shape = original_shape
    for rot in range(4):
        for x in range(-2, 12):
            if not check_collision(board, x, 0, current_shape):
                y = 0
                while not check_collision(board, x, y + 1, current_shape):
                    y += 1
                
                # Simulate placement
                temp_board = [row[:] for row in board]
                for r in range(len(current_shape)):
                    for c in range(len(current_shape[r])):
                        if current_shape[r][c]:
                            ny = y + r
                            nx = x + c
                            if 0 <= ny < 20 and 0 <= nx < 10:
                                temp_board[ny][nx] = 1
                                
                h_agg, lines, holes, bump, row_trans, col_trans, wells = calculate_tetris_features(temp_board)
                features = [float(h_agg), float(lines), float(holes), float(bump)]
                
                # Predict score via chosen model
                score = -999999.0
                if model_type == "mlp" and tetris_mlp_model is not None:
                    feat_tensor = torch.tensor([features], dtype=torch.float32)
                    with torch.no_grad():
                        score = float(tetris_mlp_model(feat_tensor).item())
                elif model_type == "tree" and tetris_tree_model is not None:
                    score = float(tetris_tree_model.predict([features])[0])
                elif model_type == "knn" and tetris_knn_model is not None:
                    grid_flat = np.array(temp_board, dtype=float).flatten()
                    score = float(tetris_knn_model.predict([grid_flat])[0])
                elif model_type == "genetic":
                    score = (
                        tetris_active_weights["height"] * h_agg
                        + tetris_active_weights["lines"] * lines
                        + tetris_active_weights["holes"] * holes
                        + tetris_active_weights["bumpiness"] * bump
                    )
                else:
                    score = -0.510066 * h_agg + 0.760666 * lines - 0.35663 * holes - 0.184483 * bump
                    
                placements.append({
                    "score": score,
                    "x": x,
                    "shape": current_shape,
                    "features": features,
                    "extra": [row_trans, col_trans, wells],
                    "temp_board": temp_board
                })
        current_shape = rotate_matrix(current_shape)
        
    if not placements:
        return {"classification": "Average", "explanation": "Move not evaluated (unusual state)."}
        
    # Sort placements by score descending
    placements.sort(key=lambda p: p["score"], reverse=True)
    best_move = placements[0]
    
    # 2. Evaluate the chosen move
    chosen_y = 0
    while not check_collision(board, chosen_x, chosen_y + 1, chosen_shape):
        chosen_y += 1
        
    simulated_chosen_board = [row[:] for row in board]
    for r in range(len(chosen_shape)):
        for c in range(len(chosen_shape[r])):
            if chosen_shape[r][c]:
                ny = chosen_y + r
                nx = chosen_x + c
                if 0 <= ny < 20 and 0 <= nx < 10:
                    simulated_chosen_board[ny][nx] = 1
                    
    chosen_h_agg, chosen_lines, chosen_holes, chosen_bump, chosen_row_trans, chosen_col_trans, chosen_wells = calculate_tetris_features(simulated_chosen_board)
    chosen_features = [float(chosen_h_agg), float(chosen_lines), float(chosen_holes), float(chosen_bump)]
    
    chosen_score = -999999.0
    if model_type == "mlp" and tetris_mlp_model is not None:
        feat_tensor = torch.tensor([chosen_features], dtype=torch.float32)
        with torch.no_grad():
            chosen_score = float(tetris_mlp_model(feat_tensor).item())
    elif model_type == "tree" and tetris_tree_model is not None:
        chosen_score = float(tetris_tree_model.predict([chosen_features])[0])
    elif model_type == "knn" and tetris_knn_model is not None:
        grid_flat = np.array(simulated_chosen_board, dtype=float).flatten()
        chosen_score = float(tetris_knn_model.predict([grid_flat])[0])
    elif model_type == "genetic":
        chosen_score = (
            tetris_active_weights["height"] * chosen_h_agg
            + tetris_active_weights["lines"] * chosen_lines
            + tetris_active_weights["holes"] * chosen_holes
            + tetris_active_weights["bumpiness"] * chosen_bump
        )
    else:
        chosen_score = -0.510066 * chosen_h_agg + 0.760666 * chosen_lines - 0.35663 * chosen_holes - 0.184483 * chosen_bump
        
    # Find rank of chosen_score
    match_index = -1
    for i, p in enumerate(placements):
        if p["x"] == chosen_x and np.array_equal(p["shape"], chosen_shape):
            match_index = i
            break
            
    if match_index != -1:
        rank = match_index + 1
    else:
        rank = sum(1 for p in placements if p["score"] > chosen_score + 1e-5) + 1
        
    total_possibilities = len(placements)
    percentile = (rank - 1) / total_possibilities if total_possibilities > 1 else 0.0
    
    # Classify move
    classification = "Mistake"
    if rank == 1 or percentile <= 0.10:
        classification = "Excellent"
    elif percentile <= 0.30:
        classification = "Good"
    elif percentile <= 0.60:
        classification = "Average"
    else:
        classification = "Mistake"
        
    # 3. Generate explanation
    best_h_agg, best_lines, best_holes, best_bump = best_move["features"]
    
    explanation = ""
    if classification == "Excellent":
        explanation = "Excellent move! You placed the piece perfectly. "
        if chosen_lines > 0:
            explanation += f"You cleared {chosen_lines} line(s)!"
        else:
            explanation += "You kept the board clean without creating new holes."
    else:
        reasons = []
        if chosen_holes > best_holes:
            reasons.append(f"created {int(chosen_holes - best_holes)} extra hole(s)")
        if chosen_h_agg > best_h_agg:
            reasons.append(f"increased height by {int(chosen_h_agg - best_h_agg)} units")
        if chosen_bump > best_bump:
            reasons.append(f"increased bumpiness by {int(chosen_bump - best_bump)} units")
            
        reason_str = ", ".join(reasons) if reasons else "has a less optimal placement"
        explanation = f"The move is classified as a {classification.lower()} because it {reason_str}. "
        explanation += f"The best move would have been at column {best_move['x'] + 1} "
        if best_lines > 0:
            explanation += f"(it would have cleared {int(best_lines)} line(s))."
        else:
            explanation += "(it would have left the board smoother and without holes)."
            
    return {
        "classification": classification,
        "score": round(chosen_score, 2),
        "best_score": round(best_move["score"], 2),
        "explanation": explanation,
        "best_x": best_move["x"],
        "metrics": {
            "height": int(chosen_h_agg),
            "lines": int(chosen_lines),
            "holes": int(chosen_holes),
            "bumpiness": int(chosen_bump),
            "row_transitions": int(chosen_row_trans),
            "col_transitions": int(chosen_col_trans),
            "wells": int(chosen_wells)
        }
    }

class TetrisGameOverRequest(BaseModel):
    score: int
    history: list[dict] # list of {"features": list, "grid": list, "reward": float}

@app.post("/api/tetris/game-over")
async def post_tetris_game_over(req: TetrisGameOverRequest):
    global tetris_mlp_model, tetris_tree_model, tetris_knn_model
    
    # 1. Update Genetic ES weights
    update_weights(req.score)
    
    # 2. Process history for TD-learning of MLP, Tree, KNN
    history = req.history
    if len(history) > 1:
        gamma = 0.95
        # Compute target values backwards (TD-style bootstrapping)
        for i in range(len(history)):
            state = history[i]
            reward = state.get("reward", 0.0)
            
            if i == len(history) - 1:
                # Terminal step penalty
                target = reward - 100.0
            else:
                next_state = history[i+1]
                h_agg, lines, holes, bump = next_state["features"]
                # Bootstrap next state value from active heuristic weights
                v_next = (
                    tetris_base_weights["height"] * h_agg
                    + tetris_base_weights["lines"] * lines
                    + tetris_base_weights["holes"] * holes
                    + tetris_base_weights["bumpiness"] * bump
                )
                target = reward + gamma * v_next
                
            tetris_replay_buffer.append({
                "features": state["features"],
                "grid": state["grid"],
                "target": target
            })
            
        # 3. Retrain ML models
        if len(tetris_replay_buffer) >= 80:
            try:
                X_features = np.array([item["features"] for item in tetris_replay_buffer])
                X_grids = np.array([item["grid"] for item in tetris_replay_buffer])
                y = np.array([item["target"] for item in tetris_replay_buffer])
                
                # Fit Decision Tree
                tetris_tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
                tetris_tree_model.fit(X_features, y)
                
                # Fit KNN
                tetris_knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
                tetris_knn_model.fit(X_grids, y)
                
                # Train MLP
                if tetris_mlp_model is None:
                    tetris_mlp_model = TetrisMLP()
                opt = optim.Adam(tetris_mlp_model.parameters(), lr=0.01)
                crit = nn.MSELoss()
                X_t = torch.tensor(X_features, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
                
                tetris_mlp_model.train()
                for _ in range(5):
                    opt.zero_grad()
                    loss = crit(tetris_mlp_model(X_t), y_t)
                    loss.backward()
                    opt.step()
                tetris_mlp_model.eval()
                print(f"Online learning done: retrained Tetris models on {len(X_features)} samples.")
            except Exception as e:
                print(f"Error training online Tetris models: {e}")
                
    return {
        "status": "success",
        "avg_score": round(tetris_avg_score, 1),
        "weights": tetris_base_weights
    }


@app.on_event("startup")
async def startup_event():
    pretrain_models_at_startup()
    pretrain_tetris_models()
    asyncio.create_task(training_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data_str = await websocket.receive_text()
            try:
                data = json.loads(data_str)
                if data.get("type") == "mode":
                    server_state["is_manual"] = data.get("manual", False)
                    if server_state["is_manual"]:
                        server_state["manual_key"] = None
                        server_state["reset_requested"] = True
                elif data.get("type") == "ai_mode":
                    server_state["ai_mode"] = data.get("ai_mode", "dqn")
                elif data.get("type") == "action":
                    server_state["manual_key"] = data.get("key")
                elif data.get("type") == "reset":
                    server_state["reset_requested"] = True
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5050, reload=False)
