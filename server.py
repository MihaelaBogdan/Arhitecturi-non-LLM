import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from game_logic import SnakeGameHeadless
from agent import Agent
from pydantic import BaseModel
import chess
from chess_ai import get_best_move, evaluate_board, detect_opening, explain_move, get_material_balance
from chess_cnn import get_best_move_cnn, cnn_evaluate
import json
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

class ChessAnalyzeRequest(BaseModel):
    fen: str
    last_move: str | None = None

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
    board = chess.Board(req.fen)
    # Hint always for the side to move (should be White = player)
    best_move = await asyncio.to_thread(get_best_move, board, min(req.depth, 2))
    return {"hint": best_move}

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
    "ai_mode": "dqn"  # "dqn" | "tree" | "bayes"
}

# --- Snake AI Models & Data Buffers ---
MAX_BUFFER = 5_000
training_data_x = deque(maxlen=MAX_BUFFER)
training_data_y = deque(maxlen=MAX_BUFFER)

collision_data_x = deque(maxlen=MAX_BUFFER)
collision_data_y = deque(maxlen=MAX_BUFFER)

scores_history = []

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
        return "Arborele se antrenează... Așteaptă finalul primului joc."
    
    tree_ = model.tree_
    node = 0
    conditions = []
    
    action_names = ["ÎNAINTE", "DREAPTA", "STÂNGA"]
    feature_names = [
        "Pericol Înainte",
        "Pericol Dreapta",
        "Pericol Stânga",
        "Direcție Stânga",
        "Direcție Dreapta",
        "Direcție Sus",
        "Direcție Jos",
        "Mâncare Stânga",
        "Mâncare Dreapta",
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

async def training_loop():
    global server_state, tree_model, bayes_model, score_predictor_model, predicted_next_score
    agent = Agent()
    game = SnakeGameHeadless(w=400, h=400)
    
    # Initialize plot scores from startup pre-training to populate the chart immediately
    plot_scores = list(scores_history)
    total_score = sum(plot_scores)
    plot_mean_scores = []
    for i in range(1, len(plot_scores) + 1):
        plot_mean_scores.append(sum(plot_scores[:i]) / i)
        
    agent.n_games = len(plot_scores)
    record = max(plot_scores) if plot_scores else 0
    
    while True:
        if not manager.active_connections:
            await asyncio.sleep(0.5)
            continue

        state_old = agent.get_state(game)
        
        # Calculate Naive Bayes risks for current step
        current_bayes_risks = {"straight": 0.0, "right": 0.0, "left": 0.0}
        if bayes_model is not None:
            try:
                risks = []
                for a in range(3):
                    sample = np.append(state_old, a).reshape(1, -1)
                    proba = bayes_model.predict_proba(sample)[0][1]
                    risks.append(float(proba))
                current_bayes_risks = {"straight": risks[0], "right": risks[1], "left": risks[2]}
            except:
                pass

        if server_state["is_manual"]:
            # Convert manual key to relative action
            action = [1, 0, 0] # default straight
            if server_state["manual_key"]:
                k = server_state["manual_key"]
                d = game.direction
                from game_logic import Direction
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
        else:
            # AI Control
            ai_mode = server_state.get("ai_mode", "dqn")
            
            # Pre-calculate greedy DQN action (without exploration noise)
            state_tensor = torch.tensor(state_old, dtype=torch.float)
            with torch.no_grad():
                pred = agent.model(state_tensor)
            greedy_action_idx = torch.argmax(pred).item()
            
            if ai_mode == "tree" and tree_model is not None:
                try:
                    action_idx = tree_model.predict(state_old.reshape(1, -1))[0]
                    final_move = [0, 0, 0]
                    final_move[action_idx] = 1
                except:
                    final_move = agent.get_action(state_old)
            elif ai_mode == "bayes":
                # Start with greedy DQN action as baseline
                action_idx = greedy_action_idx
                if bayes_model is not None:
                    try:
                        risks = []
                        for a in range(3):
                            sample = np.append(state_old, a).reshape(1, -1)
                            proba = bayes_model.predict_proba(sample)[0][1]
                            risks.append(proba)
                        # Safety override: if the greedy move is risky (> 0.4), choose the safest one!
                        if risks[action_idx] > 0.4:
                            action_idx = np.argmin(risks)
                    except:
                        pass
                final_move = [0, 0, 0]
                final_move[action_idx] = 1
            else:
                # Default to DQN (includes exploration during training)
                final_move = agent.get_action(state_old)
            
            reward, done, score = game.play_step(final_move)
            
            # Data collection for tree and bayes (collect whenever AI plays)
            # We train the Decision Tree ONLY on greedy/smart actions of DQN to avoid exploration noise
            training_data_x.append(state_old)
            training_data_y.append(greedy_action_idx)
            
            # We record actual collision data based on the chosen move and its outcome
            action_chosen_idx = np.argmax(final_move)
            collision_data_x.append(np.append(state_old, action_chosen_idx))
            collision_data_y.append(1 if done else 0)

            # DQN updates (only if in DQN mode)
            if ai_mode == "dqn":
                state_new = agent.get_state(game)
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)

        # Broadcast state
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
            "predicted_next_score": predicted_next_score
        }
        await manager.broadcast(state_msg)
        # Control speed for visualization
        await asyncio.sleep(0.04) # 25 FPS

        if done:
            game.reset()
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
            
            # Update scores history
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

# --- Tetris AI Models & Helper Functions ---
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

class TetrisMoveRequest(BaseModel):
    board: list[list[int]]
    shape: list[list[int]]
    model_type: str # "mlp" | "tree" | "knn"

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
        explanation = f"Evaluare KNN: Calitate prezisă din cele mai similare 5 grile de 20x10 din istoric (Scor = {round(best_score, 2)})."
    elif model_type == "mlp":
        explanation = f"Evaluare MLP: Scor plasare prezis = {round(best_score, 2)} pe baza metricilor tablei."
        
    return {
        "x": best_x,
        "shape": best_shape,
        "explanation": explanation,
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

@app.post("/api/tetris/evaluate-move")
async def post_tetris_evaluate_move(req: TetrisEvaluateRequest):
    board = req.board
    original_shape = req.shape
    chosen_x = req.chosen_x
    chosen_shape = req.chosen_shape
    
    # 1. Simulate all possible placements and evaluate them using KNN
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
                
                # Predict score via KNN
                score = -999999.0
                if tetris_knn_model is not None:
                    grid_flat = np.array(temp_board, dtype=float).flatten()
                    score = float(tetris_knn_model.predict([grid_flat])[0])
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
        return {"classification": "Medie", "explanation": "Mutare neevaluată (stare neobișnuită)."}
        
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
    if tetris_knn_model is not None:
        grid_flat = np.array(simulated_chosen_board, dtype=float).flatten()
        chosen_score = float(tetris_knn_model.predict([grid_flat])[0])
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
    classification = "Greșeală"
    if rank == 1 or percentile <= 0.10:
        classification = "Excelentă"
    elif percentile <= 0.30:
        classification = "Bună"
    elif percentile <= 0.60:
        classification = "Medie"
    else:
        classification = "Greșeală"
        
    # 3. Generate explanation
    best_h_agg, best_lines, best_holes, best_bump = best_move["features"]
    
    explanation = ""
    if classification == "Excelentă":
        explanation = "Mutare excelentă! Ai plasat piesa perfect. "
        if chosen_lines > 0:
            explanation += f"Ai curățat {chosen_lines} linie/linii!"
        else:
            explanation += "Ai menținut tabla curată, fără a crea goluri noi."
    else:
        reasons = []
        if chosen_holes > best_holes:
            reasons.append(f"a creat {int(chosen_holes - best_holes)} gol(uri) în plus")
        if chosen_h_agg > best_h_agg:
            reasons.append(f"a crescut înălțimea cu {int(chosen_h_agg - best_h_agg)} unități")
        if chosen_bump > best_bump:
            reasons.append(f"a mărit denivelarea cu {int(chosen_bump - best_bump)} unități")
            
        reason_str = ", ".join(reasons) if reasons else "are o așezare mai puțin optimă"
        explanation = f"Mutarea este clasificată ca {classification.lower()} deoarece {reason_str}. "
        explanation += f"Cea mai bună mutare ar fi fost la coloana {best_move['x'] + 1} "
        if best_lines > 0:
            explanation += f"(ar fi curățat {int(best_lines)} linie/linii)."
        else:
            explanation += "(ar fi lăsat tabla mai netedă și fără goluri)."
            
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
                elif data.get("type") == "ai_mode":
                    server_state["ai_mode"] = data.get("ai_mode", "dqn")
                elif data.get("type") == "action":
                    server_state["manual_key"] = data.get("key")
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5050, reload=False)
