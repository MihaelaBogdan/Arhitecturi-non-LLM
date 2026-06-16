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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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
        game = SnakeGameHeadless()
        
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
    game = SnakeGameHeadless()
    
    # Initialize plot scores from startup pre-training to populate the chart immediately
    plot_scores = list(scores_history)
    total_score = sum(plot_scores)
    plot_mean_scores = []
    for i in range(1, len(plot_scores) + 1):
        plot_mean_scores.append(sum(plot_scores[:i]) / i)
        
    agent.n_games = len(plot_scores)
    record = max(plot_scores) if plot_scores else 0
    
    while True:
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
        if manager.active_connections:
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
        else:
            await asyncio.sleep(0.001)

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

@app.on_event("startup")
async def startup_event():
    pretrain_models_at_startup()
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
