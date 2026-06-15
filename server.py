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
import asyncio

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
    "manual_key": None
}

async def training_loop():
    global server_state
    agent = Agent()
    game = SnakeGameHeadless()
    record = 0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    while True:
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
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
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
                "mean_scores": plot_mean_scores
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
                agent.train_long_memory()

            if score > record:
                record = score
                if not server_state["is_manual"]:
                    agent.model.save()
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

@app.on_event("startup")
async def startup_event():
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
                elif data.get("type") == "action":
                    server_state["manual_key"] = data.get("key")
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5050, reload=False)
