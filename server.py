import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from game_logic import SnakeGameHeadless
from agent import Agent
from pydantic import BaseModel
import chess
from chess_ai import get_best_move, evaluate_board, detect_opening, explain_move, get_material_balance
from chess_cnn import get_best_move_cnn, cnn_evaluate
from next_move_predictor import predict_next_move
from opening_coach import get_opening_coach
from board_recognition_cnn import recognize_board, recognize_board_with_confidence
import json
import asyncio
import base64
import io
import tempfile

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
    board = chess.Board(req.fen)
    # Hint always for the side to move (should be White = player)
    best_move = await asyncio.to_thread(get_best_move, board, min(req.depth, 2))
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
                "error": "Modelul nu a detectat ambii regi pe tablă. Asigură-te că imaginea conține întreaga tablă de șah și că este decupată corect pe conturul acesteia.",
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
                "error": "Modelul nu a detectat ambii regi pe tablă. Asigură-te că imaginea conține întreaga tablă de șah și că este decupată corect pe conturul acesteia.",
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
        return "⬜ Avantaj ușor pentru Alb"
    elif s > -30:
        return "= Poziție egală"
    elif s > -100:
        return "⬛ Avantaj ușor pentru Negru"
    elif s > -200:
        return "⬛ Avantaj mare pentru Negru"
    else:
        return "⬛ Negrul câștigă decisiv"


def _detect_tactics(board: chess.Board) -> list[str]:
    """Detect basic tactical patterns in the position."""
    tactics = []
    
    if board.is_check():
        tactics.append("♚ Regele este în ȘAH!")
    
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
                        chess.PAWN: "Pion", chess.KNIGHT: "Cal",
                        chess.BISHOP: "Nebun", chess.ROOK: "Turn",
                        chess.QUEEN: "Regină", chess.KING: "Rege"
                    }.get(piece.piece_type, "Piesă")
                    sq_name = chess.square_name(sq).upper()
                    tactics.append(f"🎯 {piece_name} neapărat pe {sq_name}!")
    
    if not tactics:
        tactics.append("🔍 Nu am detectat tactici imediate")
    
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
