import chess
import torch
import torch.nn as nn
import numpy as np
import os

MODEL_PATH = os.path.join("model", "chess_cnn.pth")

# --- Board encoding ---
PIECE_IDX = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a chess board as a (12, 8, 8) float tensor."""
    t = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            idx = PIECE_IDX[(piece.piece_type, piece.color)]
            t[idx][rank][file] = 1.0
    return torch.tensor(t)


# --- CNN Model ---
class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# --- Singleton model loader ---
_model: ChessCNN | None = None

def get_model() -> ChessCNN:
    global _model
    if _model is None:
        _model = ChessCNN()
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        _model.eval()
    return _model


def cnn_evaluate(board: chess.Board) -> float:
    """Return CNN evaluation score for the position (positive = White advantage)."""
    model = get_model()
    with torch.no_grad():
        t = board_to_tensor(board).unsqueeze(0)  # (1, 12, 8, 8)
        score = model(t).item()
    return score


# --- Minimax using CNN as evaluation function ---
def _minimax_cnn(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    if depth == 0 or board.is_game_over():
        if board.is_checkmate():
            return -99999 if board.turn else 99999
        if board.is_game_over():
            return 0
        return cnn_evaluate(board)

    if maximizing:
        best = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            best = max(best, _minimax_cnn(board, depth - 1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = float("inf")
        for move in board.legal_moves:
            board.push(move)
            best = min(best, _minimax_cnn(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def get_best_move_cnn(board: chess.Board, depth: int = 2) -> str | None:
    """Return the best move UCI string using CNN evaluation inside Minimax."""
    best_move = None
    if board.turn == chess.WHITE:
        best_val = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            val = _minimax_cnn(board, depth - 1, -float("inf"), float("inf"), False)
            board.pop()
            if val > best_val:
                best_val = val
                best_move = move
    else:
        best_val = float("inf")
        for move in board.legal_moves:
            board.push(move)
            val = _minimax_cnn(board, depth - 1, -float("inf"), float("inf"), True)
            board.pop()
            if val < best_val:
                best_val = val
                best_move = move
    return best_move.uci() if best_move else None
