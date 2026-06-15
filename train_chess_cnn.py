"""
Train ChessCNN on positions evaluated by the Minimax+PST teacher.
Run once: python train_chess_cnn.py
Saves weights to model/chess_cnn.pth
"""
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from chess_cnn import ChessCNN, board_to_tensor, MODEL_PATH

# ---- Piece-square tables (teacher evaluation) ----
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]
KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]
BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]
ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
]
QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]
KING_TABLE_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

PIECE_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE_MG,
}

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

def pst_evaluate(board: chess.Board) -> float:
    """Material + piece-square table evaluation (teacher signal)."""
    if board.is_checkmate():
        return -99999 if board.turn else 99999
    if board.is_game_over():
        return 0

    score = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        table = PIECE_TABLES[piece.piece_type]
        # For white: rank 0 = rank 1 on board. For black: mirror
        if piece.color == chess.WHITE:
            pst_idx = (7 - rank) * 8 + file
            score += PIECE_VALUES[piece.piece_type] + table[pst_idx]
        else:
            pst_idx = rank * 8 + file
            score -= PIECE_VALUES[piece.piece_type] + table[pst_idx]
    return score


def generate_positions(n: int = 2500, max_moves: int = 60) -> list[tuple]:
    """Generate (board_tensor, score) pairs from random self-play."""
    data = []
    print(f"Generating {n} training positions...")
    while len(data) < n:
        board = chess.Board()
        positions_in_game = []
        for _ in range(max_moves):
            if board.is_game_over():
                break
            positions_in_game.append(board.copy())
            move = random.choice(list(board.legal_moves))
            board.push(move)
        for pos in positions_in_game:
            score = pst_evaluate(pos) / 10000.0  # normalize
            tensor = board_to_tensor(pos)
            data.append((tensor, torch.tensor([score], dtype=torch.float32)))
        if len(data) % 500 == 0:
            print(f"  {len(data)}/{n} positions collected")
    return data[:n]


def train(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    os.makedirs("model", exist_ok=True)
    model = ChessCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    data = generate_positions(2500)
    random.shuffle(data)

    print(f"\nTraining CNN for {epochs} epochs...")
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        batches = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            tensors = torch.stack([b[0] for b in batch])
            labels  = torch.stack([b[1] for b in batch])

            optimizer.zero_grad()
            preds = model(tensors)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        avg = total_loss / batches
        print(f"  Epoch {epoch+1:2d}/{epochs}  loss={avg:.5f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
