"""
Train ChessCNN on a real Kaggle chess evaluation dataset.

Supported CSV formats:
  - kaggle: fatemehmehrparvar/chess-positions-analyses  → columns: FEN, Evaluation
  - kaggle: ronakbadhe/chess-evaluations                → columns: FEN, Evaluation

Evaluation column values:
  - centipawn: "+0.25", "-1.30"  (in pawns, NOT centipawns in this dataset)
  - mate:      "+M5", "-M3"

Usage:
    python train_chess_cnn.py --csv data/chess_data.csv --samples 80000 --epochs 30
"""

import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import argparse
import pandas as pd
from chess_cnn import ChessCNN, board_to_tensor, MODEL_PATH


# ---- Evaluation parsing ----

def parse_evaluation(raw: str) -> float | None:
    """
    Parse a Stockfish evaluation string to a normalized float in [-1, 1].
    Returns None if the position should be skipped (e.g. impossible FEN).
    """
    raw = str(raw).strip()
    try:
        # Mate scores: "#5", "+M5", "-M3", "#-3"
        if 'M' in raw.upper() or raw.startswith('#'):
            raw_clean = raw.replace('+', '').replace('#', '')
            mate_in = int(raw_clean.replace('M', '').replace('m', ''))
            # White mates → +1.0, Black mates → -1.0
            return 1.0 if mate_in > 0 else -1.0

        # Centipawn scores: "+0.25", "-1.30", "0.00"
        val = float(raw.replace('+', ''))
        # Clamp and normalize: ±10 pawns → ±1.0
        return max(-1.0, min(1.0, val / 10.0))
    except (ValueError, TypeError):
        return None


# ---- Dataset loader ----

def load_dataset(csv_path: str, max_samples: int = 80_000, seed: int = 42) -> list[tuple]:
    """Load FEN positions and evaluations from a CSV file."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows in CSV: {len(df):,}")

    # Detect column names (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    fen_col  = col_map.get('fen')
    eval_col = col_map.get('evaluation') or col_map.get('eval')

    if not fen_col or not eval_col:
        raise ValueError(f"CSV must have 'FEN' and 'Evaluation' columns. Found: {list(df.columns)}")

    # Sample a subset if dataset is large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"  Sampled {max_samples:,} positions")

    data = []
    skipped = 0
    for _, row in df.iterrows():
        fen_str = str(row[fen_col]).strip()
        eval_str = str(row[eval_col]).strip()

        score = parse_evaluation(eval_str)
        if score is None:
            skipped += 1
            continue

        try:
            board = chess.Board(fen_str)
            tensor = board_to_tensor(board)
            label  = torch.tensor([score], dtype=torch.float32)
            data.append((tensor, label))
        except Exception:
            skipped += 1
            continue

    print(f"  Loaded: {len(data):,} valid positions  |  Skipped: {skipped:,}")
    return data


# ---- Fallback: generate synthetic data (PST teacher) ----

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_TABLE = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
BISHOP_TABLE = [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20]
ROOK_TABLE   = [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0]
QUEEN_TABLE  = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20]
KING_TABLE   = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]
PIECE_TABLES = {chess.PAWN:PAWN_TABLE,chess.KNIGHT:KNIGHT_TABLE,chess.BISHOP:BISHOP_TABLE,chess.ROOK:ROOK_TABLE,chess.QUEEN:QUEEN_TABLE,chess.KING:KING_TABLE}
PIECE_VALUES = {chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:20000}

def pst_evaluate(board):
    if board.is_checkmate(): return -1.0 if board.turn else 1.0
    if board.is_game_over(): return 0.0
    score = 0.0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p: continue
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        idx = (7-rank)*8+file if p.color==chess.WHITE else rank*8+file
        val = PIECE_VALUES[p.piece_type] + PIECE_TABLES[p.piece_type][idx]
        score += val if p.color==chess.WHITE else -val
    return max(-1.0, min(1.0, score/10000.0))

def generate_synthetic(n=3000):
    print(f"Generating {n} synthetic positions (PST teacher)...")
    data = []
    while len(data) < n:
        board = chess.Board()
        positions = []
        for _ in range(60):
            if board.is_game_over(): break
            positions.append(board.copy())
            board.push(random.choice(list(board.legal_moves)))
        for pos in positions:
            score = pst_evaluate(pos)
            data.append((board_to_tensor(pos), torch.tensor([score], dtype=torch.float32)))
        if len(data) % 500 == 0:
            print(f"  {len(data)}/{n}")
    return data[:n]


# ---- Training loop ----

def train(data: list, epochs: int = 30, batch_size: int = 128, lr: float = 1e-3):
    os.makedirs("model", exist_ok=True)
    model = ChessCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    random.shuffle(data)
    val_size = int(len(data) * 0.1)
    val_data  = data[:val_size]
    train_data = data[val_size:]
    print(f"\nTraining on {len(train_data):,} positions, validating on {len(val_data):,}")
    print(f"Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {lr}\n")

    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Train
        model.train()
        random.shuffle(train_data)
        train_loss = 0.0
        batches = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            tensors = torch.stack([b[0] for b in batch])
            labels  = torch.stack([b[1] for b in batch])
            optimizer.zero_grad()
            loss = criterion(model(tensors), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                tensors = torch.stack([b[0] for b in batch])
                labels  = torch.stack([b[1] for b in batch])
                val_loss += criterion(model(tensors), labels).item()
        val_loss /= max(1, len(val_data) // batch_size)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            best_marker = " ← best ✅"
        else:
            best_marker = ""

        print(f"  Epoch {epoch+1:2d}/{epochs}  train={train_loss/batches:.5f}  val={val_loss:.5f}{best_marker}")

    print(f"\n✅ Best model saved to {MODEL_PATH}  (val_loss={best_val_loss:.5f})")


# ---- Entry point ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chess CNN")
    parser.add_argument('--csv',     type=str,   default=None,  help="Path to CSV dataset (FEN + Evaluation columns)")
    parser.add_argument('--samples', type=int,   default=80_000, help="Max positions to use from CSV")
    parser.add_argument('--epochs',  type=int,   default=30,    help="Training epochs")
    parser.add_argument('--batch',   type=int,   default=128,   help="Batch size")
    parser.add_argument('--lr',      type=float, default=1e-3,  help="Learning rate")
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        data = load_dataset(args.csv, max_samples=args.samples)
    else:
        if args.csv:
            print(f"⚠️  CSV not found: {args.csv}")
        print("Using synthetic PST-based data as fallback.")
        data = generate_synthetic(3000)

    if not data:
        print("❌ No data loaded. Exiting.")
        exit(1)

    train(data, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
