"""
Next-Move Predictor — Predict the best next move using patterns from real games.

Architecture:
  - CNN that takes board state (12 channels × 8×8) as input
  - Outputs a policy over all possible moves (64×64 from-to = 4096)
  - Trained on positions from real finished games

This complements the existing Minimax+CNN evaluator by providing
move suggestions based on patterns learned from human games.

Usage:
    python next_move_predictor.py --csv data/chess_games.csv --epochs 15
"""

import os
import sys
import random
import argparse
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim

from chess_cnn import board_to_tensor

MODEL_PATH = os.path.join("model", "next_move.pth")

# Move encoding: from_square (0-63) × to_square (0-63) = 4096 possible moves
NUM_MOVE_CLASSES = 4096


def move_to_index(move: chess.Move) -> int:
    """Encode a move as from_sq * 64 + to_sq."""
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int) -> tuple[int, int]:
    """Decode index back to (from_square, to_square)."""
    return idx // 64, idx % 64


class NextMoveCNN(nn.Module):
    """CNN policy network for next move prediction."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, NUM_MOVE_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 12, 8, 8)
        Output: (B, 4096) — logits for each possible from-to move
        """
        features = self.features(x)
        return self.policy_head(features)


# ──────────────────────────────────────────────────
# Singleton loader
# ──────────────────────────────────────────────────

_model: NextMoveCNN | None = None


def get_next_move_model() -> NextMoveCNN:
    """Load the next-move prediction model (singleton)."""
    global _model
    if _model is None:
        _model = NextMoveCNN()
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
            print(f"✅ Next-move model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  No trained model at {MODEL_PATH}")
        _model.eval()
    return _model


# ──────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────

def predict_next_move(board: chess.Board, top_k: int = 5) -> list[dict]:
    """
    Predict the best next moves for the given position.
    
    Returns list of dicts with 'move' (UCI), 'score', 'explanation'.
    """
    model = get_next_move_model()
    tensor = board_to_tensor(board).unsqueeze(0)  # (1, 12, 8, 8)

    with torch.no_grad():
        logits = model(tensor)  # (1, 4096)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (4096,)

    # Get legal moves and their probabilities
    legal_moves = list(board.legal_moves)
    move_scores = []

    for move in legal_moves:
        idx = move_to_index(move)
        score = probs[idx].item()
        move_scores.append((move, score))

    # Sort by score descending
    move_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for move, score in move_scores[:top_k]:
        # Generate explanation
        explanation = _explain_predicted_move(board, move, score)
        results.append({
            "move": move.uci(),
            "score": round(score * 100, 2),
            "explanation": explanation,
        })

    return results


def _explain_predicted_move(board: chess.Board, move: chess.Move, confidence: float) -> str:
    """Generate a Romanian explanation for a predicted move."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return "Mutare sugerată din analiza jocurilor reale."

    PIECE_NAMES = {
        chess.PAWN: "Pionul", chess.KNIGHT: "Calul", chess.BISHOP: "Nebunul",
        chess.ROOK: "Turnul", chess.QUEEN: "Regina", chess.KING: "Regele",
    }

    piece_name = PIECE_NAMES.get(piece.piece_type, "Piesa")
    from_sq = chess.square_name(move.from_square).upper()
    to_sq = chess.square_name(move.to_square).upper()

    quality = "excelentă" if confidence > 0.3 else ("bună" if confidence > 0.1 else "interesantă")

    # Check special moves
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        cap_name = PIECE_NAMES.get(captured.piece_type, "piesa").lower() if captured else "piesa"
        return f"Mutare {quality}: {piece_name} capturează {cap_name} pe {to_sq} (conf: {confidence*100:.0f}%)"

    if board.is_castling(move):
        side = "scurtă" if board.is_kingside_castling(move) else "lungă"
        return f"Rocadă {side} — mutare {quality} din baza de date (conf: {confidence*100:.0f}%)"

    # Center control
    center = {chess.E4, chess.E5, chess.D4, chess.D5}
    if move.to_square in center:
        return f"{piece_name} controlează centrul pe {to_sq} — mutare {quality} (conf: {confidence*100:.0f}%)"

    # Check
    board_copy = board.copy()
    board_copy.push(move)
    if board_copy.is_check():
        return f"{piece_name} pe {to_sq} dă ȘAH! — mutare {quality} (conf: {confidence*100:.0f}%)"

    return f"{piece_name} mută pe {to_sq} — mutare {quality} din analiza a mii de partide (conf: {confidence*100:.0f}%)"


# ──────────────────────────────────────────────────
# Training Data Preparation
# ──────────────────────────────────────────────────

def prepare_training_data(csv_path: str, max_positions: int = 50000,
                          min_rating: int = 1200) -> list[tuple]:
    """
    Extract (board_state, next_move) pairs from game CSV.
    Only uses winning side's moves from decisive games.
    """
    import pandas as pd

    print(f"  📂 Pregătire date din: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total jocuri: {len(df):,}")

    # Detect columns
    col_map = {c.lower().replace(' ', '_'): c for c in df.columns}
    moves_col = col_map.get('moves') or col_map.get('pgn')
    winner_col = col_map.get('winner') or col_map.get('result')

    if not moves_col:
        print(f"  ❌ Nu am găsit coloana 'moves'. Coloane: {list(df.columns)}")
        return []

    # Filter decisive games with decent ratings
    if winner_col:
        decisive = df[df[winner_col].isin(['white', 'black', '1-0', '0-1'])]
    else:
        decisive = df

    # Filter by rating if available
    rating_col = col_map.get('white_rating') or col_map.get('whiteelo')
    if rating_col and rating_col in df.columns:
        decisive = decisive[df[rating_col] >= min_rating]

    print(f"  Jocuri decisive: {len(decisive):,}")

    data = []
    skipped = 0

    for _, row in decisive.iterrows():
        if len(data) >= max_positions:
            break

        moves_str = str(row[moves_col]).strip()
        if not moves_str or moves_str == 'nan':
            skipped += 1
            continue

        # Determine winner
        if winner_col:
            winner_val = str(row[winner_col]).strip().lower()
            if winner_val in ('white', '1-0'):
                winning_color = chess.WHITE
            elif winner_val in ('black', '0-1'):
                winning_color = chess.BLACK
            else:
                continue
        else:
            winning_color = chess.WHITE  # default

        # Parse moves (support both UCI and SAN)
        board = chess.Board()
        moves_list = moves_str.split()

        for move_str in moves_list:
            if board.is_game_over():
                break

            try:
                # Try UCI first
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    # Try SAN
                    move = board.parse_san(move_str)
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                break

            # Only learn from the winning side's moves
            if board.turn == winning_color:
                tensor = board_to_tensor(board)
                label = move_to_index(move)
                data.append((tensor, torch.tensor(label, dtype=torch.long)))

            board.push(move)

        if len(data) % 5000 == 0 and len(data) > 0:
            print(f"    {len(data):,} poziții extrase...")

    print(f"  ✅ Total: {len(data):,} poziții | Skipped: {skipped:,}")
    return data


# ──────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────

def train_model(data: list, epochs: int = 15, batch_size: int = 128, lr: float = 1e-3):
    """Train the next-move prediction model."""
    if not data:
        print("  ❌ Nicio dată de antrenament!")
        return

    os.makedirs("model", exist_ok=True)
    model = NextMoveCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Split data
    random.shuffle(data)
    val_size = int(len(data) * 0.1)
    val_data = data[:val_size]
    train_data = data[val_size:]

    print(f"\n  📊 Train: {len(train_data):,} | Val: {len(val_data):,}")
    print(f"  ⚙️  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}\n")

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        random.shuffle(train_data)
        train_loss = 0.0
        train_correct = 0
        batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            tensors = torch.stack([b[0] for b in batch])
            labels = torch.stack([b[1] for b in batch])

            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            batches += 1

        scheduler.step()
        train_acc = train_correct / max(1, len(train_data))

        # Validate
        model.eval()
        val_correct = 0
        val_top5 = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                tensors = torch.stack([b[0] for b in batch])
                labels = torch.stack([b[1] for b in batch])

                logits = model(tensors)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()

                # Top-5 accuracy
                top5 = logits.topk(5, dim=1).indices
                for j, label in enumerate(labels):
                    if label in top5[j]:
                        val_top5 += 1

        val_acc = val_correct / max(1, len(val_data))
        val_top5_acc = val_top5 / max(1, len(val_data))

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ← best ✅"

        print(
            f"  Epoch {epoch+1:2d}/{epochs} | "
            f"loss={train_loss/batches:.4f} train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f} top5={val_top5_acc:.3f}{marker}"
        )

    print(f"\n  ✅ Model salvat: {MODEL_PATH} (val_acc={best_val_acc:.3f})")


# ──────────────────────────────────────────────────
# Fallback: generate training data from random games
# ──────────────────────────────────────────────────

def generate_heuristic_data(n_positions: int = 20000) -> list[tuple]:
    """Generate training data using heuristic evaluation to pick 'good' moves with variety."""
    from chess_ai import evaluate_board

    print(f"  🎮 Generez {n_positions} poziții din jocuri cu euristică...")
    data = []
    random.seed(42)

    while len(data) < n_positions:
        board = chess.Board()
        for _ in range(80):
            if board.is_game_over() or len(data) >= n_positions:
                break

            legal = list(board.legal_moves)
            if not legal:
                break

            # Add noise and selection variety
            if random.random() < 0.15:
                # 15% exploration/randomness
                best_move = random.choice(legal)
            else:
                move_scores = []
                from chess_ai import minimax
                for move in legal:
                    board.push(move)
                    opponent_maximizing = not board.turn
                    score = minimax(board, 1, -float('inf'), float('inf'), opponent_maximizing)
                    board.pop()
                    # Add tiny random noise to score to break ties and add variety
                    score_noise = score + random.uniform(-15, 15)
                    move_scores.append((move, score_noise))
                
                # Sort moves based on score
                if board.turn == chess.WHITE:
                    move_scores.sort(key=lambda x: x[1], reverse=True)
                else:
                    move_scores.sort(key=lambda x: x[1])
                
                candidates = move_scores[:min(3, len(move_scores))]
                best_move = random.choice(candidates)[0]

            if best_move:
                tensor = board_to_tensor(board)
                label = move_to_index(best_move)
                data.append((tensor, torch.tensor(label, dtype=torch.long)))
                board.push(best_move)

        if len(data) % 5000 == 0 and len(data) > 0:
            print(f"    {len(data):,}/{n_positions} poziții generate...")

    return data[:n_positions]


# ──────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Next-Move Predictor")
    parser.add_argument('--csv', type=str, default=None,
                        help="Path to games CSV")
    parser.add_argument('--samples', type=int, default=30000,
                        help="Max positions to extract")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-rating', type=int, default=1200)
    args = parser.parse_args()

    csv_path = args.csv or os.path.join("data", "chess_games.csv")

    if os.path.exists(csv_path):
        data = prepare_training_data(csv_path, args.samples, args.min_rating)
    else:
        print(f"  ⚠️  CSV nu există: {csv_path}")
        print("  🔄 Generez date din jocuri cu euristică...")
        data = generate_heuristic_data(min(args.samples, 20000))

    train_model(data, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
