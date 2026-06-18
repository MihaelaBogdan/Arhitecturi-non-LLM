"""
Train the Board Recognition CNN on chess position images.

The images have FEN-encoded filenames (dashes instead of slashes).
Example filename: "r1bk3r-pp2pppp-2n2n2-2b1p3-2B1P3-5N2-PPPP1PPP-RNBQK2R.jpeg"

Usage:
    python train_board_recognition.py                          # Default: use data/chess_images/
    python train_board_recognition.py --epochs 20 --batch 64   # Custom training
    python train_board_recognition.py --generate 5000          # Generate synthetic images first
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from board_recognition_cnn import (
    BoardRecognitionCNN, MODEL_PATH, INPUT_SIZE,
    fen_to_labels, filename_to_fen, labels_to_fen
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "chess_images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")


# ──────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────

class ChessBoardDataset(Dataset):
    """Dataset of chess board images with FEN labels."""

    def __init__(self, image_dir: str, augment: bool = False):
        self.image_dir = image_dir
        self.augment = augment
        self.samples = []

        if not os.path.isdir(image_dir):
            print(f"  ⚠️  Director inexistent: {image_dir}")
            return

        for fname in os.listdir(image_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.jpeg', '.jpg', '.png', '.svg'):
                continue
            # Extract FEN from filename
            fen_placement = filename_to_fen(fname)
            try:
                labels = fen_to_labels(fen_placement)
                self.samples.append((os.path.join(image_dir, fname), labels))
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]

        # Load image
        try:
            if img_path.lower().endswith('.svg'):
                # SVG handling
                try:
                    import cairosvg
                    import io
                    png_data = cairosvg.svg2png(url=img_path, output_width=INPUT_SIZE, output_height=INPUT_SIZE)
                    img = Image.open(io.BytesIO(png_data)).convert('RGB')
                except ImportError:
                    img = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (200, 200, 200))
            else:
                img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (128, 128, 128))

        if img.size != (INPUT_SIZE, INPUT_SIZE):
            img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

        # To tensor
        arr = np.array(img, dtype=np.float32) / 255.0

        # Data augmentation
        if self.augment and random.random() > 0.5:
            # Random brightness
            factor = random.uniform(0.8, 1.2)
            arr = np.clip(arr * factor, 0, 1)

        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        labels_tensor = torch.from_numpy(labels).long()
        return tensor, labels_tensor


# ──────────────────────────────────────────────────
# Synthetic image generation
# ──────────────────────────────────────────────────

def generate_synthetic_images(n: int = 5000, val_ratio: float = 0.2):
    """Generate synthetic board images using python-chess SVG rendering."""
    import chess
    import chess.svg

    print(f"\n🎨 Generare {n} imagini sintetice de table de șah...")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    existing = len(os.listdir(TRAIN_DIR)) + len(os.listdir(VAL_DIR))
    if existing >= n * 0.8:
        print(f"  ✅ Deja {existing} imagini existente. Skip.")
        return

    val_count = int(n * val_ratio)
    pieces_pool = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    # Board styles (different colors)
    board_colors = [
        {"square light": "#F0D9B5", "square dark": "#B58863"},   # classic
        {"square light": "#FFFFDD", "square dark": "#86A666"},   # green
        {"square light": "#DEE3E6", "square dark": "#8CA2AD"},   # blue
        {"square light": "#F0E6D3", "square dark": "#9E7863"},   # brown
        {"square light": "#E8E8E8", "square dark": "#7B8794"},   # gray
    ]

    random.seed(42)

    for i in range(n):
        board = chess.Board.empty()

        # Place 2 kings
        wk_sq = random.randint(0, 63)
        bk_sq = random.randint(0, 63)
        while bk_sq == wk_sq or chess.square_distance(wk_sq, bk_sq) < 2:
            bk_sq = random.randint(0, 63)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        # Place 3-13 random pieces
        num_extra = random.randint(3, 13)
        occupied = {wk_sq, bk_sq}
        for _ in range(num_extra):
            sq = random.randint(0, 63)
            while sq in occupied:
                sq = random.randint(0, 63)
            occupied.add(sq)
            piece_type = random.choices(
                pieces_pool,
                weights=[30, 20, 20, 20, 10],  # Pawn most common
                k=1
            )[0]
            color = random.choice([chess.WHITE, chess.BLACK])
            # No pawns on rank 1 or 8
            rank = chess.square_rank(sq)
            if piece_type == chess.PAWN and rank in (0, 7):
                piece_type = random.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK])
            board.set_piece_at(sq, chess.Piece(piece_type, color))

        # Generate FEN filename
        fen_full = board.fen()
        fen_placement = fen_full.split(' ')[0]
        fen_filename = fen_placement.replace('/', '-')

        # Random board style
        colors = random.choice(board_colors)

        # Generate SVG
        svg_data = chess.svg.board(
            board, size=256, coordinates=False,
            colors=colors
        )

        target_dir = VAL_DIR if i < val_count else TRAIN_DIR
        
        # Save as PNG
        try:
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), output_width=256, output_height=256)
            png_path = os.path.join(target_dir, f"{fen_filename}.png")
            with open(png_path, 'wb') as f:
                f.write(png_data)
        except ImportError:
            # Fallback to SVG if cairosvg not installed
            svg_path = os.path.join(target_dir, f"{fen_filename}.svg")
            with open(svg_path, 'w') as f:
                f.write(svg_data)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} imagini generate...")

    train_count = len(os.listdir(TRAIN_DIR))
    val_count_actual = len(os.listdir(VAL_DIR))
    print(f"  ✅ Generate: {train_count} train + {val_count_actual} val")


# ──────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────

def train(epochs: int = 20, batch_size: int = 32, lr: float = 1e-3):
    """Train the board recognition CNN."""
    print(f"\n{'='*60}")
    print(f"  🧠 Antrenament Board Recognition CNN")
    print(f"{'='*60}")

    # Create datasets
    train_dataset = ChessBoardDataset(TRAIN_DIR, augment=True)
    val_dataset = ChessBoardDataset(VAL_DIR, augment=False)

    if len(train_dataset) == 0:
        print("  ❌ Nicio imagine de antrenament! Rulează mai întâi:")
        print("     python train_board_recognition.py --generate 5000")
        return

    print(f"  📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  ⚙️  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"  🚀 Using device: {device}")

    # Model
    model = BoardRecognitionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("model", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)  # (B, 64, 13)
            B = logits.shape[0]

            # Reshape for cross-entropy: (B*64, 13) vs (B*64,)
            logits_flat = logits.reshape(B * 64, -1)
            labels_flat = labels.reshape(B * 64)

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = logits_flat.argmax(dim=1)
            train_correct += (predictions == labels_flat).sum().item()
            train_total += labels_flat.numel()

        scheduler.step()
        train_acc = train_correct / max(1, train_total)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_board_correct = 0  # boards with ALL 64 squares correct
        val_board_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                B = logits.shape[0]

                logits_flat = logits.reshape(B * 64, -1)
                labels_flat = labels.reshape(B * 64)

                val_loss += criterion(logits_flat, labels_flat).item()
                predictions = logits_flat.argmax(dim=1)
                val_correct += (predictions == labels_flat).sum().item()
                val_total += labels_flat.numel()

                # Per-board accuracy
                pred_boards = predictions.reshape(B, 64)
                label_boards = labels_flat.reshape(B, 64)
                for b in range(B):
                    if (pred_boards[b] == label_boards[b]).all():
                        val_board_correct += 1
                    val_board_total += 1

        val_acc = val_correct / max(1, val_total)
        board_acc = val_board_correct / max(1, val_board_total)

        # Save best
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save state dict (move model to cpu first to ensure portability, then back to device)
            model.to("cpu")
            torch.save(model.state_dict(), MODEL_PATH)
            model.to(device)
            marker = " ← best ✅"

        n_batches = len(train_loader)
        print(
            f"  Epoch {epoch+1:2d}/{epochs} | "
            f"train_loss={train_loss/n_batches:.4f} train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f} board_acc={board_acc:.3f}{marker}"
        )

    print(f"\n  ✅ Best model saved: {MODEL_PATH} (val_acc={best_val_acc:.3f})")


# ──────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Board Recognition CNN")
    parser.add_argument('--generate', type=int, default=0,
                        help="Generate N synthetic images before training")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--samples', type=int, default=0,
                        help="Alias for --generate")
    args = parser.parse_args()

    gen_count = args.generate or args.samples
    if gen_count > 0:
        generate_synthetic_images(gen_count)

    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
