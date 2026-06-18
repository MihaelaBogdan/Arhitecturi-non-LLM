"""
Board Recognition CNN — Recunoaște poziția pieselor dintr-o imagine de tablă de șah.

Input:  Imagine RGB (400×400) sau orice dimensiune (se redimensionează)
Output: FEN string al poziției recunoscute

Architecture:
  - ResNet-like CNN cu 64 output heads (câte unul per pătrățel)
  - Fiecare head clasifică în 13 clase: empty + 12 piese (6 albe + 6 negre)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

MODEL_PATH = os.path.join("model", "board_recognition.pth")

# Mapping: index → piece character in FEN
# 0 = empty, 1-6 = white (P,N,B,R,Q,K), 7-12 = black (p,n,b,r,q,k)
IDX_TO_FEN = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
FEN_TO_IDX = {c: i for i, c in enumerate(IDX_TO_FEN)}

NUM_CLASSES = 13  # empty + 12 pieces
BOARD_SIZE = 8
INPUT_SIZE = 256  # Resize images to 256x256 for efficiency and MPS compatibility


class ResBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class BoardRecognitionCNN(nn.Module):
    """CNN that recognizes chess pieces from a board image."""

    def __init__(self):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 100x100
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 50x50
            nn.BatchNorm2d(64),
            nn.ReLU(),

            ResBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 25x25
            nn.BatchNorm2d(128),
            nn.ReLU(),

            ResBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 25x25
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((8, 8)),  # → 256 × 8 × 8
        )

        # Per-square classifier: each of 64 squares gets its own classification
        # We reshape 256×8×8 → 64 squares × 256 features, then classify each
        self.square_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3, 200, 200)
        Output: (B, 64, 13) — logits for each square
        """
        features = self.features(x)  # (B, 256, 8, 8)
        B, C, H, W = features.shape

        # Reshape to per-square features: (B, 8, 8, 256) → (B, 64, 256)
        features = features.permute(0, 2, 3, 1).reshape(B, 64, C)

        # Classify each square
        logits = self.square_classifier(features)  # (B, 64, 13)
        return logits


# ──────────────────────────────────────────────────
# FEN parsing utilities
# ──────────────────────────────────────────────────

def fen_to_labels(fen_placement: str) -> np.ndarray:
    """Convert FEN placement string to 64 integer labels (0-12).
    
    FEN ranks go from rank 8 (top) to rank 1 (bottom).
    Our label array: index 0 = a8, index 1 = b8, ..., index 63 = h1.
    """
    labels = np.zeros(64, dtype=np.int64)
    ranks = fen_placement.split('/')
    idx = 0
    for rank_str in ranks:
        for ch in rank_str:
            if ch.isdigit():
                idx += int(ch)  # empty squares
            else:
                labels[idx] = FEN_TO_IDX.get(ch, 0)
                idx += 1
    return labels


def labels_to_fen(labels: np.ndarray) -> str:
    """Convert 64 integer labels back to FEN placement string."""
    fen_rows = []
    for rank in range(8):
        row = ""
        empty_count = 0
        for file in range(8):
            idx = rank * 8 + file
            piece_idx = labels[idx]
            if piece_idx == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    row += str(empty_count)
                    empty_count = 0
                row += IDX_TO_FEN[piece_idx]
        if empty_count > 0:
            row += str(empty_count)
        fen_rows.append(row)
    return "/".join(fen_rows)


def filename_to_fen(filename: str) -> str:
    """Convert a filename like 'r1bk3r-pp2pppp-...' to FEN placement."""
    name = os.path.splitext(filename)[0]
    return name.replace('-', '/')


# ──────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────

_recognition_model: BoardRecognitionCNN | None = None


def get_recognition_model() -> BoardRecognitionCNN:
    """Load the board recognition model (singleton)."""
    global _recognition_model
    if _recognition_model is None:
        _recognition_model = BoardRecognitionCNN()
        if os.path.exists(MODEL_PATH):
            _recognition_model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
            print(f"✅ Board recognition model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  No trained model at {MODEL_PATH}, using random weights")
        _recognition_model.eval()
    return _recognition_model


def preprocess_image(image_input) -> torch.Tensor:
    """
    Preprocess an image for the CNN.
    
    Args:
        image_input: PIL Image, numpy array, or file path string
    
    Returns:
        Tensor of shape (1, 3, INPUT_SIZE, INPUT_SIZE)
    """
    from PIL import Image

    if isinstance(image_input, str):
        # File path
        if image_input.lower().endswith('.svg'):
            # Convert SVG to PNG via cairosvg if available
            try:
                import cairosvg
                import io
                png_data = cairosvg.svg2png(url=image_input, output_width=INPUT_SIZE, output_height=INPUT_SIZE)
                img = Image.open(io.BytesIO(png_data)).convert('RGB')
            except ImportError:
                # Fallback: create a blank image (can't parse SVG without cairo)
                img = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (200, 200, 200))
        else:
            img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    # Resize
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

    # To tensor: (H, W, 3) → (3, H, W), normalize to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    # Normalize with ImageNet-like stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)  # (1, 3, H, W)


def recognize_board(image_input) -> str:
    """
    Recognize a chess position from an image.
    
    Args:
        image_input: PIL Image, numpy array, or file path
    
    Returns:
        Full FEN string (with default side-to-move etc.)
    """
    model = get_recognition_model()
    tensor = preprocess_image(image_input)

    with torch.no_grad():
        logits = model(tensor)  # (1, 64, 13)
        predictions = logits.argmax(dim=2).squeeze(0).numpy()  # (64,)

    fen_placement = labels_to_fen(predictions)
    # Add default FEN fields: white to move, all castling, no en passant
    full_fen = f"{fen_placement} w KQkq - 0 1"

    return full_fen


def recognize_board_with_confidence(image_input) -> dict:
    """
    Recognize board with confidence scores per square.
    
    Returns dict with 'fen', 'confidence', 'per_square' details.
    """
    model = get_recognition_model()
    tensor = preprocess_image(image_input)

    with torch.no_grad():
        logits = model(tensor)  # (1, 64, 13)
        probs = F.softmax(logits, dim=2).squeeze(0)  # (64, 13)
        predictions = probs.argmax(dim=1).numpy()  # (64,)
        confidences = probs.max(dim=1).values.numpy()  # (64,)

    fen_placement = labels_to_fen(predictions)
    full_fen = f"{fen_placement} w KQkq - 0 1"

    # Per-square details
    squares = []
    for i in range(64):
        rank = 7 - (i // 8)
        file = i % 8
        sq_name = chr(ord('a') + file) + str(rank + 1)
        squares.append({
            "square": sq_name,
            "piece": IDX_TO_FEN[predictions[i]],
            "confidence": float(confidences[i]),
        })

    return {
        "fen": full_fen,
        "avg_confidence": float(confidences.mean()),
        "min_confidence": float(confidences.min()),
        "squares": squares,
    }
