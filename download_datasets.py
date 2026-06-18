"""
Download and prepare chess datasets for the AI platform.

Datasets:
  1. Chess Positions (Kaggle: koryakinp/chess-positions) — board images with FEN filenames
  2. Chess Games (Kaggle: datasnaek/chess) — 20k finished games with PGN
  3. Chess Openings — ECO opening database (generated locally)

Usage:
    python download_datasets.py                     # Download all
    python download_datasets.py --images-only       # Just board images
    python download_datasets.py --games-only        # Just game CSV
    python download_datasets.py --openings-only     # Just openings
    python download_datasets.py --check             # Verify datasets
    python download_datasets.py --max-images 5000   # Limit images
"""

import os
import sys
import random
import shutil
import argparse
import zipfile
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "chess_images")
TRAIN_DIR = os.path.join(IMAGES_DIR, "train")
VAL_DIR = os.path.join(IMAGES_DIR, "val")


# ──────────────────────────────────────────────────
# 1. Board images from Kaggle
# ──────────────────────────────────────────────────

def download_board_images(max_images: int = 5000, val_ratio: float = 0.2):
    """Download chess-positions dataset and extract a subset of images."""
    print(f"\n{'='*60}")
    print(f"  📷 Descărcare imagini table de șah (max {max_images})")
    print(f"{'='*60}")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # Check if already downloaded
    existing = len(os.listdir(TRAIN_DIR)) + len(os.listdir(VAL_DIR))
    if existing >= max_images * 0.8:
        print(f"  ✅ Deja descărcate {existing} imagini. Skip.")
        return True

    try:
        import opendatasets as od
        print("  📥 Descărcare de pe Kaggle (necesită API key)...")
        print("  💡 Dacă ți se cere, introdu Kaggle username și key de pe:")
        print("     https://www.kaggle.com/settings → API → Create New Token")
        od.download("https://www.kaggle.com/datasets/koryakinp/chess-positions",
                     data_dir=DATA_DIR)
    except Exception as e:
        print(f"  ⚠️  Eroare la descărcare Kaggle: {e}")
        print("  🔄 Încerc metoda alternativă...")
        return _download_images_fallback(max_images, val_ratio)

    # Find extracted folder
    kaggle_dir = os.path.join(DATA_DIR, "chess-positions")
    train_source = os.path.join(kaggle_dir, "train")
    test_source = os.path.join(kaggle_dir, "test")

    if not os.path.isdir(train_source):
        # Try to find images in any subfolder
        for root, dirs, files in os.walk(kaggle_dir):
            jpeg_files = [f for f in files if f.endswith(('.jpeg', '.jpg', '.png'))]
            if jpeg_files:
                train_source = root
                break

    if not os.path.isdir(train_source):
        print(f"  ❌ Nu am găsit imaginile în {kaggle_dir}")
        return False

    # Get all image files
    all_images = [f for f in os.listdir(train_source)
                  if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    # Also check test directory
    if os.path.isdir(test_source):
        test_images = [f for f in os.listdir(test_source)
                       if f.endswith(('.jpeg', '.jpg', '.png'))]
        all_images_with_source = [(f, train_source) for f in all_images]
        all_images_with_source += [(f, test_source) for f in test_images]
    else:
        all_images_with_source = [(f, train_source) for f in all_images]

    print(f"  📂 Găsite {len(all_images_with_source)} imagini totale")

    # Random sample
    random.seed(42)
    if len(all_images_with_source) > max_images:
        selected = random.sample(all_images_with_source, max_images)
    else:
        selected = all_images_with_source

    # Split train/val
    random.shuffle(selected)
    val_count = int(len(selected) * val_ratio)
    val_set = selected[:val_count]
    train_set = selected[val_count:]

    print(f"  📦 Copiere: {len(train_set)} train + {len(val_set)} val")

    for fname, source_dir in train_set:
        shutil.copy2(os.path.join(source_dir, fname), os.path.join(TRAIN_DIR, fname))
    for fname, source_dir in val_set:
        shutil.copy2(os.path.join(source_dir, fname), os.path.join(VAL_DIR, fname))

    print(f"  ✅ Imagini pregătite: {len(train_set)} train, {len(val_set)} val")
    return True


def _download_images_fallback(max_images: int, val_ratio: float):
    """Fallback: generate synthetic board images using python-chess + SVG rendering."""
    print("  🎨 Generez imagini sintetice de table de șah...")
    
    try:
        import chess
        import chess.svg
    except ImportError:
        print("  ❌ Lipsă chess library. pip install chess")
        return False

    # Try to use cairosvg for SVG→PNG conversion
    try:
        import cairosvg
        has_cairo = True
    except ImportError:
        has_cairo = False

    # Try Pillow SVG
    try:
        from PIL import Image
        import io
        has_pillow = True
    except ImportError:
        has_pillow = False

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    generated = 0
    val_count = int(max_images * val_ratio)
    pieces_pool = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    
    random.seed(42)
    
    for i in range(max_images):
        board = chess.Board.empty()
        
        # Place 2 kings
        wk_sq = random.randint(0, 63)
        bk_sq = random.randint(0, 63)
        while bk_sq == wk_sq:
            bk_sq = random.randint(0, 63)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        
        # Place 3-13 random pieces
        num_pieces = random.randint(3, 13)
        occupied = {wk_sq, bk_sq}
        for _ in range(num_pieces):
            sq = random.randint(0, 63)
            while sq in occupied:
                sq = random.randint(0, 63)
            occupied.add(sq)
            piece_type = random.choice(pieces_pool)
            color = random.choice([chess.WHITE, chess.BLACK])
            # No pawns on rank 1 or 8
            rank = chess.square_rank(sq)
            if piece_type == chess.PAWN and rank in (0, 7):
                piece_type = random.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK])
            board.set_piece_at(sq, chess.Piece(piece_type, color))
        
        # Generate FEN (only piece placement part)
        fen_full = board.fen()
        fen_placement = fen_full.split(' ')[0]
        fen_filename = fen_placement.replace('/', '-')
        
        # Save as SVG (lightweight, no extra deps needed)
        svg_data = chess.svg.board(board, size=400)
        
        target_dir = VAL_DIR if i < val_count else TRAIN_DIR
        
        if has_cairo:
            png_path = os.path.join(target_dir, f"{fen_filename}.png")
            cairosvg.svg2png(bytestring=svg_data.encode(), write_to=png_path,
                           output_width=400, output_height=400)
        else:
            # Save as SVG
            svg_path = os.path.join(target_dir, f"{fen_filename}.svg")
            with open(svg_path, 'w') as f:
                f.write(svg_data)
        
        generated += 1
        if generated % 500 == 0:
            print(f"    {generated}/{max_images} imagini generate...")

    print(f"  ✅ Generate {generated} imagini sintetice")
    return True


# ──────────────────────────────────────────────────
# 2. Chess Games Dataset
# ──────────────────────────────────────────────────

def download_chess_games():
    """Download the chess.com games dataset from Kaggle."""
    print(f"\n{'='*60}")
    print(f"  ♟️  Descărcare jocuri de șah terminate")
    print(f"{'='*60}")

    csv_path = os.path.join(DATA_DIR, "chess_games.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  ✅ Deja descărcat: {len(df):,} jocuri. Skip.")
        return True

    try:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/datasnaek/chess",
                     data_dir=DATA_DIR)
    except Exception as e:
        print(f"  ⚠️  Eroare Kaggle: {e}")
        print("  🔄 Generez jocuri sintetice...")
        return _generate_synthetic_games(csv_path)

    # Find the CSV
    kaggle_dir = os.path.join(DATA_DIR, "chess")
    for root, dirs, files in os.walk(kaggle_dir):
        for f in files:
            if f.endswith('.csv'):
                src = os.path.join(root, f)
                shutil.copy2(src, csv_path)
                print(f"  ✅ Copiat {f} → {csv_path}")
                return True

    print("  ❌ Nu am găsit CSV-ul cu jocuri")
    return False


def _generate_synthetic_games(csv_path: str, n_games: int = 2000):
    """Generate synthetic game data by playing games with heuristic evaluation + exploration."""
    import chess
    import pandas as pd
    from chess_ai import minimax

    print(f"  🎮 Generez {n_games} jocuri sintetice cu evaluare euristică...")
    
    games = []
    random.seed(42)

    for i in range(n_games):
        board = chess.Board()
        moves = []
        
        for _ in range(80):  # max 80 moves per game
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            
            # Epsilon-greedy exploration: 15% random move to ensure game variety
            if random.random() < 0.15:
                move = random.choice(legal)
            else:
                # 85% chance: evaluate all legal moves with minimax (1-ply opponent reply)
                move_scores = []
                for m in legal:
                    board.push(m)
                    # Next player is opponent (so maximizing_player = False if turn is White, True if turn is Black)
                    opponent_maximizing = not board.turn
                    score = minimax(board, 1, -float('inf'), float('inf'), opponent_maximizing)
                    board.pop()
                    # Add tiny random noise to avoid deterministic choices
                    score_noise = score + random.uniform(-15, 15)
                    move_scores.append((m, score_noise))
                
                # Sort by score (descending for White, ascending for Black)
                if board.turn == chess.WHITE:
                    move_scores.sort(key=lambda x: x[1], reverse=True)
                else:
                    move_scores.sort(key=lambda x: x[1])
                
                # Select randomly among the top 3 moves to add strategic variety
                candidates = move_scores[:min(3, len(move_scores))]
                move = random.choice(candidates)[0]
                
            moves.append(move.uci())
            board.push(move)

        result = board.result() if board.is_game_over() else "1/2-1/2"
        pgn_moves = " ".join(moves)
        
        games.append({
            "rated": True,
            "turns": len(moves),
            "winner": "white" if result == "1-0" else ("black" if result == "0-1" else "draw"),
            "white_rating": random.randint(1200, 2200),
            "black_rating": random.randint(1200, 2200),
            "moves": pgn_moves,
            "opening_name": "Generated",
            "opening_eco": "A00",
        })

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_games} jocuri generate...")

    df = pd.DataFrame(games)
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Generate {len(df)} jocuri → {csv_path}")
    return True


# ──────────────────────────────────────────────────
# 3. Opening Database
# ──────────────────────────────────────────────────

def create_openings_database():
    """Create a comprehensive ECO openings database."""
    print(f"\n{'='*60}")
    print(f"  📖 Creare bază de date deschideri ECO")
    print(f"{'='*60}")

    csv_path = os.path.join(DATA_DIR, "openings.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  ✅ Deja există: {len(df)} deschideri. Skip.")
        return True

    import chess
    import pandas as pd

    # Comprehensive openings database with ECO codes, moves, descriptions
    openings = [
        # A: Flank Openings
        {"eco": "A00", "name": "Deschiderea Anderssen", "name_en": "Anderssen Opening",
         "moves": "a3", "description": "O deschidere rară, pasivă. Albul cedează inițiativa.",
         "strategy": "Flexibilitate, evitarea teoriei principale.", "difficulty": "ușor"},
        {"eco": "A00", "name": "Deschiderea Grob", "name_en": "Grob Opening",
         "moves": "g4", "description": "Deschidere agresivă dar riscantă. Slăbește flancul regelui.",
         "strategy": "Surpriză, atac pe flancul regelui.", "difficulty": "avansat"},
        {"eco": "A01", "name": "Deschiderea Larsen", "name_en": "Larsen Opening",
         "moves": "b3", "description": "Fianchetto pe flancul damei. Controlează diagonala lungă.",
         "strategy": "Presiune pe diagonala a1-h8, joc pozițional.", "difficulty": "intermediar"},
        {"eco": "A02", "name": "Deschiderea Bird", "name_en": "Bird Opening",
         "moves": "f4", "description": "Control agresiv pe coloana f. Similar cu Deschiderea Olandeză inversată.",
         "strategy": "Atac pe flancul regelui, control pe e5.", "difficulty": "intermediar"},
        {"eco": "A04", "name": "Deschiderea Réti", "name_en": "Reti Opening",
         "moves": "Nf3", "description": "Deschidere hipermodernă. Controlează centrul de la distanță.",
         "strategy": "Fianchetto, control indirect al centrului.", "difficulty": "intermediar"},
        {"eco": "A05", "name": "Réti - Varianta Indiană Regelui", "name_en": "Reti KIA",
         "moves": "Nf3 Nf6", "description": "Setup flexibil. Poate transpune în multe structuri.",
         "strategy": "King's Indian Attack, presiune pe centru.", "difficulty": "intermediar"},
        {"eco": "A10", "name": "Deschiderea Engleză", "name_en": "English Opening",
         "moves": "c4", "description": "A treia cea mai populară deschidere. Control pe d5.",
         "strategy": "Joc pozițional, control pe câmpurile albe.", "difficulty": "intermediar"},
        {"eco": "A13", "name": "Engleză - Varianta Agincourt", "name_en": "English Agincourt",
         "moves": "c4 e6", "description": "Negrul permite Albului flexibilitate în centru.",
         "strategy": "Structuri Nimzo/Queen's Indian.", "difficulty": "intermediar"},
        {"eco": "A20", "name": "Engleză cu 1...e5", "name_en": "English Reversed Sicilian",
         "moves": "c4 e5", "description": "Siciliana inversată. Negrul ocupă centrul.",
         "strategy": "Albul joacă o Siciliană cu tempo în plus.", "difficulty": "intermediar"},
        {"eco": "A40", "name": "Apărarea Indiană", "name_en": "Indian Defense",
         "moves": "d4 Nf6", "description": "Baza tuturor sistemelor indiene. Extrem de popular.",
         "strategy": "Hipermodernism, contraatac pe centru.", "difficulty": "intermediar"},

        # B: Semi-Open Games (1.e4, not 1...e5)
        {"eco": "B00", "name": "Apărarea Nimzowitsch", "name_en": "Nimzowitsch Defense",
         "moves": "e4 Nc6", "description": "Apărare hipermodernă neobișnuită.",
         "strategy": "Presiune pe e5, dezvoltare rapidă.", "difficulty": "avansat"},
        {"eco": "B01", "name": "Apărarea Scandinavă", "name_en": "Scandinavian Defense",
         "moves": "e4 d5", "description": "Negrul atacă imediat centrul. Simplă dar solidă.",
         "strategy": "Schimb de pioni în centru, dezvoltare rapidă.", "difficulty": "ușor"},
        {"eco": "B02", "name": "Apărarea Alekhine", "name_en": "Alekhine Defense",
         "moves": "e4 Nf6", "description": "Provoacă pionii albi să avanseze, apoi îi atacă.",
         "strategy": "Hipermodernism, contraatac pe centrul extins.", "difficulty": "avansat"},
        {"eco": "B06", "name": "Apărarea Modernă", "name_en": "Modern Defense",
         "moves": "e4 g6", "description": "Fianchetto rapid. Presiune pe centru de la distanță.",
         "strategy": "Flexibilitate, contraatac pe centru.", "difficulty": "intermediar"},
        {"eco": "B07", "name": "Apărarea Pirc", "name_en": "Pirc Defense",
         "moves": "e4 d6 d4 Nf6", "description": "Hipermodernă. Permite Albului centru mare, apoi îl atacă.",
         "strategy": "Contraatac din fianchetto, joc pe flanc.", "difficulty": "intermediar"},
        {"eco": "B10", "name": "Apărarea Caro-Kann", "name_en": "Caro-Kann Defense",
         "moves": "e4 c6", "description": "Foarte solidă. Pregătește d5 cu suport.",
         "strategy": "Structură solidă, nebun bun pe câmpuri albe.", "difficulty": "intermediar"},
        {"eco": "B12", "name": "Caro-Kann - Varianta Advance", "name_en": "Caro-Kann Advance",
         "moves": "e4 c6 d4 d5 e5", "description": "Albul fixează centrul. Joc pe flancuri.",
         "strategy": "Spațiu, atac pe flancul regelui.", "difficulty": "intermediar"},
        {"eco": "B13", "name": "Caro-Kann - Varianta Schimb", "name_en": "Caro-Kann Exchange",
         "moves": "e4 c6 d4 d5 exd5 cxd5", "description": "Structură simetrică, joc pozițional.",
         "strategy": "Joc liniștit, pregătire endgame.", "difficulty": "ușor"},
        {"eco": "B15", "name": "Caro-Kann - Linia Principală", "name_en": "Caro-Kann Main Line",
         "moves": "e4 c6 d4 d5 Nc3 dxe4 Nxe4", "description": "Teoria principală. Joc echilibrat.",
         "strategy": "Dezvoltare armonioasă, nebunul pe câmpuri deschise.", "difficulty": "intermediar"},
        {"eco": "B20", "name": "Apărarea Siciliană", "name_en": "Sicilian Defense",
         "moves": "e4 c5", "description": "Cea mai populară apărare la 1.e4. Asimetrică și combativă.",
         "strategy": "Contraatac pe flancul damei, joc tactic.", "difficulty": "intermediar"},
        {"eco": "B21", "name": "Siciliana - Gambitul Smith-Morra", "name_en": "Smith-Morra Gambit",
         "moves": "e4 c5 d4 cxd4 c3", "description": "Gambitul de pion pentru dezvoltare rapidă.",
         "strategy": "Inițiativă, dezvoltare rapidă, atac.", "difficulty": "intermediar"},
        {"eco": "B22", "name": "Siciliana - Alapin", "name_en": "Sicilian Alapin",
         "moves": "e4 c5 c3", "description": "Albul pregătește d4. Evită teoria greoaie.",
         "strategy": "Centru stabil, evitarea Open Sicilian.", "difficulty": "ușor"},
        {"eco": "B23", "name": "Siciliana Închisă", "name_en": "Closed Sicilian",
         "moves": "e4 c5 Nc3", "description": "Albul evită deschiderea centrului.",
         "strategy": "Grand Prix Attack, fianchetto.", "difficulty": "intermediar"},
        {"eco": "B27", "name": "Siciliana - Varianta Accelerated Dragon", "name_en": "Accelerated Dragon",
         "moves": "e4 c5 Nf3 g6", "description": "Dragon fără d6. Mai flexibil.",
         "strategy": "Presiune pe diagonala h8-a1, contraatac.", "difficulty": "avansat"},
        {"eco": "B30", "name": "Siciliana - Varianta Nyezhmetdinov-Rossolimo", "name_en": "Rossolimo Sicilian",
         "moves": "e4 c5 Nf3 Nc6 Bb5", "description": "Albul evită Open Sicilian. Foarte popular.",
         "strategy": "Structură solidă, schimb de nebun.", "difficulty": "intermediar"},
        {"eco": "B33", "name": "Siciliana - Sveshnikov", "name_en": "Sicilian Sveshnikov",
         "moves": "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5",
         "description": "Structură dinamică cu gaură pe d5.",
         "strategy": "Activitate maximă, compensație pentru d5.", "difficulty": "avansat"},
        {"eco": "B35", "name": "Siciliana Dragon", "name_en": "Sicilian Dragon",
         "moves": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6",
         "description": "Una dintre cele mai ascuțite variante. Atac și contraatac.",
         "strategy": "Nebun fianchetto devastator, atac pe flancul damei.", "difficulty": "avansat"},
        {"eco": "B40", "name": "Siciliana - Varianta Kan", "name_en": "Sicilian Kan",
         "moves": "e4 c5 Nf3 e6", "description": "Flexibilă. Poate transpune în multe variante.",
         "strategy": "Scheuler-Lowenthal, structuri Hedgehog.", "difficulty": "intermediar"},
        {"eco": "B50", "name": "Siciliana - Linia Principală", "name_en": "Open Sicilian",
         "moves": "e4 c5 Nf3 d6 d4 cxd4 Nxd4",
         "description": "Cea mai critică linie. Teoria merge foarte adânc.",
         "strategy": "Joc complex, ambii jucători au șanse de atac.", "difficulty": "avansat"},
        {"eco": "B60", "name": "Siciliana - Varianta Richter-Rauzer", "name_en": "Richter-Rauzer",
         "moves": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5",
         "description": "Atac clasic pe nebunul Negru.",
         "strategy": "Presiune pe f6, rocadă pe flancuri opuse.", "difficulty": "avansat"},
        {"eco": "B80", "name": "Siciliana Scheveningen", "name_en": "Sicilian Scheveningen",
         "moves": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6",
         "description": "Structura e6+d6. Foarte solidă dar pasivă.",
         "strategy": "Hedgehog, English Attack.", "difficulty": "avansat"},
        {"eco": "B90", "name": "Siciliana Najdorf", "name_en": "Sicilian Najdorf",
         "moves": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",
         "description": "Favorizata campionilor mondiali (Fischer, Kasparov).",
         "strategy": "Maximă flexibilitate, joc tactic intens.", "difficulty": "avansat"},

        # C: Open Games (1.e4 e5)
        {"eco": "C00", "name": "Apărarea Franceză", "name_en": "French Defense",
         "moves": "e4 e6", "description": "Solidă, cu structură de pioni asimetrică.",
         "strategy": "Contraatac pe centru cu d5, joc pe flancul damei.", "difficulty": "intermediar"},
        {"eco": "C01", "name": "Franceză - Varianta Schimb", "name_en": "French Exchange",
         "moves": "e4 e6 d4 d5 exd5 exd5", "description": "Structură simetrică, egalitate rapidă.",
         "strategy": "Joc simplu, pregătire endgame.", "difficulty": "ușor"},
        {"eco": "C02", "name": "Franceză - Varianta Advance", "name_en": "French Advance",
         "moves": "e4 e6 d4 d5 e5", "description": "Albul fixează centrul. Joc pe flancuri.",
         "strategy": "Spațiu, atac pe flancul regelui. Negrul atacă baza c2-d4.", "difficulty": "intermediar"},
        {"eco": "C03", "name": "Franceză - Varianta Tarrasch", "name_en": "French Tarrasch",
         "moves": "e4 e6 d4 d5 Nd2", "description": "Evită blocarea nebunului pe c1.",
         "strategy": "Flexibilitate, dezvoltare naturală.", "difficulty": "intermediar"},
        {"eco": "C10", "name": "Franceză - Varianta Rubinstein", "name_en": "French Rubinstein",
         "moves": "e4 e6 d4 d5 Nc3 dxe4 Nxe4", "description": "Negrul rezolvă tensiunea din centru.",
         "strategy": "Dezvoltare liberă, dar Negru e ușor pasiv.", "difficulty": "intermediar"},
        {"eco": "C11", "name": "Franceză - Linia Clasică", "name_en": "French Classical",
         "moves": "e4 e6 d4 d5 Nc3 Nf6", "description": "Linia principală. Joc bogat tactic.",
         "strategy": "Negrul presează pe e4, Albul atacă pe flancul regelui.", "difficulty": "avansat"},
        {"eco": "C15", "name": "Franceză - Winawer", "name_en": "French Winawer",
         "moves": "e4 e6 d4 d5 Nc3 Bb4", "description": "Cel mai ascuțit răspuns. Structuri complexe.",
         "strategy": "Nebunul fixează calul, complicații tactice.", "difficulty": "avansat"},
        {"eco": "C20", "name": "Joc Deschis", "name_en": "Open Game",
         "moves": "e4 e5", "description": "Clasic. Ambii ocupă centrul.",
         "strategy": "Dezvoltare rapidă, luptă pentru centru.", "difficulty": "ușor"},
        {"eco": "C21", "name": "Gambitul Regelui Danez", "name_en": "Danish Gambit",
         "moves": "e4 e5 d4 exd4 c3", "description": "Sacrificiu dublu de pioni pentru dezvoltare.",
         "strategy": "Atac fulger, dezvoltare rapidă.", "difficulty": "intermediar"},
        {"eco": "C22", "name": "Gambitul Central", "name_en": "Center Game",
         "moves": "e4 e5 d4 exd4 Qxd4", "description": "Dama iese prea devreme dar centralizată.",
         "strategy": "Dezvoltare cu tempo pe damă.", "difficulty": "ușor"},
        {"eco": "C25", "name": "Jocul Vienez", "name_en": "Vienna Game",
         "moves": "e4 e5 Nc3", "description": "Pregătește f4 fără a expune calul.",
         "strategy": "Gambitul Regelui întârziat, atac pe f-file.", "difficulty": "intermediar"},
        {"eco": "C30", "name": "Gambitul Regelui", "name_en": "King's Gambit",
         "moves": "e4 e5 f4", "description": "Cel mai romantic gambit. Sacrificiu de pion pe f4.",
         "strategy": "Atac fulgerător pe flancul regelui.", "difficulty": "avansat"},
        {"eco": "C33", "name": "Gambitul Regelui Acceptat", "name_en": "King's Gambit Accepted",
         "moves": "e4 e5 f4 exf4", "description": "Negrul acceptă provocarea.",
         "strategy": "Albul deschide linii, Negrul trebuie să returneze pionul.", "difficulty": "avansat"},
        {"eco": "C36", "name": "Gambitul Regelui - Gambitul Abbazia", "name_en": "Abbazia Countergambit",
         "moves": "e4 e5 f4 exf4 d4", "description": "Gambitul modern cu d4.",
         "strategy": "Centre puternic, deschidere de diagonale.", "difficulty": "avansat"},
        {"eco": "C40", "name": "Apărarea Letonă", "name_en": "Latvian Gambit",
         "moves": "e4 e5 Nf3 f5", "description": "Gambit riscant al Negrului. Suicidal la nivel înalt.",
         "strategy": "Surpriză, contraatac nebunesc.", "difficulty": "avansat"},
        {"eco": "C42", "name": "Apărarea Rusă (Petrov)", "name_en": "Petrov Defense",
         "moves": "e4 e5 Nf3 Nf6", "description": "Simetrică și foarte solidă. Favorita jucătorilor de top.",
         "strategy": "Egalitate rapidă, joc pozițional.", "difficulty": "intermediar"},
        {"eco": "C44", "name": "Jocul Scoțian", "name_en": "Scotch Game",
         "moves": "e4 e5 Nf3 Nc6 d4", "description": "Deschide centrul devreme. Joc deschis.",
         "strategy": "Dezvoltare liberă, inițiativă.", "difficulty": "intermediar"},
        {"eco": "C45", "name": "Scoțianul - Varianta Clasică", "name_en": "Scotch Classical",
         "moves": "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5",
         "description": "Negrul dezvoltă nebunul activ.",
         "strategy": "Presiune pe d4, dezvoltare armonioasă.", "difficulty": "intermediar"},
        {"eco": "C46", "name": "Jocul Celor Trei Cai", "name_en": "Three Knights Game",
         "moves": "e4 e5 Nf3 Nc6 Nc3", "description": "Solid dar puțin pasiv.",
         "strategy": "Dezvoltare simetrică, joc pozițional.", "difficulty": "ușor"},
        {"eco": "C47", "name": "Jocul Celor Patru Cai", "name_en": "Four Knights Game",
         "moves": "e4 e5 Nf3 Nc6 Nc3 Nf6", "description": "Clasic, simetric, solid.",
         "strategy": "Joc pozițional, echilibru.", "difficulty": "ușor"},
        {"eco": "C50", "name": "Jocul Italian", "name_en": "Italian Game",
         "moves": "e4 e5 Nf3 Nc6 Bc4", "description": "Cea mai veche deschidere studiată. Elegantă.",
         "strategy": "Presiune pe f7, dezvoltare clasică.", "difficulty": "ușor"},
        {"eco": "C50", "name": "Jocul Italian - Giuoco Piano", "name_en": "Giuoco Piano",
         "moves": "e4 e5 Nf3 Nc6 Bc4 Bc5", "description": "\"Jocul liniștit\". Dezvoltare armonioasă.",
         "strategy": "Construcție pozițională, d3+c3 lent.", "difficulty": "ușor"},
        {"eco": "C51", "name": "Gambitul Evans", "name_en": "Evans Gambit",
         "moves": "e4 e5 Nf3 Nc6 Bc4 Bc5 b4", "description": "Sacrificiu de pion pentru tempo și centru.",
         "strategy": "Atac devastator cu dezvoltare rapidă.", "difficulty": "intermediar"},
        {"eco": "C53", "name": "Italian - Linia Clasică", "name_en": "Italian Classical",
         "moves": "e4 e5 Nf3 Nc6 Bc4 Bc5 c3", "description": "Pregătește d4 central.",
         "strategy": "Centru puternic, atac pe rege.", "difficulty": "intermediar"},
        {"eco": "C55", "name": "Jocul Italian cu Doi Cai", "name_en": "Two Knights Defense",
         "moves": "e4 e5 Nf3 Nc6 Bc4 Nf6", "description": "Negrul contraatacă agresiv.",
         "strategy": "Joc tactic, atac Fried Liver pe f7.", "difficulty": "intermediar"},
        {"eco": "C57", "name": "Atacul Fegatello (Fried Liver)", "name_en": "Fried Liver Attack",
         "moves": "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 Nxf7",
         "description": "Cel mai spectaculos sacrificiu de cal! Atac pe rege.",
         "strategy": "Atac de mat, sacrificiu de piesă pe f7.", "difficulty": "avansat"},
        {"eco": "C60", "name": "Deschiderea Spaniolă (Ruy López)", "name_en": "Ruy Lopez",
         "moves": "e4 e5 Nf3 Nc6 Bb5", "description": "\"Regele deschiderilor\". Cea mai studiată.",
         "strategy": "Presiune de lungă durată pe centru și flancul regelui.", "difficulty": "intermediar"},
        {"eco": "C63", "name": "Ruy López - Varianta Schliemann", "name_en": "Schliemann Defense",
         "moves": "e4 e5 Nf3 Nc6 Bb5 f5", "description": "Gambit agresiv contra Spaniolei.",
         "strategy": "Contraatac imediat, surpriză.", "difficulty": "avansat"},
        {"eco": "C65", "name": "Ruy López - Varianta Berlin", "name_en": "Berlin Defense",
         "moves": "e4 e5 Nf3 Nc6 Bb5 Nf6", "description": "\"Berlin Wall\". Extrem de solidă.",
         "strategy": "Endgame favorabil, apărare impenetrabilă.", "difficulty": "avansat"},
        {"eco": "C68", "name": "Ruy López - Varianta Schimb", "name_en": "Ruy Lopez Exchange",
         "moves": "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6", "description": "Schimb de nebun pe cal.",
         "strategy": "Majoritate de pioni pe flancul regelui.", "difficulty": "intermediar"},
        {"eco": "C70", "name": "Ruy López - Varianta Morphy", "name_en": "Morphy Defense",
         "moves": "e4 e5 Nf3 Nc6 Bb5 a6", "description": "Cea mai populară continuare.",
         "strategy": "Amenință nebunul, câștigă tempo.", "difficulty": "intermediar"},
        {"eco": "C78", "name": "Ruy López - Varianta Arkhangelsk", "name_en": "Arkhangelsk Variation",
         "moves": "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Bb7",
         "description": "Nebun activ pe b7, contraatac pe centru.",
         "strategy": "Presiune pe e4 din fianchetto.", "difficulty": "avansat"},
        {"eco": "C80", "name": "Ruy López - Deschidere Deschisă", "name_en": "Open Ruy Lopez",
         "moves": "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4",
         "description": "Negrul capturează pionul e4 imediat.",
         "strategy": "Joc tactic, complicații.", "difficulty": "avansat"},
        {"eco": "C84", "name": "Ruy López - Linia Închisă", "name_en": "Closed Ruy Lopez",
         "moves": "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7",
         "description": "Linia principală. Cea mai studiată din istoria șahului.",
         "strategy": "Manevre strategice pe termen lung.", "difficulty": "avansat"},
        {"eco": "C89", "name": "Ruy López - Atacul Marshall", "name_en": "Marshall Attack",
         "moves": "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d5",
         "description": "Gambitul lui Frank Marshall. Sacrificiu de pion spectaculos.",
         "strategy": "Atac devastator pe rege, compensație dinamică.", "difficulty": "avansat"},

        # D: Closed Games (1.d4 d5)
        {"eco": "D00", "name": "Deschiderea Pionului Damei", "name_en": "Queen's Pawn Game",
         "moves": "d4 d5", "description": "Joc închis clasic. Mai strategic decât 1.e4.",
         "strategy": "Control pozițional, joc pe flancul damei.", "difficulty": "ușor"},
        {"eco": "D02", "name": "Sistemul London", "name_en": "London System",
         "moves": "d4 d5 Nf3 Nf6 Bf4", "description": "Sistem universal. Ușor de învățat.",
         "strategy": "Structură solidă piramidală, joc pozițional.", "difficulty": "ușor"},
        {"eco": "D06", "name": "Gambitul Damei", "name_en": "Queen's Gambit",
         "moves": "d4 d5 c4", "description": "Cel mai important gambit clasic.",
         "strategy": "Control pe centru, presiune pe d5.", "difficulty": "intermediar"},
        {"eco": "D07", "name": "Gambitul Damei - Varianta Chigorin", "name_en": "Chigorin Defense",
         "moves": "d4 d5 c4 Nc6", "description": "Cal pe c6 în loc de e6. Neobișnuit dar activ.",
         "strategy": "Dezvoltare activă, joc tactic.", "difficulty": "intermediar"},
        {"eco": "D10", "name": "Gambitul Damei - Varianta Slav", "name_en": "Slav Defense",
         "moves": "d4 d5 c4 c6", "description": "Solidă ca un zid. Nebunul rămâne liber pe c8.",
         "strategy": "Structură solidă, nebun activ pe câmpuri albe.", "difficulty": "intermediar"},
        {"eco": "D20", "name": "Gambitul Damei Acceptat", "name_en": "Queen's Gambit Accepted",
         "moves": "d4 d5 c4 dxc4", "description": "Negrul ia pionul. Trebuie returnat cu tempo.",
         "strategy": "Joc deschis, dezvoltare liberă.", "difficulty": "intermediar"},
        {"eco": "D30", "name": "Gambitul Damei Refuzat", "name_en": "Queen's Gambit Declined",
         "moves": "d4 d5 c4 e6", "description": "Clasic, solid, preferat de jucătorii de elită.",
         "strategy": "Apărare solidă, nebun problematic pe c8.", "difficulty": "intermediar"},
        {"eco": "D35", "name": "GDR - Varianta Schimb", "name_en": "QGD Exchange",
         "moves": "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5",
         "description": "Structură de pioni simetrică. Joc de endgame.",
         "strategy": "Atac pe minoritate pe flancul damei.", "difficulty": "intermediar"},
        {"eco": "D37", "name": "GDR - Linia Clasică", "name_en": "QGD Classical",
         "moves": "d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7",
         "description": "Cea mai tradițională linie.",
         "strategy": "Joc pozițional profund.", "difficulty": "intermediar"},
        {"eco": "D43", "name": "Semi-Slav", "name_en": "Semi-Slav Defense",
         "moves": "d4 d5 c4 c6 Nc3 Nf6 Nf3 e6",
         "description": "Combinație între Slav și QGD. Foarte populară.",
         "strategy": "Structură solidă cu posibilități de contraatac.", "difficulty": "avansat"},
        {"eco": "D45", "name": "Semi-Slav - Varianta Meran", "name_en": "Meran Variation",
         "moves": "d4 d5 c4 c6 Nc3 Nf6 Nf3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5",
         "description": "Negrul avansează agresiv pe flancul damei.",
         "strategy": "Expansiune pe flanc, joc tactic complex.", "difficulty": "avansat"},

        # E: Indian Defenses
        {"eco": "E00", "name": "Apărarea Indiană a Damei", "name_en": "Queen's Indian",
         "moves": "d4 Nf6 c4 e6 Nf3 b6",
         "description": "Fianchetto pe b7. Controlează diagonala mare.",
         "strategy": "Control pe e4, joc pozițional fluid.", "difficulty": "intermediar"},
        {"eco": "E12", "name": "Indian Damei - Varianta Petrosian", "name_en": "QID Petrosian",
         "moves": "d4 Nf6 c4 e6 Nf3 b6 a3",
         "description": "Previne Bb4. Foarte solid.",
         "strategy": "Control pozițional, evitarea fixărilor.", "difficulty": "intermediar"},
        {"eco": "E15", "name": "Indian Damei - Varianta Nimzowitsch", "name_en": "QID Nimzowitsch",
         "moves": "d4 Nf6 c4 e6 Nf3 b6 g3",
         "description": "Fianchetto dublu. Bătălia diagonalelor.",
         "strategy": "Control pe câmpuri albe.", "difficulty": "intermediar"},
        {"eco": "E20", "name": "Apărarea Nimzo-Indiană", "name_en": "Nimzo-Indian Defense",
         "moves": "d4 Nf6 c4 e6 Nc3 Bb4",
         "description": "Una dintre cele mai respectate apărări. Fixează calul pe c3.",
         "strategy": "Presiune pe e4, structură de pioni dublați.", "difficulty": "avansat"},
        {"eco": "E32", "name": "Nimzo-Indiană - Varianta Clasică", "name_en": "Nimzo-Indian Classical",
         "moves": "d4 Nf6 c4 e6 Nc3 Bb4 Qc2",
         "description": "Evită pionii dublați. Pregătește e4.",
         "strategy": "Expansiune în centru, joc de manevră.", "difficulty": "avansat"},
        {"eco": "E40", "name": "Nimzo-Indiană - Varianta Rubinstein", "name_en": "Nimzo-Indian Rubinstein",
         "moves": "d4 Nf6 c4 e6 Nc3 Bb4 e3",
         "description": "Solidă. Acceptă pionii dublați pentru pereche de nebuni.",
         "strategy": "Nebuni puternici, centru stabil.", "difficulty": "intermediar"},
        {"eco": "E60", "name": "Apărarea Indiană a Regelui", "name_en": "King's Indian Defense",
         "moves": "d4 Nf6 c4 g6",
         "description": "Hipermodernă și combativă. Favorizata lui Kasparov.",
         "strategy": "Fianchetto, contraatac pe flancul regelui.", "difficulty": "avansat"},
        {"eco": "E62", "name": "KID - Varianta Fianchetto", "name_en": "KID Fianchetto",
         "moves": "d4 Nf6 c4 g6 Nc3 Bg7 Nf3 d6 g3",
         "description": "Linia cea mai pozițională contra KID.",
         "strategy": "Fianchetto dublu, joc strategic.", "difficulty": "intermediar"},
        {"eco": "E70", "name": "KID - Linia Clasică", "name_en": "KID Classical",
         "moves": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3",
         "description": "Cea mai naturală dezvoltare.",
         "strategy": "Centru puternic, ambii joacă pe flancuri opuse.", "difficulty": "avansat"},
        {"eco": "E73", "name": "KID - Varianta Averbakh", "name_en": "KID Averbakh",
         "moves": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5",
         "description": "Nebun agresiv pe g5, restricționează Negrul.",
         "strategy": "Presiune pozițională, restricționare.", "difficulty": "avansat"},
        {"eco": "E76", "name": "KID - Varianta Samisch", "name_en": "KID Samisch",
         "moves": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3",
         "description": "Construcție masivă de centru. Pregătește Be3.",
         "strategy": "Centru impunător, atac pe flancul damei.", "difficulty": "avansat"},
        {"eco": "E80", "name": "KID - Varianta Sämisch cu f5", "name_en": "KID Samisch f5",
         "moves": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5",
         "description": "Clasică bătălie pe flancuri opuse.",
         "strategy": "Albul atacă pe damă, Negrul pe rege.", "difficulty": "avansat"},
        {"eco": "E90", "name": "KID - Linia Principală", "name_en": "KID Main Line",
         "moves": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O",
         "description": "Pozitia de bază. De aici se ramifică toata teoria.",
         "strategy": "Ambii au planuri concrete pe flancuri opuse.", "difficulty": "avansat"},
        {"eco": "E97", "name": "KID - Varianta Mar del Plata", "name_en": "KID Mar del Plata",
         "moves": "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6",
         "description": "Cea mai faimoasă bătălie din KID. Două armate se atacă reciproc.",
         "strategy": "Albul: c5+d5 pe damă. Negrul: f5+g4 pe rege.", "difficulty": "avansat"},
        {"eco": "E61", "name": "Apărarea Grünfeld", "name_en": "Grunfeld Defense",
         "moves": "d4 Nf6 c4 g6 Nc3 d5",
         "description": "Hipermodernă. Negrul cedează centrul pentru a-l ataca.",
         "strategy": "Presiune pe d4 cu nebunul g7, contraatac.", "difficulty": "avansat"},
        {"eco": "E70", "name": "Apărarea Benoni", "name_en": "Benoni Defense",
         "moves": "d4 Nf6 c4 c5",
         "description": "Asimetrică și dinamică. Joc pe flancuri opuse.",
         "strategy": "Majoritate pe flancul damei, contraatac.", "difficulty": "avansat"},
        {"eco": "A56", "name": "Benoni Modernă", "name_en": "Modern Benoni",
         "moves": "d4 Nf6 c4 c5 d5 e6",
         "description": "Acceptă dezavantaj de spațiu pentru dinamism.",
         "strategy": "Presiune pe c-file și e-file.", "difficulty": "avansat"},
        {"eco": "A57", "name": "Gambitul Benko", "name_en": "Benko Gambit",
         "moves": "d4 Nf6 c4 c5 d5 b5",
         "description": "Sacrificiu de pion pentru presiune pe flancul damei.",
         "strategy": "Coloane deschise a și b, presiune de lungă durată.", "difficulty": "avansat"},
        {"eco": "A45", "name": "Deschiderea Trompowsky", "name_en": "Trompowsky Attack",
         "moves": "d4 Nf6 Bg5",
         "description": "Nebun agresiv pe g5. Evită teoria mainstream.",
         "strategy": "Surpriză, structuri atipice.", "difficulty": "intermediar"},
        {"eco": "D00", "name": "Sistemul Colle", "name_en": "Colle System",
         "moves": "d4 d5 Nf3 Nf6 e3",
         "description": "Sistem ușor de învățat. Piramidă de pioni.",
         "strategy": "e4 breakthrough, atac pe rege.", "difficulty": "ușor"},
    ]

    df = pd.DataFrame(openings)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Create {len(df)} deschideri → {csv_path}")
    return True


# ──────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────

def check_datasets():
    """Verify all datasets are present."""
    print(f"\n{'='*60}")
    print(f"  🔍 Verificare datasets")
    print(f"{'='*60}")

    checks = {
        "Imagini train": (TRAIN_DIR, lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0),
        "Imagini val":   (VAL_DIR, lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0),
        "Jocuri CSV":    (os.path.join(DATA_DIR, "chess_games.csv"), lambda p: os.path.exists(p)),
        "Deschideri CSV": (os.path.join(DATA_DIR, "openings.csv"), lambda p: os.path.exists(p)),
    }

    all_ok = True
    for name, (path, check_fn) in checks.items():
        result = check_fn(path)
        status = "✅" if result else "❌"
        print(f"  {status} {name}: {result}")
        if not result:
            all_ok = False

    return all_ok


# ──────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Chess Datasets")
    parser.add_argument('--images-only', action='store_true')
    parser.add_argument('--games-only', action='store_true')
    parser.add_argument('--openings-only', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--max-images', type=int, default=5000)
    args = parser.parse_args()

    if args.check:
        check_datasets()
        sys.exit(0)

    if args.images_only:
        download_board_images(args.max_images)
    elif args.games_only:
        download_chess_games()
    elif args.openings_only:
        create_openings_database()
    else:
        # Download all
        create_openings_database()
        download_chess_games()
        download_board_images(args.max_images)
        print(f"\n{'='*60}")
        check_datasets()
