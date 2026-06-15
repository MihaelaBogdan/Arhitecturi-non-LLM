import chess

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

PIECE_NAMES_RO = {
    chess.PAWN:   "Pionul",
    chess.KNIGHT: "Calul",
    chess.BISHOP: "Nebunul",
    chess.ROOK:   "Turnul",
    chess.QUEEN:  "Regina",
    chess.KING:   "Regele",
}

# --- Piece-Square Tables (White perspective, rank 0 = rank 1) ---
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
KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PIECE_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE,
}

def _pst_score(piece_type: int, color: bool, square: int) -> int:
    table = PIECE_TABLES[piece_type]
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if color == chess.WHITE:
        idx = (7 - rank) * 8 + file
    else:
        idx = rank * 8 + file
    return table[idx]

def evaluate_board(board: chess.Board) -> int:
    if board.is_checkmate():
        return -99999 if board.turn else 99999
    if board.is_stalemate() or board.is_insufficient_material() or \
       board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        val = piece_values[piece.piece_type] + _pst_score(piece.piece_type, piece.color, square)
        if piece.color == chess.WHITE:
            evaluation += val
        else:
            evaluation -= val
    return evaluation


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_best_move(board: chess.Board, depth: int = 3) -> str | None:
    best_move = None
    if board.turn == chess.WHITE:
        best_value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, -float('inf'), float('inf'), False)
            board.pop()
            if board_value > best_value:
                best_value = board_value
                best_move = move
    else:
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, -float('inf'), float('inf'), True)
            board.pop()
            if board_value < best_value:
                best_value = board_value
                best_move = move
    return best_move.uci() if best_move else None


# --- Opening recognition ---
OPENINGS = [
    (["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"], "Deschiderea Ruy López (Spaniola)"),
    (["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"], "Jocul Scoțian"),
    (["e2e4", "e7e5", "f2f4"],                  "Gambitul Regelui"),
    (["e2e4", "c7c5"],                           "Apărarea Siciliană"),
    (["e2e4", "e7e6"],                           "Apărarea Franceză"),
    (["e2e4", "c7c6"],                           "Apărarea Caro-Kann"),
    (["e2e4", "d7d5"],                           "Apărarea Scandinavă"),
    (["e2e4", "e7e5"],                           "Joc Deschis (1.e4 e5)"),
    (["e2e4"],                                   "Deschiderea Pionului Regelui"),
    (["d2d4", "d7d5", "c2c4"],                   "Gambitul Damei"),
    (["d2d4", "g8f6", "c2c4", "e7e6"],           "Apărarea Nimzo-Indiană"),
    (["d2d4", "g8f6"],                           "Apărarea Indiană"),
    (["d2d4", "d7d5"],                           "Joc Închis (1.d4 d5)"),
    (["d2d4"],                                   "Deschiderea Pionului Damei"),
    (["g1f3"],                                   "Deschiderea Reti"),
    (["c2c4"],                                   "Deschiderea Engleză"),
]

def detect_opening(board: chess.Board) -> str:
    moves = [m.uci() for m in board.move_stack]
    best = "Deschidere necunoscută"
    best_len = 0
    for seq, name in OPENINGS:
        if moves[:len(seq)] == seq and len(seq) > best_len:
            best = name
            best_len = len(seq)
    if best_len == 0 and len(moves) == 0:
        return "Poziție de start"
    return best


# --- Move explanation in Romanian ---
def explain_move(board_before: chess.Board, move: chess.Move) -> str:
    """Generate a human-readable Romanian explanation for a move."""
    piece = board_before.piece_at(move.from_square)
    if not piece:
        return "Mutare necunoscută."

    piece_name = PIECE_NAMES_RO.get(piece.piece_type, "Piesa")
    from_sq = chess.square_name(move.from_square).upper()
    to_sq   = chess.square_name(move.to_square).upper()
    color   = "Alb" if piece.color == chess.WHITE else "Negru"

    # Castling
    if board_before.is_castling(move):
        side = "scurtă (pe flancul regelui)" if board_before.is_kingside_castling(move) else "lungă (pe flancul damei)"
        return f"Rocadă {side} — Regele {color} se adăpostește în spatele pionilor."

    # Capture
    captured = board_before.piece_at(move.to_square)
    if captured:
        cap_name = PIECE_NAMES_RO.get(captured.piece_type, "piesa").lower()
        # Check if it's a good trade
        cap_val = piece_values.get(captured.piece_type, 0)
        own_val = piece_values.get(piece.piece_type, 0)
        if cap_val > own_val:
            trade = "câștigând material!"
        elif cap_val == own_val:
            trade = "schimb egal."
        else:
            trade = "sacrificând material pentru poziție."
        return f"{piece_name} {color} capturează {cap_name} pe {to_sq}, {trade}"

    # Promotion
    if move.promotion:
        prom_name = PIECE_NAMES_RO.get(move.promotion, "Regină")
        return f"Pionul {color} ajunge pe ultima linie și este promovat la {prom_name}!"

    # En passant
    if board_before.is_en_passant(move):
        return f"{piece_name} {color} capturează en passant pe {to_sq}."

    # Check (check after move)
    board_after = board_before.copy()
    board_after.push(move)
    if board_after.is_checkmate():
        return f"ȘAH MAT! {piece_name} {color} mută pe {to_sq}. Joc terminat!"
    if board_after.is_check():
        return f"{piece_name} {color} mută pe {to_sq} — ȘAH! Regele advers este atacat."

    # Center control
    center = {chess.E4, chess.E5, chess.D4, chess.D5}
    if move.to_square in center:
        return f"{piece_name} {color} controlează centrul mutând pe {to_sq}."

    # Development (from back rank)
    if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        from_rank = chess.square_rank(move.from_square)
        if (piece.color == chess.WHITE and from_rank == 0) or \
           (piece.color == chess.BLACK and from_rank == 7):
            return f"{piece_name} {color} este dezvoltat pe {to_sq}, controlând câmpuri importante."

    # Default
    return f"{piece_name} {color} mută de pe {from_sq} pe {to_sq}."


# --- Material balance ---
def get_material_balance(board: chess.Board) -> dict:
    """Return captured pieces for each side."""
    initial = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
               chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
    white_on_board = {pt: len(board.pieces(pt, chess.WHITE)) for pt in initial}
    black_on_board = {pt: len(board.pieces(pt, chess.BLACK)) for pt in initial}

    PIECE_SYMBOLS = {
        chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
        chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚",
    }
    # Pieces captured FROM white (black captured them)
    white_captured = ""
    black_captured = ""
    for pt, sym in PIECE_SYMBOLS.items():
        diff_w = initial[pt] - white_on_board[pt]
        diff_b = initial[pt] - black_on_board[pt]
        white_captured += sym * diff_b   # black lost these (white captured)
        black_captured += sym * diff_w   # white lost these (black captured)

    return {
        "white_captured": white_captured or "—",
        "black_captured": black_captured or "—",
        "advantage": evaluate_board(board)
    }
