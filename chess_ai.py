import chess

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def evaluate_board(board):
    if board.is_checkmate():
        if board.turn:
            return -99999
        else:
            return 99999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    evaluation = 0
    for piece_type in piece_values:
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    # We can add piece-square tables here for better positional play
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

def get_best_move(board, depth=3):
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
