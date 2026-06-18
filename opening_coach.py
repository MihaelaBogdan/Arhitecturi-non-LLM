"""
Opening Coach — Chess opening teacher and advisor.

Features:
  - Detectează deschiderea curentă din ECO database
  - Sugerează continuări teoretice (linii principale)
  - Oferă lecții detaliate despre fiecare deschidere
  - Recomandă deschideri bazate pe stil de joc
"""

import os
import csv
import chess
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OPENINGS_CSV = os.path.join(DATA_DIR, "openings.csv")


# ──────────────────────────────────────────────────
# Openings Database
# ──────────────────────────────────────────────────

class Opening:
    """Represents a chess opening."""
    def __init__(self, eco: str, name: str, name_en: str, moves: str,
                 description: str, strategy: str, difficulty: str):
        self.eco = eco
        self.name = name
        self.name_en = name_en
        self.moves_san = moves  # SAN notation string
        self.description = description
        self.strategy = strategy
        self.difficulty = difficulty
        
        # Parse moves to UCI for matching
        self.moves_uci = self._san_to_uci(moves)
    
    def _san_to_uci(self, san_moves: str) -> list[str]:
        """Convert SAN moves string to list of UCI moves."""
        board = chess.Board()
        uci_list = []
        for san in san_moves.split():
            try:
                move = board.parse_san(san)
                uci_list.append(move.uci())
                board.push(move)
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                break
        return uci_list
    
    def to_dict(self) -> dict:
        return {
            "eco": self.eco,
            "name": self.name,
            "name_en": self.name_en,
            "moves": self.moves_san,
            "moves_uci": " ".join(self.moves_uci),
            "description": self.description,
            "strategy": self.strategy,
            "difficulty": self.difficulty,
        }


class OpeningCoach:
    """Chess opening teacher and advisor."""

    def __init__(self):
        self.openings: list[Opening] = []
        self._load_openings()

    def _load_openings(self):
        """Load openings from CSV database."""
        if not os.path.exists(OPENINGS_CSV):
            print(f"⚠️  Nu am găsit baza de date: {OPENINGS_CSV}")
            print("   Rulează: python download_datasets.py --openings-only")
            return

        try:
            with open(OPENINGS_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        opening = Opening(
                            eco=row.get('eco', ''),
                            name=row.get('name', ''),
                            name_en=row.get('name_en', ''),
                            moves=row.get('moves', ''),
                            description=row.get('description', ''),
                            strategy=row.get('strategy', ''),
                            difficulty=row.get('difficulty', 'intermediar'),
                        )
                        if opening.moves_uci:  # only add if moves parsed successfully
                            self.openings.append(opening)
                    except Exception:
                        continue

            print(f"📖 Loaded {len(self.openings)} openings from database")
        except Exception as e:
            print(f"❌ Error loading openings: {e}")

    def detect_opening(self, board: chess.Board) -> Optional[Opening]:
        """
        Detect the opening of the current game position.
        Returns the most specific (longest) matching opening.
        """
        game_moves = [m.uci() for m in board.move_stack]
        
        best_match = None
        best_length = 0

        for opening in self.openings:
            op_moves = opening.moves_uci
            if len(op_moves) > len(game_moves):
                continue
            if game_moves[:len(op_moves)] == op_moves and len(op_moves) > best_length:
                best_match = opening
                best_length = len(op_moves)

        return best_match

    def get_opening_info(self, board: chess.Board) -> dict:
        """
        Get comprehensive information about the current opening.
        Returns info dict with opening details and suggested continuations.
        """
        game_moves = [m.uci() for m in board.move_stack]

        # Special Case: Starting Position
        if not game_moves:
            return {
                "detected": True,
                "name": "Poziție de start",
                "name_en": "Starting Position",
                "eco": "Start",
                "description": "Partida abia a început. Mută un pion de la centru (e4 sau d4) sau dezvoltă un cal (Nf3) pentru a controla centrul tablei.",
                "strategy": "Controlează centrul, dezvoltă piesele minore (cai și nebuni) și pregătește rocada pentru siguranța regelui.",
                "difficulty": "ușor",
                "suggestions": [
                    {"name": "Deschiderea Pionului Regelui (1.e4)", "eco": "C20", "next_move": "e2e4", "description": "Cea mai populară mutare, deschide calea nebunului și damei.", "difficulty": "ușor"},
                    {"name": "Deschiderea Pionului Damei (1.d4)", "eco": "D00", "next_move": "d2d4", "description": "O mutare solidă și strategică, controlează câmpul e5.", "difficulty": "ușor"},
                    {"name": "Deschiderea Réti (1.Nf3)", "eco": "A04", "next_move": "g1f3", "description": "Dezvoltare flexibilă a calului, atacând centrul de la distanță.", "difficulty": "intermediar"},
                    {"name": "Deschiderea Engleză (1.c4)", "eco": "A10", "next_move": "c2c4", "description": "Luptă pentru câmpul d5 de pe flanc.", "difficulty": "intermediar"}
                ],
                "lesson": self._get_general_opening_tips(),
            }

        opening = self.detect_opening(board)

        if not opening:
            # Try to find what the game is heading towards
            possible = self._find_possible_openings(game_moves)
            return {
                "detected": False,
                "name": "Deschidere necunoscută",
                "name_en": "Unknown Opening",
                "eco": "?",
                "description": "Această secvență de mutări nu corespunde unei deschideri standard.",
                "strategy": "",
                "difficulty": "",
                "suggestions": possible[:5],
                "lesson": self._get_general_opening_tips(),
            }

        # Find continuations (openings that extend from the current one)
        continuations = self._find_continuations(opening, game_moves)

        return {
            "detected": True,
            "name": opening.name,
            "name_en": opening.name_en,
            "eco": opening.eco,
            "description": opening.description,
            "strategy": opening.strategy,
            "difficulty": opening.difficulty,
            "moves": opening.moves_san,
            "suggestions": continuations,
            "lesson": self._get_opening_lesson(opening),
        }

    def _find_continuations(self, opening: Opening, game_moves: list[str]) -> list[dict]:
        """Find openings that are continuations of the current one."""
        continuations = []
        op_moves = opening.moves_uci

        for other in self.openings:
            other_moves = other.moves_uci
            # Must be strictly longer and start with current opening
            if len(other_moves) > len(op_moves) and \
               other_moves[:len(op_moves)] == op_moves and \
               other is not opening:
                # Only suggest if the game hasn't gone past this point
                if len(game_moves) < len(other_moves):
                    # What's the next move to reach this continuation?
                    next_move_idx = len(game_moves)
                    if next_move_idx < len(other_moves):
                        next_move = other_moves[next_move_idx]
                        continuations.append({
                            "name": other.name,
                            "eco": other.eco,
                            "next_move": next_move,
                            "description": other.description[:100],
                            "difficulty": other.difficulty,
                        })

        # Remove duplicates by next_move
        seen = set()
        unique = []
        for c in continuations:
            if c["next_move"] not in seen:
                seen.add(c["next_move"])
                unique.append(c)

        return unique[:5]

    def _find_possible_openings(self, game_moves: list[str]) -> list[dict]:
        """Find openings the game could transpose into."""
        possible = []
        n_moves = len(game_moves)

        for opening in self.openings:
            op_moves = opening.moves_uci
            # Opening must be reachable (starts with what we've played so far)
            if len(op_moves) > n_moves:
                if op_moves[:n_moves] == game_moves:
                    next_move = op_moves[n_moves]
                    possible.append({
                        "name": opening.name,
                        "eco": opening.eco,
                        "next_move": next_move,
                        "description": opening.description[:100],
                        "difficulty": opening.difficulty,
                    })

        return possible

    def _get_opening_lesson(self, opening: Opening) -> dict:
        """Generate a detailed lesson about the opening."""
        return {
            "title": f"📖 Lecție: {opening.name}",
            "eco": opening.eco,
            "english_name": opening.name_en,
            "theory": opening.description,
            "strategy": opening.strategy,
            "difficulty": opening.difficulty,
            "key_ideas": self._get_key_ideas(opening),
            "common_mistakes": self._get_common_mistakes(opening),
            "recommended_for": self._get_recommendation(opening),
        }

    def _get_key_ideas(self, opening: Opening) -> list[str]:
        """Get key ideas for the opening based on its characteristics."""
        ideas = []
        moves = opening.moves_san.lower()

        if 'e4' in moves and 'e5' in moves:
            ideas.append("🎯 Control central clasic cu pioni pe e4/e5")
        if 'd4' in moves and 'd5' in moves:
            ideas.append("🏰 Structură închisă — joc pozițional de lungă durată")
        if 'nf3' in moves.lower():
            ideas.append("🐴 Dezvoltare naturală a calului — controlează centrul")
        if 'bc4' in moves.lower() or 'bb5' in moves.lower():
            ideas.append("📐 Nebun activ — presiune pe diagonală")
        if 'o-o' in moves.lower() or 'kg1' in moves.lower():
            ideas.append("🏠 Rocadă timpurie — siguranța regelui")
        if 'c4' in moves:
            ideas.append("⚔️ Control pe câmpul d5 — strategie pe flancul damei")
        if 'f4' in moves or 'f5' in moves:
            ideas.append("🔥 Atac pe flancul regelui — joc agresiv")
        if 'g6' in moves or 'bg7' in moves.lower():
            ideas.append("🐉 Fianchetto — nebun puternic pe diagonala mare")
        if 'b4' in moves or 'b5' in moves:
            ideas.append("🚀 Expansiune pe flancul damei")
        if 'bb4' in moves.lower():
            ideas.append("📌 Fixarea calului — presiune pozițională")

        if not ideas:
            ideas.append("🎯 Dezvoltare armonioasă a pieselor")
            ideas.append("🏰 Siguranța regelui prin rocadă")
            ideas.append("⚔️ Control central")

        return ideas[:5]

    def _get_common_mistakes(self, opening: Opening) -> list[str]:
        """Get common mistakes for beginners in this opening."""
        mistakes = []
        eco = opening.eco

        if eco.startswith('B2') or eco.startswith('B3') or eco.startswith('B5'):  # Sicilian
            mistakes.append("❌ Nu juca d3 prea devreme — pierzi tempo")
            mistakes.append("❌ Nu neglija dezvoltarea pentru atacuri premature")
        elif eco.startswith('C6') or eco.startswith('C7') or eco.startswith('C8'):  # Ruy Lopez
            mistakes.append("❌ Nu lua pionul e5 prea devreme cu nebunul")
            mistakes.append("❌ Nu pierde tempo cu nebunul pe a4-b3")
        elif eco.startswith('D0') or eco.startswith('D1'):  # QGD
            mistakes.append("❌ Nu bloca nebunul pe c8 cu e6 prematur")
            mistakes.append("❌ Nu neglija flancul regelui")
        elif eco.startswith('E6') or eco.startswith('E7') or eco.startswith('E9'):  # KID
            mistakes.append("❌ Nu întârzia f5 — e esențial pentru contraatac")
            mistakes.append("❌ Nu lăsa albul să avanseze nestingherit pe flancul damei")

        mistakes.append("❌ Nu neglija dezvoltarea pieselor minore")
        mistakes.append("❌ Nu muta aceeași piesă de două ori în deschidere")
        mistakes.append("❌ Nu scoate dama prea devreme")

        return mistakes[:4]

    def _get_recommendation(self, opening: Opening) -> str:
        """Who should play this opening."""
        diff = opening.difficulty
        if diff == "ușor":
            return "👤 Recomandată pentru începători și intermediari. Ușor de învățat și de jucat."
        elif diff == "intermediar":
            return "👥 Recomandată pentru jucători intermediari. Necesită cunoștințe de bază."
        else:
            return "🏆 Recomandată pentru jucători avansați. Necesită studiu aprofundat."

    def _get_general_opening_tips(self) -> dict:
        """General opening tips when no specific opening is detected."""
        return {
            "title": "📖 Principii Generale de Deschidere",
            "theory": "Nu am detectat o deschidere specifică.",
            "strategy": "Urmează principiile fundamentale.",
            "key_ideas": [
                "🎯 Controlează centrul (e4, d4, e5, d5)",
                "🐴 Dezvoltă piesele minore (cai, nebuni) primele",
                "🏠 Rocadă devreme pentru siguranța regelui",
                "👸 Nu scoate dama prea devreme",
                "🔗 Conectează turnurile",
            ],
            "common_mistakes": [
                "❌ Nu muta aceeași piesă de două ori",
                "❌ Nu vâna pioni în deschidere",
                "❌ Nu neglija dezvoltarea",
            ],
            "recommended_for": "👤 Principii universale pentru toți jucătorii.",
        }

    def get_all_openings(self, difficulty: Optional[str] = None) -> list[dict]:
        """Get all openings, optionally filtered by difficulty."""
        openings = self.openings
        if difficulty:
            openings = [o for o in openings if o.difficulty == difficulty]
        return [o.to_dict() for o in openings]

    def search_openings(self, query: str) -> list[dict]:
        """Search openings by name, ECO code, or description."""
        query = query.lower()
        results = []
        for opening in self.openings:
            if (query in opening.name.lower() or
                query in opening.name_en.lower() or
                query in opening.eco.lower() or
                query in opening.description.lower()):
                results.append(opening.to_dict())
        return results

    def suggest_opening_for_style(self, style: str) -> list[dict]:
        """Suggest openings based on playing style."""
        style = style.lower()

        if style in ('agresiv', 'aggressive', 'atac', 'attack'):
            targets = ['Gambitul Regelui', 'Evans', 'Fried Liver', 'Dragon',
                       'Najdorf', 'Grob', 'Smith-Morra']
        elif style in ('solid', 'defensiv', 'defensive', 'pozitional'):
            targets = ['Caro-Kann', 'Petrov', 'London', 'Colle', 'Berlin',
                       'Franceză', 'Slav']
        elif style in ('tacticesc', 'tactical', 'combinativ'):
            targets = ['Siciliană', 'Italian', 'Scoțian', 'Gambitul Damei',
                       'Sveshnikov', 'Nimzo-Indiană']
        elif style in ('universal', 'flexibil', 'flexible'):
            targets = ['Réti', 'Engleză', 'Indian', 'Pirc', 'Modernă']
        else:
            targets = ['Italian', 'Spaniolă', 'Siciliană', 'London', 'Caro-Kann']

        results = []
        for opening in self.openings:
            for target in targets:
                if target.lower() in opening.name.lower():
                    results.append(opening.to_dict())
                    break

        return results[:10]


# ──────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────

_coach: OpeningCoach | None = None


def get_opening_coach() -> OpeningCoach:
    """Get the singleton opening coach instance."""
    global _coach
    if _coach is None:
        _coach = OpeningCoach()
    return _coach
