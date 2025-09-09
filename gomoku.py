"""
Gomoku core engine and CLI.

Features:
- Board model with win detection (configurable win length, overline rule)
- Turn-based game engine with undo/redo and resign
- Clean Python API for programmatic play
- Interactive CLI for playing in the terminal

Usage (CLI):
  python gomoku.py --size 15 --win 5  # start interactive game

API example:
  from gomoku import GomokuGame, GomokuConfig, Player
  game = GomokuGame(GomokuConfig(board_size=15, win_length=5))
  game.play_move(7, 7)  # Player.BLACK by default
  game.play_move(7, 8)
  print(game.render())
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable, List, Optional, Sequence, Tuple


__all__ = [
    "Player",
    "GameOutcome",
    "GameResult",
    "GomokuConfig",
    "GomokuError",
    "InvalidMoveError",
    "GameOverError",
    "CoordinateParseError",
    "GomokuGame",
    "new_game",
]


class Player(IntEnum):
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Player":
        return Player.BLACK if self is Player.WHITE else Player.WHITE


class GameOutcome(Enum):
    ONGOING = "ongoing"
    WIN = "win"
    DRAW = "draw"
    RESIGN = "resign"


@dataclass(frozen=True)
class GameResult:
    outcome: GameOutcome
    winner: Optional[Player] = None
    winning_line: Optional[List[Tuple[int, int]]] = None
    reason: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.outcome is not GameOutcome.ONGOING


@dataclass(frozen=True)
class GomokuConfig:
    board_size: int = 15
    win_length: int = 5
    allow_overline: bool = True  # when False, require exactly win_length
    first_player: Player = Player.BLACK

    def __post_init__(self) -> None:
        if self.board_size <= 0:
            raise ValueError("board_size must be positive")
        if self.win_length <= 1:
            raise ValueError("win_length must be > 1")
        if self.win_length > self.board_size:
            raise ValueError("win_length cannot exceed board_size")


class GomokuError(Exception):
    pass


class InvalidMoveError(GomokuError):
    pass


class GameOverError(GomokuError):
    pass


class CoordinateParseError(GomokuError):
    pass


@dataclass
class Move:
    row_index: int
    col_index: int
    player: Player


class Board:
    """Game board state and win detection.

    Cells are stored as integers:
      0 = empty, 1 = Player.BLACK, 2 = Player.WHITE
    """

    def __init__(self, size: int) -> None:
        self.size: int = size
        # grid[row][col] indexing
        self.grid: List[List[int]] = [[0 for _ in range(size)] for _ in range(size)]
        self.num_occupied: int = 0

    # ----------------------- basic operations -----------------------
    def in_bounds(self, row_index: int, col_index: int) -> bool:
        return 0 <= row_index < self.size and 0 <= col_index < self.size

    def get_cell(self, row_index: int, col_index: int) -> int:
        if not self.in_bounds(row_index, col_index):
            raise IndexError("cell out of bounds")
        return self.grid[row_index][col_index]

    def is_empty(self, row_index: int, col_index: int) -> bool:
        return self.get_cell(row_index, col_index) == 0

    def place_stone(self, row_index: int, col_index: int, player: Player) -> None:
        if not self.in_bounds(row_index, col_index):
            raise InvalidMoveError("move out of bounds")
        if self.grid[row_index][col_index] != 0:
            raise InvalidMoveError("cell is already occupied")
        self.grid[row_index][col_index] = int(player)
        self.num_occupied += 1

    def remove_stone(self, row_index: int, col_index: int) -> None:
        if not self.in_bounds(row_index, col_index):
            raise IndexError("cell out of bounds")
        if self.grid[row_index][col_index] == 0:
            raise InvalidMoveError("cannot remove from empty cell")
        self.grid[row_index][col_index] = 0
        self.num_occupied -= 1

    def is_full(self) -> bool:
        return self.num_occupied >= self.size * self.size

    def legal_moves(self) -> Iterable[Tuple[int, int]]:
        for r in range(self.size):
            row = self.grid[r]
            for c in range(self.size):
                if row[c] == 0:
                    yield (r, c)

    # ----------------------- win detection -----------------------
    def detect_win(
        self, row_index: int, col_index: int, win_length: int, allow_overline: bool
    ) -> Optional[List[Tuple[int, int]]]:
        """Return winning line if the last move produces a win, else None.

        Checks only lines through (row_index, col_index) for the player's stones.
        """
        player_value = self.get_cell(row_index, col_index)
        if player_value == 0:
            return None

        directions: Sequence[Tuple[int, int]] = (
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diag down-right
            (-1, 1),  # diag up-right
        )

        for dr, dc in directions:
            positions = self._collect_line_positions(row_index, col_index, dr, dc, player_value)
            length = len(positions)
            if allow_overline:
                if length >= win_length:
                    return positions
            else:
                if length == win_length:
                    # ensure no overline by checking both ends
                    before_r, before_c = positions[0][0] - dr, positions[0][1] - dc
                    after_r, after_c = positions[-1][0] + dr, positions[-1][1] + dc
                    before_ok = not (self.in_bounds(before_r, before_c) and self.grid[before_r][before_c] == player_value)
                    after_ok = not (self.in_bounds(after_r, after_c) and self.grid[after_r][after_c] == player_value)
                    if before_ok and after_ok:
                        return positions
        return None

    def _collect_line_positions(
        self, row_index: int, col_index: int, dr: int, dc: int, player_value: int
    ) -> List[Tuple[int, int]]:
        """Collect contiguous stones of the same player along (dr, dc) through cell.

        Returns positions sorted from one end to the other.
        """
        result: List[Tuple[int, int]] = [(row_index, col_index)]

        # extend backward
        r, c = row_index - dr, col_index - dc
        while self.in_bounds(r, c) and self.grid[r][c] == player_value:
            result.append((r, c))
            r -= dr
            c -= dc

        # extend forward
        r, c = row_index + dr, col_index + dc
        while self.in_bounds(r, c) and self.grid[r][c] == player_value:
            result.append((r, c))
            r += dr
            c += dc

        result.sort()
        # Sorting naive; for consistent orientation, we sort by (row, col). Good enough for reporting.
        return result

    # ----------------------- rendering -----------------------
    def render(self, symbols: Tuple[str, str, str] = (".", "X", "O"), with_coords: bool = True) -> str:
        empty_symbol, black_symbol, white_symbol = symbols

        def symbol_for(value: int) -> str:
            if value == 0:
                return empty_symbol
            if value == 1:
                return black_symbol
            return white_symbol

        lines: List[str] = []
        if with_coords:
            header = self._column_header()
            lines.append(header)

        for r in range(self.size):
            row_symbols = " ".join(symbol_for(self.grid[r][c]) for c in range(self.size))
            if with_coords:
                label = f"{r+1:>2} "
                lines.append(f"{label}{row_symbols}  {r+1:>2}")
            else:
                lines.append(row_symbols)

        if with_coords:
            lines.append(self._column_header())

        return "\n".join(lines)

    def _column_header(self) -> str:
        letters = [chr(ord('A') + i) for i in range(self.size)]
        # Compact spacing to align with content cells (single char + space)
        return "   " + " ".join(letters)


class GomokuGame:
    """Turn-based engine for Gomoku.

    Provides methods suitable for programmatic play and a CLI wrapper.
    """

    def __init__(self, config: Optional[GomokuConfig] = None) -> None:
        self.config: GomokuConfig = config or GomokuConfig()
        self.board: Board = Board(self.config.board_size)
        self._current_player: Player = self.config.first_player
        self._history: List[Move] = []
        self._redo_stack: List[Move] = []
        self._result: GameResult = GameResult(GameOutcome.ONGOING)

    # ----------------------- public API -----------------------
    @property
    def result(self) -> GameResult:
        return self._result

    @property
    def current_player(self) -> Player:
        return self._current_player

    def reset(self) -> None:
        self.board = Board(self.config.board_size)
        self._current_player = self.config.first_player
        self._history.clear()
        self._redo_stack.clear()
        self._result = GameResult(GameOutcome.ONGOING)

    def play_move(self, row_index: int, col_index: int) -> GameResult:
        """Place a stone for the current player.

        Returns the updated game result. Raises on invalid move or game over.
        """
        if self._result.is_terminal:
            raise GameOverError("game is already over")

        self.board.place_stone(row_index, col_index, self._current_player)
        move = Move(row_index, col_index, self._current_player)
        self._history.append(move)
        self._redo_stack.clear()

        winning_line = self.board.detect_win(
            row_index, col_index, self.config.win_length, self.config.allow_overline
        )
        if winning_line is not None:
            self._result = GameResult(
                outcome=GameOutcome.WIN,
                winner=self._current_player,
                winning_line=winning_line,
                reason="five-in-a-row",
            )
            return self._result

        if self.board.is_full():
            self._result = GameResult(outcome=GameOutcome.DRAW, reason="board is full")
            return self._result

        # Otherwise continue
        self._current_player = self._current_player.opponent
        self._result = GameResult(GameOutcome.ONGOING)
        return self._result

    def undo(self, steps: int = 1) -> int:
        """Undo up to `steps` moves. Returns the number of moves undone."""
        undone = 0
        while steps > 0 and self._history:
            last_move = self._history.pop()
            self.board.remove_stone(last_move.row_index, last_move.col_index)
            self._redo_stack.append(last_move)
            self._current_player = last_move.player
            steps -= 1
            undone += 1

        # Reset result to ongoing after any undo
        if undone > 0:
            self._result = GameResult(GameOutcome.ONGOING)
        return undone

    def redo(self, steps: int = 1) -> int:
        """Redo up to `steps` moves. Returns the number of moves redone."""
        redone = 0
        while steps > 0 and self._redo_stack:
            move = self._redo_stack.pop()
            if not self.board.is_empty(move.row_index, move.col_index):
                # Should not happen if stacks are consistent
                raise GomokuError("cannot redo: cell already occupied")
            self.board.place_stone(move.row_index, move.col_index, move.player)
            self._history.append(move)
            self._current_player = move.player.opponent
            steps -= 1
            redone += 1

        # After redo, recompute terminal if last move ended the game
        if redone > 0:
            last = self._history[-1]
            winning_line = self.board.detect_win(
                last.row_index, last.col_index, self.config.win_length, self.config.allow_overline
            )
            if winning_line is not None:
                self._result = GameResult(GameOutcome.WIN, winner=last.player, winning_line=winning_line)
            elif self.board.is_full():
                self._result = GameResult(GameOutcome.DRAW, reason="board is full")
            else:
                self._result = GameResult(GameOutcome.ONGOING)
        return redone

    def resign(self, player: Optional[Player] = None) -> GameResult:
        if self._result.is_terminal:
            raise GameOverError("game is already over")
        resigning_player = player or self._current_player
        winner = resigning_player.opponent
        self._result = GameResult(GameOutcome.RESIGN, winner=winner, reason=f"{resigning_player.name} resigned")
        return self._result

    def legal_moves(self) -> List[Tuple[int, int]]:
        return list(self.board.legal_moves())

    def last_move(self) -> Optional[Move]:
        return self._history[-1] if self._history else None

    def move_history(self) -> List[Move]:
        return list(self._history)

    def render(self, with_coords: bool = True) -> str:
        return self.board.render(with_coords=with_coords)

    # ----------------------- coordinate helpers -----------------------
    def parse_coord(self, token_a: str, token_b: Optional[str] = None) -> Tuple[int, int]:
        """Parse coordinates.

        Accepts either:
        - two integers as strings (1-based or 0-based). If both in [1..size], interpret as 1-based.
        - algebraic like "H8" (letter + number), case-insensitive, 1-based.
        Returns 0-based (row, col).
        """
        if token_b is not None:
            # Likely two numbers
            try:
                a = int(token_a)
                b = int(token_b)
            except ValueError as exc:
                raise CoordinateParseError("row and col must be integers") from exc
            # Interpret as 1-based if both in range 1..size
            if 1 <= a <= self.board.size and 1 <= b <= self.board.size:
                return a - 1, b - 1
            # Else assume already 0-based
            if 0 <= a < self.board.size and 0 <= b < self.board.size:
                return a, b
            raise CoordinateParseError("coordinates out of range")

        # Single token: algebraic like E5
        token = token_a.strip().upper()
        if len(token) < 2:
            raise CoordinateParseError("algebraic coord requires letter+number, e.g., E5")
        
        # Extract letters prefix and digits suffix
        i = 0
        while i < len(token) and token[i].isalpha():
            i += 1
        letters, digits = token[:i], token[i:]
        if not letters or not digits:
            raise CoordinateParseError("invalid algebraic format")

        try:
            row_one_based = int(digits)
        except ValueError as exc:
            raise CoordinateParseError("row must be an integer in algebraic coord") from exc

        col_index = self._letters_to_index(letters)
        row_index = row_one_based - 1
        if not self.board.in_bounds(row_index, col_index):
            raise CoordinateParseError("algebraic coordinate out of range")
        return row_index, col_index

    def _letters_to_index(self, letters: str) -> int:
        # Base-26 A..Z; only support up to board size
        value = 0
        for ch in letters:
            if not ('A' <= ch <= 'Z'):
                raise CoordinateParseError("invalid column letters")
            value = value * 26 + (ord(ch) - ord('A') + 1)
        index = value - 1
        if index >= self.board.size:
            raise CoordinateParseError("column letter exceeds board size")
        return index


# ----------------------- convenience API -----------------------
def new_game(board_size: int = 15, win_length: int = 5, allow_overline: bool = True) -> GomokuGame:
    return GomokuGame(GomokuConfig(board_size=board_size, win_length=win_length, allow_overline=allow_overline))


# ----------------------- CLI -----------------------
def _print_result(result: GameResult) -> str:
    message: str = ""
    if result.outcome is GameOutcome.WIN:
        line = " ".join(f"({r+1},{c+1})" for r, c in (result.winning_line or []))
        winner_name = result.winner.name if result.winner is not None else "UNKNOWN"
        winner_message = f"Winner: {winner_name} by {result.reason or 'win'}"
        message += winner_message
        print(winner_message)
        if line:
            line_message = f"Line: {line}"
            message += line_message
            print(line_message)

    elif result.outcome is GameOutcome.DRAW:
        message = "Draw: board is full"
        message += message
        print(message)
    elif result.outcome is GameOutcome.RESIGN:
        winner_name = result.winner.name if result.winner is not None else "UNKNOWN"
        message = f"Winner: {winner_name} ({result.reason})"
        message += message
        print(message)
    return message


def _show_help() -> None:
    print(
        """
Commands:
  move r c            Place stone at 1-based row, col (e.g., "move 8 8")
  move e5             Place stone using algebraic coord (e.g., "move H8")
  m ...               Alias for move
  undo [n]            Undo n moves (default 1)
  redo [n]            Redo n moves (default 1)
  show                Print the board
  turn                Show whose turn it is
  resign [B|W]        Current player resigns, or specify B/W
  reset               Reset the game
  new [size win]      Start a new game with optional size and win length
  help                Show this help
  quit/exit           Leave the program
        """.strip()
    )


def _interactive_cli(board_size: int, win_length: int, allow_overline: bool) -> int:
    game = new_game(board_size=board_size, win_length=win_length, allow_overline=allow_overline)
    print("Gomoku started.")
    print(f"Board: {board_size}x{board_size}, need {win_length} in a row. Overline allowed: {allow_overline}")
    print(game.render())
    _show_help()

    while True:
        try:
            prompt = f"[{game.current_player.name}] > " if not game.result.is_terminal else "[done] > "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            return 0

        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0].lower()
        args = tokens[1:]

        try:
            if cmd in {"quit", "exit"}:
                return 0
            elif cmd in {"help", "h", "?"}:
                _show_help()
            elif cmd in {"show", "s"}:
                print(game.render())
            elif cmd == "turn":
                if game.result.is_terminal:
                    print("Game over.")
                else:
                    print(f"Turn: {game.current_player.name}")
            elif cmd in {"move", "m", "play", "p"}:
                if game.result.is_terminal:
                    print("Game is over. Use 'reset' or 'new'.")
                    continue
                if not args:
                    print("Usage: move r c | move e5")
                    continue
                if len(args) == 1:
                    row, col = game.parse_coord(args[0])
                else:
                    row, col = game.parse_coord(args[0], args[1])
                result = game.play_move(row, col)
                print(game.render())
                if result.is_terminal:
                    _print_result(result)
            elif cmd == "undo":
                steps = int(args[0]) if args else 1
                n = game.undo(steps)
                print(f"Undone: {n}")
                print(game.render())
            elif cmd == "redo":
                steps = int(args[0]) if args else 1
                n = game.redo(steps)
                print(f"Redone: {n}")
                print(game.render())
            elif cmd == "resign":
                if args:
                    side = args[0].upper()
                    if side in {"B", "BLACK"}:
                        player = Player.BLACK
                    elif side in {"W", "WHITE"}:
                        player = Player.WHITE
                    else:
                        print("Expected B/W")
                        continue
                else:
                    player = None
                result = game.resign(player)
                _print_result(result)
            elif cmd == "reset":
                game.reset()
                print("Game reset.")
                print(game.render())
            elif cmd == "new":
                size = int(args[0]) if len(args) >= 1 else board_size
                win = int(args[1]) if len(args) >= 2 else win_length
                game = new_game(board_size=size, win_length=win, allow_overline=allow_overline)
                print(f"New game: {size}x{size}, win {win}.")
                print(game.render())
            else:
                print("Unknown command. Type 'help'.")
        except CoordinateParseError as exc:
            print(f"Coordinate error: {exc}")
        except InvalidMoveError as exc:
            print(f"Invalid move: {exc}")
        except GameOverError as exc:
            print(f"Game over: {exc}")
        except ValueError as exc:
            print(f"Value error: {exc}")
        except Exception as exc:  # pragma: no cover - safety net for CLI
            print(f"Error: {exc}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Play Gomoku in the terminal")
    parser.add_argument("--size", type=int, default=15, help="board size (default: 15)")
    parser.add_argument("--win", type=int, default=5, help="stones in a row to win (default: 5)")
    parser.add_argument(
        "--no-overline",
        action="store_true",
        help="require exactly N to win (disallow overlines)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    allow_overline = not args.no_overline
    try:
        return _interactive_cli(args.size, args.win, allow_overline)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


