from gomoku import new_game, Player
from gomoku_agent import GomokuAgent


class GomokuGameSession:
    def __init__(self, board_size: int, win_length: int, allow_overline: bool):
        self.game = new_game(board_size, win_length, allow_overline)
        self.black_agent = GomokuAgent(Player.BLACK)
        self.white_agent = GomokuAgent(Player.WHITE)
        self.system_info = (
            "Gomoku started. \n"
            f"Board: {board_size}x{board_size}, need {win_length} in a row. Overline allowed: {allow_overline}"
        )

    def run(self) -> None:
        info = self.system_info

        while not self.game.result.is_terminal:
            if self.game.current_player == Player.BLACK:
                info = self.black_agent.play_game(self.game, info)
            else:
                info = self.white_agent.play_game(self.game, info)

            info = f"{self.game.current_player.name}'s action: {info}"

        print("Game over.")


def main() -> None:
    game_session = GomokuGameSession(board_size=15, win_length=5, allow_overline=True)
    game_session.run()


if __name__ == "__main__":
    main()