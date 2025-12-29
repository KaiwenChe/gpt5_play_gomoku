import os
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI
from openai.types.responses import FunctionToolParam, ResponseOutputItem, ResponseInputParam, ResponseOutputText
from openai.types.responses.response_input_param import FunctionCallOutput
from dotenv import load_dotenv

from gomoku import GomokuGame, new_game, Player, _print_result


load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = OpenAI()

    def generate_outcome(self, instructions: str, context: ResponseInputParam, tools: list[FunctionToolParam]) -> list[ResponseOutputItem]:
        response = self.client.responses.create(
            model="gpt-5",
            instructions=instructions,
            input=context,
            tools=tools,
            reasoning={
                "effort": "medium",
                "summary": "auto"
            }
        )
        return response.output


class AgentTool(BaseModel, ABC):
    model_config = ConfigDict(extra='forbid')

    @classmethod
    def tool_name(cls) -> str:
        name = cls.__name__
        # Convert CamelCase or PascalCase to snake_case
        step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1).lower()
        return snake

    @classmethod
    def tool_parameters(cls) -> dict:
        return cls.model_json_schema()

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        pass


class PlayMove(AgentTool):
    col: str = Field(..., description="The column of the move (A-O)")
    row: int = Field(..., description="The row of the move (1-15)")

    def execute(self, game: GomokuGame) -> str:
        if game.result.is_terminal:
            return "Game already over."

        row, col = game.parse_coord(self.col + str(self.row))
        result = game.play_move(row, col)

        output: str = ""
        if result.is_terminal:
            output += _print_result(result)
        return output

    @classmethod
    def get_description(cls) -> str:
        return "Place stone using algebraic coordinates (e.g., 'move H8'). The outcome of the placement will be returned."


class Resign(AgentTool):
    use: bool = Field(..., description="Whether to use the tool")

    def execute(self, game: GomokuGame, player_side: Player) -> str:
        result = game.resign(player_side)
        return _print_result(result)

    @classmethod
    def get_description(cls) -> str:
        return "Resign the game. The outcome of the resignation will be returned."


class GomokuAgent:
    def __init__(self, player_side: Player):
        self.client = LLMClient()
        self.player_side = player_side
        self.instructions = (
            "You are a Gomoku master. \n"
            "You are playing the game with your opponent. \"user\" is the relay between you and the opponent. \n"
            "'.' represents an empty cell, 'X' represents a black stone, 'O' represents a white stone. \n"
            "You are on the side of " + self.player_side.name + ". \n"
            "You need to play the game with available tools provided. \n"
            "Go win the game and good luck!"
        )
        self.tools = self._build_tools()
        self.context_history = []

    def _build_tools(self) -> list[FunctionToolParam]:
        return [FunctionToolParam(
            name=tool.tool_name(),
            description=tool.get_description(),
            parameters=tool.tool_parameters(),
            strict=True,
            type="function",
        ) for tool in [PlayMove, Resign]]

    def play_game(self, game: GomokuGame, system_info: str | None = None) -> str:
        context = ""

        if system_info:
            context += system_info

        context += "\n\n"
        context += game.render()

        self.context_history.append({
            "role": "user",
            "content": context
        })

        print(f"Gomoku game state: \n{context}\n\n")

        tool_usages = self.client.generate_outcome(self.instructions, self.context_history, self.tools)
        self.context_history += tool_usages

        output = ""
        move_info = ""
        for tool_use in tool_usages:
            if tool_use.type == "reasoning":
                reasoning_summary = "\n".join([summary.text for summary in tool_use.summary])
                print(f"{self.player_side.name}'s reasoning: \n{reasoning_summary}\n\n")

            elif tool_use.type == "message":
                message_content = "\n".join([content.text if content is type(ResponseOutputText) else "" for content in tool_use.content])
                print(f"{self.player_side.name}'s message: \n{message_content}\n\n")

            elif tool_use.type == "function_call":
                if tool_use.name == PlayMove.tool_name():
                    play_move = PlayMove.model_validate_json(tool_use.arguments)
                    output = play_move.execute(game)
                    move_info = f"Move: ({play_move.col}, {play_move.row})"
                elif tool_use.name == Resign.tool_name():
                    output = Resign.model_validate_json(tool_use.arguments).execute(game, self.player_side)
                    move_info = "Resigned"
                print(f"{self.player_side.name}'s action: \n{move_info}\n\n")

                output += "\n\n"
                output += game.render()
                print(f"Updated game state: \n{output}\n\n")

                self.context_history.append(FunctionCallOutput(
                    call_id=tool_use.call_id,
                    output=output,
                    type="function_call_output",
                ))

        conclusion = self.client.generate_outcome(self.instructions, self.context_history, self.tools)
        for item in conclusion:
            if item.type == "message":
                message_content = "\n".join([content.text if content is type(ResponseOutputText) else "" for content in item.content])
                print(f"{self.player_side.name}'s message: \n{message_content}\n\n")
            elif item.type == "reasoning":
                reasoning_summary = "\n".join([summary.text for summary in item.summary])
                print(f"{self.player_side.name}'s reasoning: \n{reasoning_summary}\n\n")
            elif item.type == "function_call":
                print(f"WARNING: {self.player_side.name} tries to have consecutive actions. Tool call is ignored.\n\n")

        print("--------------------------------\n\n")
        self.context_history += conclusion
        return move_info


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
