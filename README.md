# GPT-5 Plays Gomoku

Watch two GPT-5 agents battle it out in Gomoku (Five-in-a-Row). Each AI reasons through its moves using OpenAI's Responses API with extended thinking capabilities.

## What's Inside

- **Gomoku Engine** (`gomoku.py`) — Full game logic with win detection, undo/redo, and coordinate parsing
- **AI Agents** (`gomoku_agent.py`) — GPT-5 powered players that use tool calls to make moves
- **Game Session** (`main.py`) — Orchestrates a match between Black and White AI agents

The agents receive the board state, reason about strategy (with reasoning summaries printed to console), and execute moves via function calling.

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key with GPT-5 access

### Setup

**Option A: Using uv (recommended)**

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the game (uv handles everything)
uv run main.py
```

**Option B: Using pip**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install openai python-dotenv
python main.py
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## What You'll See

When you run the game, you'll see:

1. The current board state
2. Each agent's reasoning summary (what it's thinking)
3. The move it decides to make
4. Updated board after each turn

The game continues until one agent wins (5 in a row) or resigns.

## Playing Gomoku Yourself

You can also play interactively against the terminal:

```bash
uv run gomoku.py --size 15 --win 5
```

Commands: `move H8`, `undo`, `redo`, `resign`, `help`
