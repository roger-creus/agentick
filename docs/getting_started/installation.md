# Installation

Install Agentick using [uv](https://github.com/astral-sh/uv), the fast Python package installer.

## Quick Install

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/agentick/agentick.git
cd agentick

# Install with all dependencies
uv sync --extra all

# Verify installation
uv run python -c "import agentick; print(agentick.__version__)"
```

## Dependency Groups

Install only what you need:

```bash
# Core library only (minimal dependencies)
uv sync

# Add specific features
uv sync --extra rl          # RL training (torch, wandb)
uv sync --extra llm         # LLM agents (openai, anthropic)
uv sync --extra viz         # Visualization (matplotlib, seaborn)
uv sync --extra docs        # Documentation (mkdocs)

# Everything
uv sync --extra all
```

## What's Included

| Extra | Dependencies | Use Case |
|-------|--------------|----------|
| `rl` | torch, wandb | RL training with PPO/DQN |
| `llm` | openai, anthropic | API-based LLM agents |
| `local-llm` | transformers, torch | Local HuggingFace models |
| `train-llm` | trl, accelerate, peft | Fine-tuning LLMs |
| `viz` | matplotlib, plotly, seaborn | Plotting and visualization |
| `docs` | mkdocs, mkdocs-material | Build documentation |
| `all` | All of the above | Complete installation |

## Next Steps

- [Quickstart Guide](quickstart.md) - Run your first environment
- [Examples](/examples/README.md) - See all runnable examples
- [CLI Reference](/cli.md) - Use the command-line interface
