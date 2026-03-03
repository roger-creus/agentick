# Quickstart

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Install uv (if needed)
git clone https://github.com/agentick/agentick.git
cd agentick
uv sync --extra all                                  # All dependencies
```

Install only what you need:

```bash
uv sync                     # Core only
uv sync --extra rl          # RL training (torch, wandb)
uv sync --extra llm         # LLM agents (openai, anthropic)
uv sync --extra local-llm   # Local HuggingFace models
uv sync --extra train-llm   # Fine-tuning (trl, peft)
uv sync --extra viz         # Visualization (matplotlib)
uv sync --extra all         # Everything
```

## Basic Usage

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy")
obs, info = env.reset(seed=42)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## Observation Modes

```python
env = agentick.make("GoToGoal-v0", render_mode="ascii")           # Text grid
env = agentick.make("GoToGoal-v0", render_mode="language")         # Natural language
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")        # Isometric pixels (512x512)
env = agentick.make("GoToGoal-v0", render_mode="rgb_array_flat")   # 2D sprites (512x512, fast)
env = agentick.make("GoToGoal-v0", render_mode="state_dict")       # Full state dict
```

## Difficulty Levels

```python
env = agentick.make("GoToGoal-v0", difficulty="easy")    # 5x5 grid
env = agentick.make("GoToGoal-v0", difficulty="medium")  # 10x10 grid
env = agentick.make("GoToGoal-v0", difficulty="hard")    # 15x15 grid
env = agentick.make("GoToGoal-v0", difficulty="expert")  # 20x20 grid
```

## Reward Modes

```python
env = agentick.make("GoToGoal-v0", reward_mode="sparse")  # +1 at goal only
env = agentick.make("GoToGoal-v0", reward_mode="dense")   # Shaped progress reward
```

## List Tasks

```python
from agentick.tasks.registry import list_tasks
tasks = list_tasks()                          # All 38 tasks
nav_tasks = list_tasks(capability="navigation")
```

```bash
uv run agentick list-tasks
uv run agentick list-suites
```

## Next Steps

- [Tasks](../tasks.md) — Browse all 38 tasks
- [Observations](../concepts/observations.md) — Observation mode details
- [RL Agents](../agents/rl_agents.md) — Train RL agents
- [LLM Agents](../agents/llm_agents.md) — Evaluate LLMs
