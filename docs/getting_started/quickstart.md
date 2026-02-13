# Quickstart

Get started with Agentick in 5 minutes.

## Basic Usage

```python
import agentick

# Create environment
env = agentick.make("GoToGoal-v0", difficulty="easy")

# Reset environment
obs, info = env.reset(seed=42)

# Run episode
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

## Observation Modes

Choose how your agent perceives the environment:

```python
# ASCII visualization
env = agentick.make("GoToGoal-v0", render_mode="ascii")

# Natural language description
env = agentick.make("GoToGoal-v0", render_mode="language")

# Structured language (JSON-like)
env = agentick.make("GoToGoal-v0", render_mode="language_structured")

# RGB pixel array (84x84x3)
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")

# Python dictionary with full state
env = agentick.make("GoToGoal-v0", render_mode="state_dict")
```

## Difficulty Levels

Scale task complexity:

```python
env = agentick.make("GoToGoal-v0", difficulty="easy")    # 5×5 grid, no obstacles
env = agentick.make("GoToGoal-v0", difficulty="medium")  # 10×10 grid, sparse walls
env = agentick.make("GoToGoal-v0", difficulty="hard")    # 15×15 grid, moderate walls
env = agentick.make("GoToGoal-v0", difficulty="expert")  # 20×20 grid, dense walls
```

## Reward Shaping

Choose sparse or dense rewards:

```python
# Sparse: reward only at goal
env = agentick.make("GoToGoal-v0", reward_mode="sparse")

# Dense: shaped reward for progress
env = agentick.make("GoToGoal-v0", reward_mode="dense")
```

## List Available Tasks

```python
from agentick.tasks.registry import list_tasks

# All tasks
tasks = list_tasks()
print(f"Total tasks: {len(tasks)}")

# Filter by capability
nav_tasks = list_tasks(capability="navigation")
memory_tasks = list_tasks(capability="memory")
```

## Command Line

```bash
# List all tasks
uv run agentick list-tasks

# List benchmark suites
uv run agentick list-suites

# Show version
uv run agentick --version
```

## Next Steps

- [Examples](/examples/README.md) - 40+ runnable examples
- [Tasks](/concepts/tasks.md) - Browse all 38 tasks
- [Observations](/concepts/observations.md) - Learn about observation modes
- [RL Training](/agents/rl_agents.md) - Train RL agents
- [LLM Agents](/agents/llm_agents.md) - Use GPT-4o or Claude
