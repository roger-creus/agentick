# Architecture

Agentick is a modular gridworld framework built around the Gymnasium API. This page covers the key modules and how they fit together.

## Environment Stack

```
agentick.make("GoToGoal-v0")
  → registry looks up TaskSpec subclass
  → TaskSpec.generate(seed) → (Grid, config)
  → TaskEnv(task, grid, config) wraps AgentickEnv
```

`AgentickEnv` (`core/env.py`) owns the Grid, Agent, renderer, and action space. Tasks subclass `TaskSpec` (`tasks/base.py`), and `TaskEnv` (`tasks/registry.py`) bridges them — delegating reward, success checking, and per-step hooks back to the task.

## Core Module: `agentick/core/`

**grid.py** — Multi-layer numpy grid with `terrain`, `objects`, `agents`, `metadata` layers. Positions are `(x, y)` tuples; array indexing is `[y, x]`.

```python
grid = Grid(height=10, width=10)
grid.terrain[y, x]  # CellType: EMPTY, WALL, HAZARD, WATER, ICE, HOLE
grid.objects[y, x]   # ObjectType: GOAL, KEY, DOOR, BOX, SWITCH, TOOL
grid.is_walkable((x, y))
grid.flood_fill((x, y))
```

**entity.py** — `Entity` and `Agent` data classes (position, orientation, inventory).

**actions.py** — `ActionSpace` with basic actions (MOVE\_\*, NOOP, PICKUP, DROP, USE, INTERACT) and optional extended actions (ROTATE\_\*, MOVE\_FORWARD). `env.get_valid_actions()` returns a binary mask.

**renderer.py** — Dispatches to the configured renderer based on `render_mode`:

| Mode | Renderer | Output |
|------|----------|--------|
| `ascii` | ASCIIRenderer | ANSI text grid |
| `language` | EnhancedLanguageRenderer | Natural language |
| `rgb_array` | IsometricRenderer | 512x512 isometric sprites |
| `human` | IsometricRenderer | Same as rgb_array (for display) |
| `state_dict` | StateDictRenderer | Structured numpy dict |

## Tasks Module: `agentick/tasks/`

Tasks across 6 categories: navigation, planning, reasoning, memory, generalization, and multi-agent.

**TaskSpec** defines the task interface:

```python
class TaskSpec(ABC):
    difficulty_configs: dict[str, DifficultyConfig]

    def generate(self, seed: int) -> tuple[Grid, dict]: ...
    def compute_dense_reward(self, old_state, action, new_state, info) -> float: ...
    def check_success(self, state) -> bool: ...
```

**Registry**: `@register_task("Name-v0", tags=[...])` decorator. Task modules auto-imported via category `__init__.py` files.

```python
env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language", seed=42)
all_tasks = agentick.list_tasks()
```

## Generation Module: `agentick/generation/`

- **maze.py**: Recursive backtracking, cellular automata, Kruskal's algorithm
- **room.py**: Binary space partitioning, corridor networks
- **seed.py**: Deterministic RNG for reproducibility
- **validation.py**: Solvability checking, optimal path finding

## Oracle System: `agentick/oracles/`

Hand-coded optimal policies for all tasks. Used for expert trajectory generation, task verification, and score upper bounds. One file per category (`navigation_oracles.py`, `planning_oracles.py`, etc.).

```python
from agentick.oracles import get_oracle, list_oracles

env = agentick.make("GoToGoal-v0", difficulty="hard")
oracle = get_oracle("GoToGoal-v0", env)

obs, info = env.reset(seed=42)
oracle.reset(obs, info)
action = oracle.act(obs, info)
```

Oracles are commonly paired with `DataCollector` to generate training data — see [Fine-Tuning](../agents/finetuning.md).

## Other Modules

| Module | Purpose |
|--------|---------|
| `wrappers/` | Observation/reward middleware (ObservationWrapper, RewardWrapper, RecordingWrapper) |
| `agents/` | LLM/VLM agent harness — backends (OpenAI, Anthropic, HuggingFace) + harness presets. See [LLM Agents](../agents/llm_agents.md) |
| `training/` | SFT, behavior cloning, Tinker trainers. See [Fine-Tuning](../agents/finetuning.md) |
| `data/` | DataCollector for recording trajectories and exporting to HuggingFace format |
| `leaderboard/` | Evaluation suites, scoring, adapters (API, Docker, HuggingFace) |
| `experiments/` | ExperimentRunner for reproducible YAML-based evaluation. See [Experiments](../experiments.md) |

## Key Interfaces

```python
# Standard Gymnasium loop (works with any render_mode)
env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)

# Reward modes
env = agentick.make("GoToGoal-v0", reward_mode="sparse")  # +1 on success
env = agentick.make("GoToGoal-v0", reward_mode="dense")   # Shaped progress reward

# Evaluation suites
from agentick.leaderboard.suites import get_suite
full_suite = get_suite("full")   # All tasks
quick_suite = get_suite("quick") # 5 representative tasks
```
