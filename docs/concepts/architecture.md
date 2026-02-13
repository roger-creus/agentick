# Architecture

Agentick is a modular gridworld framework with protocol-driven architecture for custom tasks, renderers, and multi-modal agents.

## Core Module: agentick/core/

### grid.py: Multi-Layer Grid

```python
grid = Grid(height=10, width=10)
grid.terrain[y, x]  # CellType: EMPTY, WALL, HAZARD, WATER, ICE, HOLE
grid.objects[y, x]  # ObjectType: GOAL, KEY, DOOR, BOX, SWITCH, TOOL
grid.agents[y, x]   # Agent positions
grid.is_walkable((x, y))
grid.flood_fill((x, y))  # Reachable positions
```

### entity.py: Entity System

```python
entity = Entity(id="key_0", entity_type="key", position=(3, 4))
agent = Agent(id="agent_0", position=(1, 1), orientation=Direction.NORTH, inventory=[])
```

### actions.py: Action Space

```python
action_space = ActionSpace(extended=True)
# Basic: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, NOOP, PICKUP, DROP, USE, INTERACT
# Extended: ROTATE_LEFT, ROTATE_RIGHT, MOVE_FORWARD
valid_mask = env.get_valid_actions()  # Binary mask
```

### renderer.py: Multi-Modal Rendering
Observation modes: ASCII, Language, RGB Pixel, State Dict, Human (Pygame)

## Tasks Module: agentick/tasks/

### base.py: TaskSpec Protocol

```python
class TaskSpec(ABC):
    name: str
    capability_tags: list[str]
    difficulty_configs: dict[str, DifficultyConfig]

    @abstractmethod
    def generate(self, seed: int) -> tuple[Grid, dict]:
        pass

    @abstractmethod
    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        pass

    @abstractmethod
    def check_success(self, state) -> bool:
        pass
```

### registry.py: Task Management

```python
from agentick.tasks.registry import make, list_tasks

env = make("GoToGoal-v0", difficulty="medium", render_mode="language", seed=42)
all_tasks = list_tasks()  # All 38 tasks
nav_tasks = list_tasks(capability="navigation")
```

## Generation Module: agentick/generation/

- **maze.py**: Recursive backtracking, cell automata, Kruskal's algorithm
- **room.py**: Binary space partitioning, corridor networks
- **seed.py**: Deterministic RNG for reproducibility
- **validation.py**: Solvability checking, optimal path finding

## Wrappers Module: agentick/wrappers/

```python
from agentick.wrappers import ObservationWrapper, RewardWrapper, RecordingWrapper

class NormalizeObservationWrapper(Wrapper):
    """Normalize observation values."""
    pass
```

## Interfaces

```python
# RL Interface
env = make("GoToGoal-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)

# LLM Interface
env = make("GoToGoal-v0", render_mode="language")
obs, info = env.reset()  # obs is natural language

# VLM Interface
env = make("GoToGoal-v0", render_mode="rgb_array")
obs, info = env.reset()  # obs is (H, W, 3) pixel array

# Bot Interface
env = make("GoToGoal-v0", render_mode="state_dict")
obs, info = env.reset()  # obs is full state dictionary
```

## Benchmark Module: agentick/benchmark/

```python
from agentick.benchmark import make_suite

full_suite = make_suite("full")  # All 38 tasks
quick_suite = make_suite("quick")  # 5 representative tasks
nav_suite = make_suite("navigation")
```

**Baseline Agents**: RandomAgent, GreedyAgent, BFSAgent

## Leaderboard Module: agentick/leaderboard/

- **submission.py**: Submission format
- **evaluator.py**: Run evaluations
- **scoring.py**: Rankings and scores
- **adapters/**: api_adapter, docker_adapter, huggingface_adapter

## Design Patterns

**TaskSpec Protocol**: All tasks implement abstract base for runtime discovery, unified reward computation

**Procedural Generation**: Deterministic seeding
```python
grid, config = task.generate(seed=42)  # Reproducible
```

**Multi-Modal Rendering**: Single environment, multiple observation types
```python
env_lang = make("GoToGoal-v0", render_mode="language")
env_pixels = make("GoToGoal-v0", render_mode="rgb_array")
```

**Action Masking**: Valid actions computed per-state
```python
valid_mask = env.get_valid_actions()
```

**Dense/Sparse Rewards**: Both modes supported
```python
env_sparse = make("GoToGoal-v0", reward_mode="sparse")
env_dense = make("GoToGoal-v0", reward_mode="dense")
```

## Module Responsibilities

| Module | Responsibility |
|--------|---|
| **core/** | Base primitives and rendering |
| **tasks/** | Task definition and registry (38 tasks) |
| **generation/** | Procedural content generation |
| **wrappers/** | Observation/reward middleware |
| **benchmark/** | Evaluation infrastructure |
| **leaderboard/** | Submission & ranking |

## Key APIs

```python
# Environment creation
env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")

# Task info
env.task.name
env.task.capability_tags
env.task.get_optimal_return()

# Observation access
env.render()
env.get_text_observation()
env.get_pixel_observation()
env.get_state_dict()
```

See [Observations](observations.md) and [Tasks](tasks.md) for details.
