# Architecture

Agentick is a comprehensive gridworld framework with modular, protocol-driven architecture enabling seamless integration of custom tasks, renderers, and multi-modal agents.

## System Overview

High-level component hierarchy:

```
                    ┌─────────────────────────────────────┐
                    │    User Applications / Agents       │
                    │  (RL, LLM, VLM, Bot, Human)        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │   Gymnasium-Compatible API          │
                    │  (reset, step, render, close)       │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
    ┌───▼────────┐      ┌──────────▼─────────┐    ┌──────────▼───────┐
    │   Core     │      │   Tasks &          │    │  Observation     │
    │  Module    │      │   Generation       │    │   Renderers      │
    └────────────┘      └────────────────────┘    └──────────────────┘
        │                       │                         │
        │ • Grid               │ • TaskSpec              │ • ASCII
        │ • Entity             │ • Registry              │ • Language
        │ • Agent              │ • Procedural Gen        │ • RGB Pixel
        │ • Renderer           │ • Validation            │ • State Dict
        │ • Actions            │ • Difficulty Config     │ • Human (Pygame)
        │ • Language           │                         │
        │                      │                         │
        └──────────────────────┴─────────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
    ┌───▼──────────────┐    ┌───────────▼────────┐
    │   Storage        │    │   Utilities        │
    ├──────────────────┤    ├────────────────────┤
    │ • Leaderboard    │    │ • Wrappers         │
    │ • Recording      │    │ • Benchmarking     │
    │ • Evaluation     │    │ • Logging          │
    └──────────────────┘    └────────────────────┘
```

## Data Flow: User → Environment → Observation

Complete execution flow for a single step:

```
User Action (int)
      │
      ▼
┌─────────────────────┐
│ Environment.step()  │ Input validation, action conversion
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Execute Action      │ Update grid, entity positions
│ (movement,          │ Check validity constraints
│  interaction)       │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Compute Reward      │ Task-specific reward function
│ (sparse/dense)      │ Dense: progress-based shaping
│                     │ Sparse: success boolean
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Check Success       │ Task success condition
└─────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Render Observation              │
│ ┌─────────────────────────────┐ │
│ │ Mode Selected:              │ │
│ │ • ASCII (colored grid text) │ │
│ │ • Language (natural desc)   │ │
│ │ • RGB (pixel array)         │ │
│ │ • State Dict (full state)   │ │
│ │ • Human (pygame window)     │ │
│ └─────────────────────────────┘ │
└──────────────────┬──────────────┘
                   │
                   ▼
      (obs, reward, terminated, truncated, info)
```

## Core Module: agentick/core/

Foundation layer providing base primitives:

### grid.py: Multi-Layer Grid
2D gridworld representation with four layers:

```python
grid = Grid(height=10, width=10)

# Layer 1: terrain - Cell types
grid.terrain[y, x]  # CellType: EMPTY, WALL, HAZARD, WATER, ICE, HOLE

# Layer 2: objects - Collectibles and interactive items
grid.objects[y, x]  # ObjectType: GOAL, KEY, DOOR, BOX, SWITCH, TOOL, RESOURCE, BREADCRUMB

# Layer 3: agents - Agent positions
grid.agents[y, x]   # Agent count at position

# Layer 4: metadata - Custom per-cell data
grid.metadata[y, x] # User-defined data

# Key methods:
grid.is_walkable((x, y))        # Can agent move here?
reachable = grid.flood_fill((x, y))  # Get all reachable positions
neighbors = grid.get_neighbors((x, y))  # Get adjacent cells
```

### entity.py: Entity System
Base classes for gridworld objects:

```python
# Entity: Any object in the world
entity = Entity(
    id="key_0",
    entity_type="key",
    position=(3, 4),
    properties={"color": "gold"}
)

# Agent: Entity with agent-specific capabilities
agent = Agent(
    id="agent_0",
    entity_type="agent",
    position=(1, 1),
    orientation=Direction.NORTH,  # What direction agent faces
    inventory=[],  # Carried items
    energy=1.0,    # Energy level (0.0-1.0)
    health=1.0     # Health level (0.0-1.0)
)

# Orientation directions: NORTH, SOUTH, EAST, WEST
agent.orientation.rotate_left()   # Turn left
agent.orientation.to_delta()      # Get (dx, dy) movement
```

### actions.py: Action Space
Discrete action space with action masking:

```python
action_space = ActionSpace(extended=True)

# Basic actions (9)
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, NOOP
PICKUP, DROP, USE, INTERACT

# Extended actions (12) - includes orientation
ROTATE_LEFT, ROTATE_RIGHT, MOVE_FORWARD

# Get action mask for current state (which actions are valid)
valid_mask = env.get_valid_actions()  # Binary mask [1, 0, 1, ...]
valid_names = info["valid_actions"]   # ["move_up", "pickup", ...]

# Convert between representations
action_type = action_space.get_action_type(action_idx)  # int -> ActionType
action_name = action_space.get_action_name(action_idx)  # int -> str
action_idx = action_space.get_action_index("move_up")  # str -> int
```

### renderer.py: Multi-Modal Rendering
Observation generation in different formats (see Observations section)

### language.py: Natural Language Engine
Rich language descriptions with:
- Perspective: first-person, third-person, omniscient
- Verbosity: minimal, standard, verbose
- Features: spatial reasoning, memory, threats, goals

## Tasks Module: agentick/tasks/

Task definition and discovery system:

### base.py: TaskSpec Protocol
Abstract base class for all tasks. Defines contract for task implementations:

```python
class TaskSpec(ABC):
    # Metadata
    name: str  # Task identifier (e.g., "GoToGoal-v0")
    description: str  # Human description
    capability_tags: list[str]  # What this tests: ["navigation", "planning"]

    # Difficulty configurations
    difficulty_configs: dict[str, DifficultyConfig]  # easy/medium/hard/expert

    def __init__(self, difficulty: str = "medium", **kwargs):
        """Initialize task with difficulty level."""
        pass

    @abstractmethod
    def generate(self, seed: int) -> tuple[Grid, dict]:
        """Generate task instance with given seed.

        Returns: (grid, config) where config contains:
        - agent_start: Agent starting position
        - goal_positions: Positions of goals
        - max_steps: Episode length limit
        - Task-specific config
        """
        pass

    @abstractmethod
    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        """Compute shaped dense reward for transition."""
        pass

    def compute_sparse_reward(self, old_state, action, new_state, info) -> float:
        """Compute sparse reward (1.0 on success, 0.0 otherwise)."""
        pass

    @abstractmethod
    def check_success(self, state) -> bool:
        """Determine if task is complete."""
        pass

    def validate_instance(self, grid: Grid, config: dict) -> bool:
        """Verify generated instance is solvable."""
        pass

    def get_optimal_return(self, difficulty=None) -> float:
        """Best possible score for this task."""
        pass

    def get_random_baseline(self, difficulty=None) -> float:
        """Expected score for random agent."""
        pass
```

### registry.py: Task Management
Central registry for task discovery and instantiation:

```python
from agentick.tasks.registry import make, list_tasks, get_task_class

# Create environment for task
env = make(
    "GoToGoal-v0",
    difficulty="medium",
    render_mode="language",
    reward_mode="sparse",
    seed=42
)

# List available tasks
all_tasks = list_tasks()  # All 41 tasks
nav_tasks = list_tasks(capability="navigation")  # Filter by tag
easy_tasks = list_tasks(difficulty="easy")  # Filter by difficulty

# Get task class
TaskClass = get_task_class("GoToGoal-v0")
task = TaskClass(difficulty="medium")
grid, config = task.generate(seed=42)

# Create task suites for benchmarking
quick_env = make_suite("quick")  # 5 representative tasks
nav_envs = make_suite("navigation")  # All navigation tasks
full_envs = make_suite("full")  # All 41 tasks
```

### configs.py: Configuration Models
Data classes for task parameters:

```python
from agentick.tasks.configs import DifficultyConfig

config = DifficultyConfig(
    name="medium",
    grid_size=10,  # Grid dimensions (square)
    max_steps=50,  # Episode length
    params={"wall_density": 0.1}  # Task-specific parameters
)
```

## Generation Module: agentick/generation/

Procedural content generation:

### maze.py: Maze Generation
Algorithms for maze generation:
- Recursive backtracking
- Cell automata
- Kruskal's algorithm

### room.py: Room-Based Generation
Level generation using:
- Binary space partitioning (BSP)
- Corridor networks
- Room connectivity

### seed.py: Seeding & RNG
Deterministic random number generation for reproducibility

### validation.py: Instance Validation
```python
from agentick.generation.validation import verify_solvable, find_optimal_path

# Check if instance has solution
is_valid = verify_solvable(grid, agent_start, goal_positions)

# Find shortest path
path, length = find_optimal_path(grid, agent_start, goal_positions)
```

## Wrappers Module: agentick/wrappers/

Middleware for observation and reward transformation:

### observation.py: Observation Wrappers
```python
from agentick.wrappers import ObservationWrapper

class NormalizeObservationWrapper(Wrapper):
    """Normalize observation values to [0, 1]."""
    pass

wrapped = NormalizeObservationWrapper(env)
```

### reward.py: Reward Wrappers
```python
class ScaleRewardWrapper(Wrapper):
    """Scale reward by constant factor."""
    pass

class ClipRewardWrapper(Wrapper):
    """Clip reward to [-1, 1]."""
    pass
```

### recording.py: Episode Recording
```python
class RecordingWrapper(Wrapper):
    """Record all episodes to disk."""
    pass

wrapped = RecordingWrapper(env, output_dir="./episodes/")
```

## Interfaces Module: agentick/interfaces/

Protocol definitions for different agent types:

### RL Interface (Standard Gymnasium)
```python
env = make("GoToGoal-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)
for step in range(max_steps):
    action = policy(obs)  # Your RL policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### LLM Interface (Language Observations)
```python
env = make("GoToGoal-v0", render_mode="language")
obs, info = env.reset()  # obs is natural language string

# Format prompt with observation and valid actions
prompt = f"{obs}\nValid actions: {info['valid_actions']}\nWhat action?"
action_name = llm_agent.get_action(prompt)
action_idx = action_space.get_action_index(action_name)
obs, reward, _, _, info = env.step(action_idx)
```

### VLM Interface (Pixel Observations)
```python
env = make("GoToGoal-v0", render_mode="rgb_array")
obs, info = env.reset()  # obs is (H, W, 3) pixel array

# Pass pixels to vision model
action = vlm_agent.get_action(
    image=obs,
    valid_actions=info["valid_actions"]
)
obs, reward, _, _, info = env.step(action)
```

### Bot Interface (State Dict)
```python
env = make("GoToGoal-v0", render_mode="state_dict")
obs, info = env.reset()  # obs is full state dictionary

# Access state programmatically
state = obs
agent_pos = state["agent"]["position"]
goal_positions = state["info"]["goal_positions"]
grid = state["grid"]

# Use pathfinding or planning
action = pathfind_to_goal(state, goal_positions)
obs, _, _, _, _ = env.step(action)
```

### Human Interface (Pygame UI)
```python
env = make("GoToGoal-v0", render_mode="human")
obs, info = env.reset()
# Opens pygame window for human control
```

## Benchmark Module: agentick/benchmark/

Evaluation infrastructure:

### suites.py: Benchmark Suites
Pre-defined suites of tasks for evaluation:

```python
from agentick.benchmark import make_suite

# Full benchmark: all 41 tasks
full_suite = make_suite("full")

# Quick eval: 5 representative tasks
quick_suite = make_suite("quick")
# GoToGoal, MazeNavigation, KeyDoorPuzzle, SokobanPush, PreciseNavigation

# Category suites
nav_suite = make_suite("navigation")  # 5 navigation tasks
memory_suite = make_suite("memory")  # 4 memory tasks
reasoning_suite = make_suite("reasoning")  # 5 reasoning tasks
```

### baselines.py: Baseline Agents
Reference implementations:
- RandomAgent: Uniformly random actions
- GreedyAgent: Path-following heuristics
- BFSAgent: Breadth-first search solver
- DFSAgent: Depth-first search solver

### metrics.py: Evaluation Metrics
- Success rate
- Return (cumulative reward)
- Efficiency (steps to solve)
- Learning curves

## Leaderboard Module: agentick/leaderboard/

Submission and ranking system:

### submission.py: Submission Format
Define agent submission format and requirements

### evaluator.py: Run Evaluations
Execute evaluations on submitted agents

### scoring.py: Score Computation
Compute rankings and aggregate scores

### adapters/
Support for different frameworks:
- **api_adapter.py**: HTTP endpoint agents
- **docker_adapter.py**: Containerized agents
- **huggingface_adapter.py**: HuggingFace model hub agents

## Design Patterns

### 1. TaskSpec Protocol
All tasks implement abstract `TaskSpec` base class:
- Enables runtime discovery via registry
- Unified reward/success computation
- Procedural generation with difficulty scaling
- Instance validation

### 2. Procedural Generation
Deterministic seeding ensures reproducibility:
```python
grid1, config1 = task.generate(seed=42)  # Same result
grid2, config2 = task.generate(seed=42)  # Same result
grid3, config3 = task.generate(seed=43)  # Different
```

### 3. Multi-Modal Rendering
Single environment supports multiple observation types:
```python
# Change at environment creation
env_ascii = make("GoToGoal-v0", render_mode="ascii")
env_lang = make("GoToGoal-v0", render_mode="language")
env_pixels = make("GoToGoal-v0", render_mode="rgb_array")
env_state = make("GoToGoal-v0", render_mode="state_dict")

# Or access programmatically
obs_text = env.get_text_observation()
obs_pixels = env.get_pixel_observation()
obs_state = env.get_state_dict()
```

### 4. Action Masking
Valid actions computed per-state:
```python
valid_mask = env.get_valid_actions()  # [1, 0, 1, 0, ...] binary mask

# Use in policy
masked_logits = logits - 1e8 * (1 - valid_mask)  # Zero out invalid actions
action = np.argmax(masked_logits)
```

### 5. Dense/Sparse Rewards
Tasks support both modes:
```python
env_sparse = make("GoToGoal-v0", reward_mode="sparse")
# Only reward on success: 1.0 when goal reached, 0.0 otherwise

env_dense = make("GoToGoal-v0", reward_mode="dense")
# Shaped reward: -0.01 per step + 0.1 * distance_reduction
```

## Extension Points

### Creating Custom Tasks
Implement `TaskSpec` for new tasks:

```python
from agentick.tasks.base import TaskSpec
from agentick.tasks.registry import register_task
from agentick.tasks.configs import DifficultyConfig

@register_task("MyTask-v0", tags=["custom", "example"])
class MyCustomTask(TaskSpec):
    name = "MyTask-v0"
    description = "My custom task description"
    capability_tags = ["custom", "example"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=5, max_steps=20),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=50),
        "hard": DifficultyConfig(name="hard", grid_size=15, max_steps=100),
    }

    def generate(self, seed: int) -> tuple[Grid, dict]:
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size

        grid = Grid(size, size)
        # Populate grid with walls, objects, etc.

        return grid, {
            "agent_start": (1, 1),
            "goal_positions": [(size-2, size-2)],
            "max_steps": self.get_max_steps(),
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        return -0.01  # Step penalty

    def check_success(self, state):
        agent = state["agent"]
        grid = state["grid"]
        x, y = agent.position
        return grid.objects[y, x] == ObjectType.GOAL
```

Create environment:
```python
env = make("MyTask-v0", difficulty="medium")
```

### Custom Renderers
Implement `Renderer` protocol:

```python
class CustomRenderer:
    def render(self, grid: Grid, entities: list[Entity], agent: Agent, info: dict):
        # Your rendering logic
        return your_observation_format
```

### Custom Wrappers
Transform observations/rewards:

```python
from gymnasium import Wrapper

class MyCustomWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = transform_observation(obs)
        reward = transform_reward(reward)
        return obs, reward, terminated, truncated, info

wrapped_env = MyCustomWrapper(env)
```

## Module Responsibilities

| Module | Responsibility | Key Files |
|--------|---|---|
| **core/** | Base primitives and rendering | grid.py, entity.py, actions.py, renderer.py, language.py |
| **tasks/** | Task definition and registry | base.py, registry.py, configs.py, 41 task implementations |
| **generation/** | Procedural content generation | maze.py, room.py, seed.py, validation.py |
| **wrappers/** | Observation/reward middleware | observation.py, reward.py, recording.py |
| **interfaces/** | Multi-modal agent support | rl.py, llm.py, vlm.py, bot.py, human.py |
| **benchmark/** | Evaluation infrastructure | suites.py, baselines.py, metrics.py |
| **leaderboard/** | Submission & ranking | submission.py, evaluator.py, scoring.py, adapters/ |

## Key APIs

### Gymnasium Interface
```python
env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### Task Management
```python
env.task.name              # Task identifier
env.task.difficulty        # Current difficulty
env.task.capability_tags   # What this tests
env.task.get_optimal_return()     # Best possible score
env.task.get_random_baseline()    # Random agent score
```

### Observation Access
```python
env.render()              # Current render mode
env.get_text_observation()    # Force text
env.get_pixel_observation()   # Force pixels
env.get_state_dict()          # Force state dict
```

## Performance Considerations

- **Fast mode**: Use `fast_mode=True` to disable expensive state_dict conversions
- **Rendering speed**: ASCII < Language < Pixel (faster to slower)
- **Grid size**: Linear scaling with environment size
- **Caching**: Renderer caches computed results when state unchanged
- **Batch processing**: Use `gymnasium.vector.AsyncVectorEnv` for parallel evaluation

## Thread Safety

- Each environment instance is independent
- Environments are NOT thread-safe by default
- Use `gymnasium.vector.AsyncVectorEnv` for safe parallel execution

See [Observations](observations.md) for detailed rendering modes and [Tasks](tasks.md) for the complete task catalog.
