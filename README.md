# Agentick 🎯

Universal benchmark for evaluating generally capable AI agents across all paradigms: deep RL, LLMs, VLMs, programmatic bots, and humans.

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/anthropics/agentick)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)]()

## Why Agentick?

Agentick is the first comprehensive benchmark designed for **training AND evaluation** of generally capable AI agents. While ARC-AGI-3 focuses on abstract reasoning and AutumnBench evaluates reasoning chains, Agentick provides **40+ gridworld tasks** with **multi-modal observations**, **training-first architecture**, and **capability-decomposed diagnostics** for all agent types.

| Feature | Agentick | ARC-AGI-3 | AutumnBench | MiniGrid | ProcGen |
|---------|----------|-----------|-------------|----------|---------|
| Trainable Agents | ✅ | ❌ | ❌ | ✅ | ✅ |
| Multi-Modal Obs | ✅ | ❌ | ❌ | ❌ | ❌ |
| Capability Profile | ✅ | ❌ | ✅ | ❌ | ❌ |
| All Agent Types | ✅ | ❌ | ❌ | Partial | Partial |
| Procedural Generation | ✅ | ❌ | ❌ | ✅ | ✅ |
| Fast Vectorized Envs | ✅ | ❌ | ❌ | ✅ | ✅ |
| State Validation | ✅ | ✅ | ❌ | ❌ | ❌ |

## Quick Start

```python
import agentick

# Create environment
env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii", seed=42)

# Run episode
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(env.render())
```

## Installation

### With uv (Recommended)

```bash
uv add agentick
```

### With pip

```bash
pip install agentick
```

### From Source

```bash
git clone https://github.com/anthropics/agentick
cd agentick
uv sync  # or: pip install -e .
```

### Optional Dependencies

Agentick has modular optional dependencies for different use cases:

```bash
# For RL training with PyTorch, Weights & Biases
uv add agentick[rl]

# For LLM evaluation (OpenAI, Anthropic, Google APIs)
uv add agentick[llm]

# For local LLMs (Hugging Face Transformers, vLLM)
uv add agentick[local-llm]

# For fine-tuning LLMs (TRL, SFT, DPO pipelines)
uv add agentick[train-llm]

# For visualization and plotting
uv add agentick[viz]

# For documentation generation
uv add agentick[docs]

# All of the above
uv add agentick[all]
```

## Task Gallery

Agentick includes 40+ tasks organized by cognitive capability. Each is **procedurally generated**, **seeded for reproducibility**, and supports **4 difficulty levels** (easy, medium, hard, expert).

### Navigation (5 tasks)
Test spatial reasoning, planning, and exploration under various constraints.

| Task | Description | Difficulty |
|------|-------------|------------|
| **GoToGoal-v0** | Navigate to a visible goal in an open grid | ⭐ Easy |
| **MazeNavigation-v0** | Find path through procedural mazes | ⭐⭐ Medium |
| **MultiGoalRoute-v0** | Visit multiple goals in sequence | ⭐⭐⭐ Hard |
| **DynamicObstacles-v0** | Navigate around moving obstacles | ⭐⭐⭐ Hard |
| **FogOfWar-v0** | Navigate with partial observability | ⭐⭐⭐⭐ Expert |

### Memory (5 tasks)
Test episodic and working memory over long horizons.

| Task | Description | Difficulty |
|------|-------------|------------|
| **KeyDoorPuzzle-v0** | Remember key locations, solve locked doors | ⭐ Easy |
| **SequenceMemory-v0** | Recall hidden sequences | ⭐⭐ Medium |
| **BreadcrumbTrail-v0** | Follow trail of breadcrumbs | ⭐⭐ Medium |
| **DelayedGratification-v0** | Wait to achieve better rewards | ⭐⭐⭐ Hard |
| **BacktrackPuzzle-v0** | Retrace path to unreachable locations | ⭐⭐⭐⭐ Expert |

### Reasoning (5 tasks)
Test logic, pattern matching, and causal inference.

| Task | Description | Difficulty |
|------|-------------|------------|
| **SokobanPush-v0** | Push boxes to goals with physics constraints | ⭐⭐ Medium |
| **SwitchCircuit-v0** | Activate switches in correct sequence | ⭐⭐⭐ Hard |
| **SymbolMatching-v0** | Match visual symbols and patterns | ⭐ Easy |
| **CausalChain-v0** | Identify causal chains of events | ⭐⭐⭐ Hard |
| **RuleInduction-v0** | Learn and apply hidden rules | ⭐⭐⭐⭐ Expert |

### Skill Composition (5 tasks)
Test discovery and composition of multiple skills.

| Task | Description | Difficulty |
|------|-------------|------------|
| **ToolUse-v0** | Use tools to interact with environment | ⭐⭐ Medium |
| **RecipeAssembly-v0** | Combine items in correct sequence | ⭐⭐⭐ Hard |
| **MultiRoomEscape-v0** | Escape multi-room dungeons | ⭐⭐⭐ Hard |
| **ResourceManagement-v0** | Manage limited resources optimally | ⭐⭐⭐⭐ Expert |
| **EmergentStrategy-v0** | Discover emergent strategies | ⭐⭐⭐⭐ Expert |

### Control (4 tasks)
Test precise control, timing, and multi-objective coordination.

| Task | Description | Difficulty |
|------|-------------|------------|
| **PreciseNavigation-v0** | Navigate with position/velocity constraints | ⭐⭐ Medium |
| **TimingChallenge-v0** | Execute actions at precise moments | ⭐⭐⭐ Hard |
| **ChaseEvade-v0** | Chase/evade a moving target | ⭐⭐⭐ Hard |
| **Herding-v0** | Control herd of NPCs | ⭐⭐⭐⭐ Expert |

### Combinatorial (4 tasks)
Test constraint satisfaction and combinatorial optimization.

| Task | Description | Difficulty |
|------|-------------|------------|
| **LightsOut-v0** | Solve classic lights-out puzzle | ⭐⭐ Medium |
| **TileSorting-v0** | Sort tiles into correct order | ⭐⭐⭐ Hard |
| **GraphColoring-v0** | Color graph with minimum colors | ⭐⭐⭐ Hard |
| **PackingPuzzle-v0** | Pack items into constrained space | ⭐⭐⭐⭐ Expert |

### Composition & Abstraction (3 tasks)
Test ability to compose learned behaviors and handle abstraction.

| Task | Description | Difficulty |
|------|-------------|------------|
| **InstructionFollowing-v0** | Execute natural language instructions | ⭐⭐⭐ Hard |
| **ProgramSynthesis-v0** | Generate programs to solve tasks | ⭐⭐⭐⭐ Expert |
| **RecursiveRooms-v0** | Navigate recursively nested structures | ⭐⭐⭐⭐ Expert |

### World Model (3 tasks)
Test ability to learn and generalize world models.

| Task | Description | Difficulty |
|------|-------------|------------|
| **PhysicsDiscovery-v0** | Discover physics rules through interaction | ⭐⭐⭐ Hard |
| **EnvironmentShift-v0** | Adapt to environment changes | ⭐⭐⭐⭐ Expert |
| **RuleDiscoveryNavigation-v0** | Discover navigation rules | ⭐⭐⭐⭐ Expert |

### Adversarial (3 tasks)
Test robustness to adversarial conditions.

| Task | Description | Difficulty |
|------|-------------|------------|
| **NoisyObservation-v0** | Handle noisy observations | ⭐⭐ Medium |
| **DistributionShift-v0** | Adapt to distribution shifts | ⭐⭐⭐⭐ Expert |
| **DeceptiveReward-v0** | Avoid reward hacking | ⭐⭐⭐⭐ Expert |

### Meta-Learning (2 tasks)
Test ability to learn to learn.

| Task | Description | Difficulty |
|------|-------------|------------|
| **FewShotAdaptation-v0** | Adapt quickly to new tasks | ⭐⭐⭐⭐ Expert |
| **TaskInterference-v0** | Multitask without forgetting | ⭐⭐⭐⭐ Expert |

### Multi-Agent (2 tasks)
Test cooperation and competition.

| Task | Description | Difficulty |
|------|-------------|------------|
| **CooperativeTransport-v0** | Cooperate to move objects | ⭐⭐⭐ Hard |
| **CompetitiveTag-v0** | Compete in pursuit-evasion | ⭐⭐⭐⭐ Expert |

## Multi-Modal Observations

Agentick's signature feature: **observe the same task state in 6 different modalities**. Choose the representation that best suits your agent.

### Example: GoToGoal-v0 at position (2,3)

#### 1. ASCII Rendering
```
#####
#A.G#
#.#.#
#...#
#####
```

#### 2. Language Rendering
```
You are in a 5x5 grid room at position (1, 1). The goal is visible to your east at (1, 3).
There are walls forming a cross pattern in the center. You can move north, south, east, or west.
```

#### 3. Language Structured (Dict)
```python
{
    "description": "5x5 grid with agent and goal",
    "position": {"x": 1, "y": 1},
    "orientation": "facing_east",
    "surroundings": {
        "north": "wall",
        "south": "empty",
        "east": "empty",
        "west": "wall"
    },
    "visible_entities": [
        {"type": "goal", "direction": "east", "distance": 2}
    ],
    "inventory": [],
    "energy": 1.0,
    "health": 1.0,
    "valid_actions": ["move_north", "move_south", "move_east", "move_west"],
    "step_count": 0,
    "max_steps": 50
}
```

#### 4. Pixel Rendering (RGB Array)
```
3D numpy array of shape (160, 160, 3) with rendered tiles:
- Agent rendered as blue square
- Goal rendered as green circle
- Walls rendered as dark gray
- Empty cells rendered as light background
- HUD overlay with step counter
```

#### 5. Human Play
```
Press W/A/S/D to move or SPACE to interact
Step 1/50: Moved east, reward +0.0
Visualization window shows current state in 32x32 pixel tiles
```

#### 6. State Dict
```python
{
    "grid": {
        "terrain": np.ndarray shape (5, 5) of CellType values,
        "objects": np.ndarray shape (5, 5) of ObjectType values,
    },
    "agent": {
        "position": (1, 1),
        "entity_type": "agent",
        "id": "agent_0",
    },
    "entities": [
        {
            "type": "goal",
            "position": (1, 3),
            "id": "goal_0",
        }
    ],
    "config": {
        "agent_start": (1, 1),
        "goal_positions": [(1, 3)],
        "max_steps": 50,
    }
}
```

## Agent Support

Agentick works seamlessly with **all agent paradigms**. Choose your observation mode and we handle the rest.

| Agent Type | Best Observation | Wrapper | Example Script |
|-----------|------------------|---------|-----------------|
| Deep RL (PPO/DQN) | `rgb_array` | RLInterface | `examples/rl_training.py` |
| LLM (GPT/Claude) | `language` | LLMAgentInterface | `examples/llm_agent.py` |
| VLM (GPT-4V/Gemini) | `rgb_array` + `language` | VLMAgentInterface | `examples/vlm_agent.py` |
| Programmatic Bot | `state_dict` | BotInterface | `examples/programmatic_bot.py` |
| Human Player | `rgb_array` or `ascii` | HumanInterface | `examples/human_play.py` |
| Search Agent | `state_dict` | SearchInterface | docs/agents/custom_agents.md |
| Curriculum Learning | `rgb_array` | CurriculumWrapper | `examples/curriculum_training.py` |

### Example: RL Agent with Vectorized Environments

```python
import agentick
from agentick.interfaces import RLInterface

# Create vectorized environments for fast training
envs = RLInterface.make_vectorized_env(
    "GoToGoal-v0",
    n_envs=16,
    render_mode="rgb_array",
    difficulty="medium"
)

# Your RL training loop (e.g., with CleanRL PPO)
for update in range(100):
    obs, _ = envs.reset()
    for step in range(256):
        actions = agent(obs)
        obs, rewards, terminated, truncated, info = envs.step(actions)
```

### Example: LLM Agent

```python
import agentick
from anthropic import Anthropic

env = agentick.make("GoToGoal-v0", render_mode="language")
client = Anthropic()

obs, info = env.reset()
for _ in range(50):
    # Get natural language observation
    prompt = f"Current state: {obs}\n\nChoose an action from: {info['valid_actions']}"

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    action_text = response.content[0].text
    action = env.action_space.sample() if "north" in action_text else ...
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

### Example: Programmatic Bot

```python
import agentick
from agentick.interfaces import BotInterface

env = agentick.make("GoToGoal-v0", render_mode="state_dict")
bot = BotInterface(env)

obs, info = env.reset()
for _ in range(50):
    # Access game state programmatically
    agent_pos = bot.get_agent_position()
    goal_pos = bot.get_goal_positions()[0]

    # Compute optimal path
    path = bot.get_shortest_path(goal_pos)

    # Take next step on path
    next_pos = path[0] if path else agent_pos
    action = bot.compute_action_to(next_pos)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Benchmark Suites & Scoring

Agentick includes pre-defined benchmark suites for systematic evaluation. Scores are **normalized** to [0, 1] where 0 is random baseline and 1 is perfect performance.

### Normalized Scoring Formula

```
normalized_score = (agent_performance - random_baseline) / (optimal_performance - random_baseline)
```

Each task provides:
- **Random baseline**: Mean performance of random actions
- **Optimal baseline**: Best known solution performance
- **Sparse rewards**: Success/failure at episode end
- **Dense rewards**: Shaped rewards using potential-based functions

### Available Suites

- **quick**: 5 representative tasks (navigation, memory, reasoning, skill, control)
- **navigation**: 5 navigation tasks
- **memory**: 5 memory tasks
- **reasoning**: 5 reasoning tasks
- **skill**: 5 skill composition tasks
- **control**: 4 control tasks
- **combinatorial**: 4 combinatorial tasks
- **full**: All 40+ tasks

### Example: Running a Benchmark

```python
import agentick
from agentick.benchmark import BenchmarkRunner

# Create suite of tasks
envs = agentick.make_suite("quick", difficulty="medium")

# Run benchmark
results = {}
for env in envs:
    success_count = 0
    for episode in range(10):
        obs, _ = env.reset()
        for _ in range(env.spec.max_episode_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                success_count += 1
                break

    results[env.spec.id] = {
        "success_rate": success_count / 10,
        "episodes": 10,
    }

# Print results
for task_name, metrics in results.items():
    print(f"{task_name}: {metrics['success_rate']:.1%}")
```

See [docs/concepts/scoring.md](docs/concepts/scoring.md) for full methodology including:
- Capability profile computation
- Statistical significance testing
- Per-difficulty scaling
- Sample efficiency metrics

## Leaderboard

Submit your agent to the **Agentick Leaderboard** and compare against state-of-the-art approaches.

**Leaderboard**: [https://leaderboard.agentick.ai](https://leaderboard.agentick.ai) (Coming soon)

### How to Submit

1. Run your agent on the official benchmark suite
2. Generate metrics file with `agentick benchmark submit --config results.json`
3. Submit via web interface with model details
4. Results appear on leaderboard within 24 hours

### Example Top Entries (Placeholder)

| Rank | Agent | Framework | GoToGoal | MazeNav | KeyDoor | Sokoban | Avg Score |
|------|-------|-----------|----------|---------|---------|---------|-----------|
| 1 | GPT-4V + Reasoning | Vision-Language | 98% | 92% | 88% | 75% | 88.3% |
| 2 | PPO + Curriculum | Deep RL | 96% | 94% | 72% | 68% | 82.5% |
| 3 | Claude-3-Opus | LLM | 94% | 88% | 85% | 60% | 81.8% |

See [docs/leaderboard/submitting.md](docs/leaderboard/submitting.md) for detailed submission guide.

## Training Agents

Agentick is optimized for **agent training** with fast vectorized environments, trajectory export, and curriculum learning.

### Deep RL (PPO with CleanRL)

```python
import agentick
import torch
from agentick.interfaces import RLInterface

# Create vectorized environments
envs = RLInterface.make_vectorized_env("GoToGoal-v0", n_envs=8, render_mode="rgb_array")

# Initialize PPO agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = PPOAgent(envs.single_env.observation_space, envs.single_env.action_space, device)

# Training loop
for epoch in range(1000):
    obs, _ = envs.reset()

    for step in range(512):
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(torch.tensor(obs, device=device))

        obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

        # Collect experience and update policy
        agent.store_transition(obs, reward, terminated, truncated, logprob, value)

    agent.update()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Mean Return = {agent.last_mean_return:.2f}")
```

Full example: [examples/rl_training.py](examples/rl_training.py)
Documentation: [docs/agents/rl_agents.md](docs/agents/rl_agents.md)

### LLM Zero-Shot

```python
import agentick
from anthropic import Anthropic

env = agentick.make("GoToGoal-v0", render_mode="language")
client = Anthropic()

# System prompt for reasoning
system_prompt = """You are a highly intelligent agent in a gridworld.
Analyze the current state and choose the best action.
Available actions: move_north, move_south, move_east, move_west, interact.
Return only the action name."""

obs, info = env.reset()
for step in range(100):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=20,
        system=system_prompt,
        messages=[{"role": "user", "content": f"State: {obs}\nAction:"}]
    )

    action_text = response.content[0].text.strip().lower()
    action = {
        "move_north": 0, "move_south": 1, "move_east": 2, "move_west": 3, "interact": 4
    }.get(action_text, env.action_space.sample())

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Full example: [examples/llm_agent.py](examples/llm_agent.py)
Documentation: [docs/agents/llm_agents.md](docs/agents/llm_agents.md)

### Vision-Language Models

```python
import agentick
from openai import OpenAI
import base64

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
client = OpenAI()

obs, info = env.reset()
for step in range(100):
    # Convert image to base64
    import numpy as np
    obs_uint8 = (obs * 255).astype(np.uint8)

    # Query VLM
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What action should the blue square take to reach the green circle? Reply with: north, south, east, or west."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(obs_uint8).decode()}"}}
            ]
        }],
        max_tokens=10
    )

    action_map = {"north": 0, "south": 1, "east": 2, "west": 3}
    action_text = response.choices[0].message.content.lower().strip()
    action = action_map.get(action_text, env.action_space.sample())

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Full example: [examples/vlm_agent.py](examples/vlm_agent.py)
Documentation: [docs/agents/vlm_agents.md](docs/agents/vlm_agents.md)

### Fine-Tuning with RL or SFT

```python
# Collect trajectories from human demonstrations or RL rollouts
trajectories = agentick.collect_trajectories(
    env="GoToGoal-v0",
    agent=expert_agent,
    n_episodes=100,
    export_format="huggingface"
)

# Fine-tune language model on trajectories
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

trainer = SFTTrainer(
    model,
    train_dataset=trajectories,
    tokenizer=tokenizer,
    args=SFTConfig(output_dir="./finetuned_agent", num_train_epochs=3)
)

trainer.train()
```

Full guide: [docs/agents/finetuning.md](docs/agents/finetuning.md)

## Extending Agentick

Create custom tasks by subclassing `TaskSpec`.

### Custom Task Example

```python
import numpy as np
import agentick
from agentick import TaskSpec, register_task
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.configs import DifficultyConfig


@register_task("MyTask-v0", tags=["custom", "navigation"])
class MyCustomTask(TaskSpec):
    """Navigate to randomly placed goals."""

    name = "MyTask-v0"
    description = "Navigate to randomly placed goals with obstacles"
    capability_tags = ["navigation", "custom"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=8, max_steps=50),
        "hard": DifficultyConfig(name="hard", grid_size=16, max_steps=200),
    }

    def generate(self, seed):
        """Generate task instance with random walls and goals."""
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        grid = Grid(size, size)

        # Add perimeter walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Add random internal walls
        for _ in range(size):
            y, x = rng.integers(1, size - 1, 2)
            grid.terrain[y, x] = CellType.WALL

        # Place goal at random location
        while True:
            y, x = rng.integers(1, size - 1, 2)
            if grid.terrain[y, x] == CellType.EMPTY:
                grid.objects[y, x] = ObjectType.GOAL
                break

        return grid, {
            "agent_start": (1, 1),
            "goal_positions": [(y, x)],
            "max_steps": self.difficulty_config.max_steps,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Reward based on distance to goal."""
        if "grid" not in new_state:
            return -0.01

        agent_pos = new_state.get("position", (1, 1))
        goal_pos = self.task_config["goal_positions"][0]
        dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

        # Potential-based shaping: reward moving closer to goal
        return 0.1 if dist < old_state.get("last_dist", float('inf')) else -0.01

    def check_success(self, state):
        """Check if agent reached goal."""
        if "agent" not in state or "grid" not in state:
            return False

        x, y = state["agent"].position
        return state["grid"].objects[y, x] == ObjectType.GOAL


# Use the custom task
if __name__ == "__main__":
    env = agentick.make("MyTask-v0", difficulty="easy")
    obs, info = env.reset(seed=42)
    print(env.render())
```

See [docs/extending/custom_tasks.md](docs/extending/custom_tasks.md) for complete guide including:
- Task validation
- Reward shaping strategies
- Testing custom tasks
- Publishing to registry

## Project Structure

```
agentick/
├── core/                    # Core environment and grid system
│   ├── env.py              # AgentickEnv base class (Gymnasium interface)
│   ├── grid.py             # Grid data structure
│   ├── entity.py           # Agent and Entity classes
│   ├── actions.py          # Action space definition
│   ├── renderer.py         # Multi-modal rendering engine
│   ├── language.py         # Language observation generation
│   └── sprites.py          # Pixel rendering and sprites
├── tasks/                   # Task implementations
│   ├── base.py             # TaskSpec abstract base class
│   ├── registry.py         # Task registration and make()
│   ├── configs.py          # DifficultyConfig definitions
│   ├── navigation/         # Navigation tasks (GoToGoal, MazeNav, etc.)
│   ├── memory/             # Memory tasks (KeyDoor, Sequence, etc.)
│   ├── reasoning/          # Reasoning tasks (Sokoban, Switch, etc.)
│   ├── skill/              # Skill composition tasks
│   ├── control/            # Control and coordination tasks
│   ├── combinatorial/      # Combinatorial optimization tasks
│   ├── compositional/      # Abstraction and composition tasks
│   ├── worldmodel/         # World model learning tasks
│   ├── adversarial/        # Adversarial/robustness tasks
│   ├── meta/               # Meta-learning tasks
│   └── multi_agent/        # Multi-agent coordination tasks
├── generation/             # Procedural generation utilities
│   ├── maze.py            # Maze generation algorithms
│   ├── validation.py      # Instance validation and solvability
│   └── pathfinding.py     # Pathfinding utilities
├── interfaces/            # Agent interfaces
│   ├── rl_interface.py    # Deep RL interface with vectorization
│   ├── llm_interface.py   # LLM agent interface
│   ├── vlm_interface.py   # Vision-language model interface
│   ├── bot_interface.py   # Programmatic bot interface
│   └── human_interface.py # Human play interface
└── utils/                 # Utilities
    ├── metrics.py         # Scoring and evaluation
    ├── trajectory.py      # Trajectory recording and export
    └── rendering.py       # Additional rendering utilities

docs/                       # Documentation
├── index.md               # Main documentation index
├── getting_started/       # Installation, quickstart, first experiment
├── concepts/              # Architecture, tasks, observations, scoring
├── agents/                # Agent training guides
├── extending/             # Custom task development
├── experiments/           # Running and analyzing experiments
├── leaderboard/           # Leaderboard documentation
├── api/                   # Auto-generated API reference
└── faq.md                # Frequently asked questions

examples/                  # Example scripts
├── random_agent.py       # Simple random exploration
├── rl_training.py        # Deep RL training with PPO
├── llm_agent.py          # LLM zero-shot evaluation
├── vlm_agent.py          # Vision-language model evaluation
├── programmatic_bot.py   # Rule-based bot
├── human_play.py         # Interactive human play
├── custom_task.py        # Creating custom tasks
└── curriculum_training.py # Curriculum learning setup

tests/                     # Test suite
├── test_env.py           # Environment tests
├── test_tasks.py         # Task registration and generation
├── test_observations.py  # Observation mode tests
├── test_agents.py        # Agent interface tests
└── test_integration.py   # End-to-end integration tests
```

## Citation

If you use Agentick in your research, please cite:

```bibtex
@software{agentick2025,
  title={Agentick: A Comprehensive Benchmark for Evaluating Generally Capable AI Agents},
  author={Agentick Team},
  year={2025},
  url={https://github.com/anthropics/agentick}
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork and branch**: Create a feature branch from `main`
2. **Code style**: Run `uv run ruff format` and `uv run ruff check --fix`
3. **Tests**: All new code must have tests. Run `uv run pytest tests/ -v`
4. **Docs**: Update relevant documentation in `docs/`
5. **Commit**: Write clear commit messages
6. **PR**: Submit pull request with description of changes

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

### Development Setup

```bash
# Clone and install
git clone https://github.com/anthropics/agentick
cd agentick
uv sync --all-extras

# Run tests
uv run pytest tests/ -v --timeout=300

# Check coverage
uv run pytest --cov=agentick --cov-report=term-missing

# Lint and format
uv run ruff check agentick/ --fix
uv run ruff format agentick/
uv run mypy agentick/ --ignore-missing-imports

# Build docs
uv run mkdocs build
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Questions?** Check the [FAQ](docs/faq.md) or open an issue on [GitHub](https://github.com/anthropics/agentick/issues).

**Want to stay updated?** Star us on GitHub and follow for releases.

**Ready to benchmark your agent?** Start with [Installation](docs/getting_started/installation.md) and [Quick Start](docs/getting_started/quickstart.md).
