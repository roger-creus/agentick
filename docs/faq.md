# Frequently Asked Questions (FAQ)

Comprehensive answers to common questions about Agentick.

## Getting Started

### What is Agentick?

Agentick is a universal benchmark for evaluating AI agents across all paradigms. It provides 40+ procedurally generated gridworld tasks with multi-modal observations, making it suitable for training and evaluating deep RL agents, LLMs, vision-language models, programmatic bots, and human players.

### How is Agentick different from other benchmarks like ARC-AGI-3, MiniGrid, or ProcGen?

Key differences:

| Feature | Agentick | ARC-AGI-3 | MiniGrid | ProcGen |
|---------|----------|-----------|----------|---------|
| **Trainable agents** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Multi-modal observations** | ✅ Yes (6 modes) | ❌ No | ❌ No | ❌ No |
| **All agent types** | ✅ RL, LLM, VLM, bot, human | ❌ Limited | ⚠️ Partial | ⚠️ Partial |
| **Capability profile** | ✅ Detailed breakdown | ❌ No | ❌ No | ❌ No |
| **Training-first design** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Reproducibility** | ✅ Seeded generation | ⚠️ Partial | ✅ Yes | ✅ Yes |
| **Leaderboard** | ✅ Community-run | ❌ No | ❌ No | ⚠️ Limited |

**In short**: Agentick is the first benchmark designed for *both* training *and* evaluation of generally capable agents across all paradigms, with rich multi-modal observations and detailed diagnostic profiles.

### Is Agentick free and open source?

Yes! Agentick is completely open source under the MIT license. You can use it for research, commercial projects, and everything in between.

### What Python versions are supported?

Python 3.11, 3.12, and 3.13. Earlier versions (3.10 and below) are not supported due to type annotation features used in the codebase.

## Installation & Setup

### I'm getting dependency conflicts. What should I do?

Agentick has modular optional dependencies. Try installing without optional dependencies first:

```bash
uv sync
```

If you need specific features, install them separately:

```bash
# For RL training with PyTorch
uv sync --extra rl

# For LLM evaluation
uv sync --extra llm

# For all features
uv sync --extra all
```

If you still have conflicts, use `uv` which has better dependency resolution:

```bash
uv add agentick
uv add agentick[rl,llm,viz]
```

### Can I use Agentick on a GPU?

Yes! The core Agentick library runs on CPU or GPU. For GPU support with optional dependencies:

```bash
# Install Agentick with RL support (includes PyTorch)
uv sync --extra rl
```

Then use GPU in your training code:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### I'm getting pygame display errors. How do I fix this?

In headless environments (servers, Docker), set:

```bash
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
```

Then Agentick will work without a display:

```python
import agentick
env = agentick.make("GoToGoal-v0", render_mode="ascii")
# Works fine even without display
```

### What are the system requirements?

**Minimum**:
- Python 3.11+
- 2GB RAM
- 500MB disk space

**Recommended for training**:
- Python 3.12+
- 16GB+ RAM
- GPU with 8GB+ VRAM (NVIDIA CUDA 11.8+)
- 10GB disk space for models

## Core Concepts

### What's the difference between difficulty levels?

Each task has 4 difficulty levels: easy, medium, hard, expert. They scale in:

- **Grid size**: Small (easy) to large (expert)
- **Obstacle density**: Minimal (easy) to dense (expert)
- **Episode length**: Short time limits (easy) to long (expert)
- **Distraction/noise**: Simple (easy) to complex (expert)

Example for GoToGoal-v0:

```python
# Easy: 5x5 grid, 20 steps, no obstacles
env_easy = agentick.make("GoToGoal-v0", difficulty="easy")

# Expert: 20x20 grid, 200 steps, 25% obstacles
env_expert = agentick.make("GoToGoal-v0", difficulty="expert")
```

### What observation modes are available?

Agentick supports 6 observation modes for the same task:

1. **ASCII**: Text grid representation
2. **Language**: Natural language description
3. **Language Structured**: JSON/dict with state details
4. **RGB Array**: Pixel observations (3D numpy array)
5. **State Dict**: Full internal state representation
6. **Human**: Interactive display with keyboard control

```python
# Use different observation modes
env_ascii = agentick.make("GoToGoal-v0", render_mode="ascii")
env_pixels = agentick.make("GoToGoal-v0", render_mode="rgb_array")
env_text = agentick.make("GoToGoal-v0", render_mode="language")
env_state = agentick.make("GoToGoal-v0", render_mode="state_dict")
```

### How do I choose which observation mode for my agent?

- **LLM agents**: Use `language` or `language_structured` for text-based reasoning
- **Vision-language agents**: Use `rgb_array` with text descriptions
- **Deep RL agents**: Use `rgb_array` for pixel-based learning
- **Programmatic bots**: Use `state_dict` for direct state access
- **Human play**: Use `human` for interactive play

### What does "seeded" mean?

Seeded means the random generation is deterministic. With the same seed, you always get the same task instance:

```python
# Same seed = same task
env1 = agentick.make("GoToGoal-v0", seed=42)
env2 = agentick.make("GoToGoal-v0", seed=42)
# env1 and env2 are identical

# Different seed = different task
env3 = agentick.make("GoToGoal-v0", seed=43)
# env3 is different from env1 and env2
```

This is critical for reproducibility in research.

## Task Details

### How many tasks are there?

Agentick includes 40+ tasks organized into 11 categories:

- Navigation (5 tasks): GoToGoal, MazeNavigation, MultiGoalRoute, DynamicObstacles, FogOfWar
- Memory (5 tasks): KeyDoorPuzzle, SequenceMemory, BreadcrumbTrail, DelayedGratification, BacktrackPuzzle
- Reasoning (5 tasks): SokobanPush, SwitchCircuit, SymbolMatching, CausalChain, RuleInduction
- Skill Composition (5 tasks): ToolUse, RecipeAssembly, MultiRoomEscape, ResourceManagement, EmergentStrategy
- Control (4 tasks): PreciseNavigation, TimingChallenge, ChaseEvade, Herding
- Combinatorial (4 tasks): LightsOut, TileSorting, GraphColoring, PackingPuzzle
- Compositional (3 tasks): InstructionFollowing, ProgramSynthesis, RecursiveRooms
- Adversarial (3 tasks): NoisyObservation, DistributionShift, DeceptiveReward
- Meta-Learning (2 tasks): FewShotAdaptation, TaskInterference
- Multi-Agent (2 tasks): CooperativeTransport, CompetitiveTag

See [docs/concepts/tasks.md](concepts/tasks.md) for complete task descriptions.

### How do I list all tasks programmatically?

```python
import agentick

# List all tasks
all_tasks = agentick.list_tasks()
print(all_tasks)
# Output: ['GoToGoal-v0', 'MazeNavigation-v0', ...]

# Filter by capability
navigation_tasks = agentick.list_tasks(capability="navigation")
# Output: ['GoToGoal-v0', 'MazeNavigation-v0', ...]

# Filter by difficulty
hard_tasks = agentick.list_tasks(difficulty="hard")
```

Or via CLI:

```bash
uv run agentick list
uv run agentick list --capability navigation
uv run agentick list --difficulty hard
```

### How do I get detailed information about a task?

```python
import agentick

# Create environment
env = agentick.make("GoToGoal-v0", difficulty="medium")

# Access task info
print(env.spec.id)  # "GoToGoal-v0"
print(env.spec.max_episode_steps)  # 50
print(env.observation_space)  # Box(...) for rgb_array
print(env.action_space)  # Discrete(5) - 5 actions
```

Or via CLI:

```bash
uv run agentick info GoToGoal-v0
uv run agentick info GoToGoal-v0 --difficulty hard
uv run agentick info GoToGoal-v0 --format json
```

## Running Benchmarks

### How do I run a single benchmark?

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="medium")

obs, info = env.reset(seed=42)
episode_reward = 0
steps = 0

for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    steps += 1

    if terminated or truncated:
        print(f"Episode done! Reward: {episode_reward}, Steps: {steps}")
        break
```

Or via CLI:

```bash
uv run agentick evaluate --task GoToGoal-v0 --n-episodes 10
```

### How do I run a benchmark suite?

```python
import agentick

# Create suite
envs = agentick.make_suite("quick", difficulty="medium")  # 5 representative tasks

results = {}
for env in envs:
    success_count = 0
    total_reward = 0

    for episode in range(10):
        obs, _ = env.reset()
        episode_reward = 0

        for _ in range(env.spec.max_episode_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated:
                success_count += 1
                break

        total_reward += episode_reward

    results[env.spec.id] = {
        "success_rate": success_count / 10,
        "avg_reward": total_reward / 10,
    }

for task, metrics in results.items():
    print(f"{task}: {metrics['success_rate']:.1%} success")
```

Or via CLI:

```bash
uv run agentick evaluate --suite quick --difficulty medium --n-episodes 10
uv run agentick evaluate --suite full --difficulty hard --n-episodes 20
```

### What are the different benchmark suites?

- **quick** (5 tasks): Quick sanity check - GoToGoal, MazeNavigation, KeyDoorPuzzle, SokobanPush, PreciseNavigation
- **navigation** (5 tasks): All navigation tasks
- **memory** (5 tasks): All memory tasks
- **reasoning** (5 tasks): All reasoning tasks
- **control** (4 tasks): All control tasks
- **full** (40+ tasks): Comprehensive benchmark

```python
# Create different suites
quick_envs = agentick.make_suite("quick")  # 5 tasks
nav_envs = agentick.make_suite("navigation")  # 5 tasks
full_envs = agentick.make_suite("full")  # 40+ tasks
```

### How do I get reproducible results?

Always use seeds:

```python
import agentick
import numpy as np
import torch

# Fix all random seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create environment with seed
env = agentick.make("GoToGoal-v0", seed=seed)

# Reset with seed
obs, _ = env.reset(seed=seed)

# Run deterministically
# Results will be identical across runs
```

## Using Different Agent Types

### How do I use Agentick with deep RL (PyTorch, TensorFlow)?

```python
import agentick
import gymnasium as gym

# Create vectorized environments for fast training
def make_env():
    return agentick.make("GoToGoal-v0", render_mode="rgb_array", difficulty="medium")

envs = gym.vector.SyncVectorEnv([make_env for _ in range(16)])

# Use with your RL training code
# Example with PyTorch + CleanRL
for epoch in range(100):
    obs, _ = envs.reset()

    for step in range(256):
        # Your policy network here
        actions = agent(torch.tensor(obs))
        obs, rewards, terminated, truncated, info = envs.step(actions.cpu().numpy())

        # Your training loop
        agent.update(obs, actions, rewards)
```

See [docs/agents/rl_agents.md](agents/rl_agents.md) for complete examples.

### How do I use Agentick with LLMs (GPT, Claude)?

```python
import agentick
from anthropic import Anthropic

env = agentick.make("GoToGoal-v0", render_mode="language")
client = Anthropic()

obs, info = env.reset()

for _ in range(50):
    # Build prompt from natural language observation
    prompt = f"""You are an agent in a gridworld.
Current state: {obs}
Available actions: {', '.join(info['valid_actions'])}
Choose the best action."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse action from response
    action_text = response.content[0].text.lower()
    action = {"north": 0, "south": 1, "east": 2, "west": 3}.get(action_text, env.action_space.sample())

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

See [docs/agents/llm_agents.md](agents/llm_agents.md) for complete examples.

### How do I use Agentick with vision-language models (GPT-4o, Claude Sonnet)?

```python
import agentick
from openai import OpenAI
import base64

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
client = OpenAI()

obs, _ = env.reset()

for _ in range(50):
    # Convert image to base64
    obs_uint8 = (obs * 255).astype('uint8')
    image_b64 = base64.b64encode(obs_uint8).decode()

    # Query VLM
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What action should the blue square take to reach the green circle? Reply with: north, south, east, or west."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }],
        max_tokens=10
    )

    # Parse action
    action_text = response.choices[0].message.content.lower()
    action = {"north": 0, "south": 1, "east": 2, "west": 3}.get(action_text, env.action_space.sample())

    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
```

See [docs/agents/vlm_agents.md](agents/vlm_agents.md) for complete examples.

### Can I use Agentick with Ray/Weights & Biases/Hugging Face?

Yes! Agentick integrates with standard ML tools:

```python
# With Weights & Biases (for RL training)
import wandb
wandb.init(project="agentick-training")

# With Ray for distributed training
from ray.tune import CLIReporter, tune
tune.run(my_training_fn, config={"env": "GoToGoal-v0"})

# Export trajectories to Hugging Face Datasets
trajectories = agentick.collect_trajectories(
    env="GoToGoal-v0",
    agent=my_agent,
    n_episodes=100,
    export_format="huggingface"
)
# trajectories is a HF Dataset ready for fine-tuning
```

## Leaderboard & Submission

### How do I submit to the leaderboard?

1. Run your agent on the benchmark:

```bash
uv run agentick evaluate --suite full --difficulty medium --n-episodes 10 --output results/
```

2. Create submission.yaml:

```yaml
agent_name: "MyAgent-v1"
author: "Your Name"
description: "Description of your agent"
agent_type: "api"
config:
  provider: "openai"
  model: "gpt-4o"
  api_key_env: "OPENAI_API_KEY"
observation_mode: "rgb_array"
suites: ["full"]
```

3. Submit:

```bash
uv run agentick submit --config submission.yaml --results results/evaluation.json --verify
```

See [docs/leaderboard/submitting.md](leaderboard/submitting.md) for detailed guide.

### How is the score calculated?

Agentick uses normalized scoring:

```
normalized_score = (agent_performance - random_baseline) / (optimal_performance - random_baseline)
```

- **0** = Random agent performance
- **1** = Perfect performance
- **0.5** = Midway between random and optimal

For each task, Agentick computes:
- Random baseline from 100 random action episodes
- Optimal baseline from oracle solution
- Agent performance from evaluation episodes
- Normalized score in [0, 1]

Overall score is the average across all tasks in the suite.

### What's the difference between public and private leaderboard?

- **Public**: All submissions visible
- **Private**: Only your submissions visible (for blind evaluation)

By default, submissions are public. To keep private:

```yaml
submission_mode: "private"
```

### Can I update a submission?

Yes, you can submit improved versions. The leaderboard tracks:
- All submission versions
- Date and time
- Performance trend

Just submit again with the same agent_name but updated results.

## Troubleshooting

### Why is my agent failing on tasks?

Common issues:

1. **Wrong observation mode**: Your agent expects different observation format
   ```python
   # If your agent uses text, use language mode
   env = agentick.make("GoToGoal-v0", render_mode="language")

   # If your agent uses pixels, use rgb_array
   env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
   ```

2. **Invalid actions**: You're sending invalid action indices
   ```python
   # Check valid actions
   print(env.action_space)  # Discrete(5) means 0-4

   # Or check info
   obs, info = env.reset()
   print(info['valid_actions'])  # List of allowed actions
   ```

3. **Memory issues**: Running out of RAM
   ```python
   # Reduce vectorization
   import gymnasium as gym
   envs = gym.vector.SyncVectorEnv([lambda: agentick.make("GoToGoal-v0") for _ in range(4)])  # Not 16
   ```

### Why are results not reproducible?

Common causes:

1. **Not setting seeds**:
   ```python
   np.random.seed(42)
   torch.manual_seed(42)
   env.reset(seed=42)
   ```

2. **Floating point non-determinism**: Some operations aren't bitwise reproducible
   ```python
   # Use lower precision or deterministic algorithms
   torch.use_deterministic_algorithms(True)
   ```

3. **Different Python/library versions**: Always pin versions
   ```bash
   pip freeze > requirements.txt
   ```

### I'm getting out of memory errors. What can I do?

```python
# Option 1: Reduce number of vectorized environments
import gymnasium as gym
envs = gym.vector.SyncVectorEnv([lambda: agentick.make("GoToGoal-v0") for _ in range(4)])

# Option 2: Use lower precision
torch.set_default_dtype(torch.float16)

# Option 3: Process episodes serially
for episode in range(100):
    env = agentick.make("GoToGoal-v0")
    obs, _ = env.reset()
    # ... run episode, environment freed after
```

### I'm getting slow performance. How do I optimize?

```python
# Use fast_mode to skip expensive conversions
env = agentick.make("GoToGoal-v0", fast_mode=True)

# Use state_dict for programmatic access
env = agentick.make("GoToGoal-v0", render_mode="state_dict", fast_mode=True)

# Vectorize for RL
import gymnasium as gym
envs = gym.vector.SyncVectorEnv([lambda: agentick.make("GoToGoal-v0") for _ in range(16)])
```

## Contributing & Custom Tasks

### How do I add a custom task?

```python
import numpy as np
from agentick import TaskSpec, register_task
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.configs import DifficultyConfig

@register_task("MyTask-v0", tags=["custom", "navigation"])
class MyTask(TaskSpec):
    """My custom task."""

    name = "MyTask-v0"
    description = "Navigate to randomly placed goals"
    capability_tags = ["navigation", "custom"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=5, max_steps=20),
        "hard": DifficultyConfig(name="hard", grid_size=15, max_steps=100),
    }

    def generate(self, seed):
        """Generate task instance."""
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        grid = Grid(size, size)

        # Add walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Place goal
        y, x = rng.integers(1, size-1, 2)
        grid.objects[y, x] = ObjectType.GOAL

        return grid, {
            "agent_start": (1, 1),
            "goal_positions": [(y, x)],
            "max_steps": self.difficulty_config.max_steps,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Dense reward function."""
        return 0.1  # Or implement custom logic

    def check_success(self, state):
        """Check if task solved."""
        return True  # Or implement custom logic

# Now use it
env = agentick.make("MyTask-v0", difficulty="easy")
```

See [docs/extending/custom_tasks.md](extending/custom_tasks.md) for complete guide.

### How do I submit a custom task to the official benchmark?

1. Implement task following the custom task guide
2. Create comprehensive tests
3. Document task thoroughly
4. Fork repository and create PR with:
   - Task implementation
   - Tests
   - Documentation
   - Examples

See [CONTRIBUTING.md](https://github.com/agentick/agentick/blob/main/CONTRIBUTING.md) for contribution guidelines.

## Design Philosophy

### Why is Agentick grid-based instead of 3D environments?

Grid-based design provides:
- **Efficiency**: Fast rendering and execution
- **Clarity**: Easy to visualize and debug
- **Scalability**: Easy to procedurally generate variants
- **Focus**: Tests reasoning, not graphics quality
- **Accessibility**: Works on any hardware

### Why procedural generation instead of hand-crafted levels?

Procedural generation enables:
- **Infinite curriculum**: Unlimited training data at any difficulty
- **Reproducibility**: Same seed = same task
- **Generalization testing**: Unseen instances at test time
- **Scalability**: 1000s of task variations

### Why Gymnasium API instead of something custom?

Gymnasium compatibility means:
- **Ecosystem**: Works with RL libraries (CleanRL, Stable Baselines)
- **Standardization**: Familiar interface for RL researchers
- **Simplicity**: Standard reset(), step(), render()
- **Portability**: Easy migration to/from other envs

### Why multiple observation modes?

Different agents need different representations:
- **LLMs**: Natural language understanding
- **VLMs**: Visual understanding
- **RL agents**: Pixels or structured state
- **Bots**: Direct state access
- **Humans**: Visual + playable

Multi-modal support means one benchmark for all agent types.

## Performance & Scaling

### What's the episode length for each task?

Varies by task and difficulty:

```
Easy: 20-50 steps
Medium: 50-200 steps
Hard: 100-500 steps
Expert: 200-1000+ steps
```

You can query programmatically:

```python
env = agentick.make("GoToGoal-v0", difficulty="hard")
print(env.spec.max_episode_steps)  # 100
```

### How fast is Agentick?

With vectorization:

```python
import gymnasium as gym
import agentick

# 16 parallel environments
envs = gym.vector.SyncVectorEnv([lambda: agentick.make("GoToGoal-v0") for _ in range(16)])

# ~10,000 steps per second on modern hardware
obs, _ = envs.reset()
for _ in range(1000):
    actions = envs.action_space.sample()  # Or from agent
    obs, rewards, terminated, truncated, _ = envs.step(actions)
    # Process at ~10k steps/sec
```

Rendering modes affect performance:

- **state_dict**: ~15k steps/sec (fastest)
- **ascii**: ~10k steps/sec
- **rgb_array**: ~5k steps/sec (depends on resolution)

## Getting Help

### Where can I find more documentation?

- **Getting Started**: [docs/getting_started/](getting_started/)
- **Concepts**: [docs/concepts/](concepts/)
- **Agent Guides**: [docs/agents/](agents/)
- **API Reference**: [docs/api/](api/)
- **Extending**: [docs/extending/](extending/)

### How do I report bugs or request features?

Please open issues on GitHub: https://github.com/anthropics/agentick/issues

Include:
- Clear description of problem/feature
- Reproducible example code
- Python version, Agentick version
- Relevant error messages

### How do I get help?

1. **Check FAQ** (you're reading it!)
2. **Search existing issues**: github.com/anthropics/agentick/issues
3. **Join community**: Discussion forums (coming soon)
4. **Email**: support@agentick.ai

## Licensing & Citation

### Can I use Agentick commercially?

Yes! Agentick is MIT licensed, which allows:
- Commercial use
- Modification
- Distribution
- Private use

The only requirement is including the license notice.

### How do I cite Agentick?

```bibtex
@software{agentick2025,
  title={Agentick: A Comprehensive Benchmark for Evaluating Generally Capable AI Agents},
  author={Agentick Team},
  year={2025},
  url={https://github.com/anthropics/agentick}
}
```

Or in text:

> We evaluated our agent on Agentick, a comprehensive benchmark for generally capable AI agents (Agentick Team, 2025).

---

**Can't find an answer?** Open an issue or reach out to the community!
