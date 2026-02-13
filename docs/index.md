# Agentick

**Universal benchmark for evaluating AI agents across all paradigms**

Agentick provides 38 procedurally generated tasks spanning navigation, memory, reasoning, skill discovery, control, and more. Evaluate any agent type: RL, LLM, VLM, hybrid, or human.

## Key Features

### 🎯 Multi-Modal Observations
- **Text**: ASCII, natural language, structured language
- **Visual**: RGB pixels (84×84×3)
- **Structured**: Python dictionaries with full state
- **Human**: Interactive pygame interface

### 📊 Capability-Decomposed Scoring
Evaluate agents across 7+ capabilities with radar charts showing strengths and weaknesses.

### 🚂 Training-First Design
- Vectorized environments for fast RL training
- Curriculum learning support
- Trajectory export for imitation learning
- Built-in logging and checkpointing

### 🤖 Universal Agent Support
Works with any agent type out of the box:
- **RL**: CleanRL, Stable Baselines3, custom policies
- **LLM**: OpenAI, Anthropic, HuggingFace
- **VLM**: Vision-language models
- **Hybrid**: Combine approaches
- **Human**: Human-in-the-loop evaluation

### 📈 Experiment System
- Pre-configured benchmark suites
- Paper-ready plotting
- Reproducible evaluation
- Leaderboard submission

## Quick Start

```python
import agentick

# Create environment
env = agentick.make("GoToGoal-v0")
obs, info = env.reset()

# Agent loop
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Agentick
git clone https://github.com/agentick/agentick.git
cd agentick
uv sync --extra all
```

See [Installation Guide](getting_started/installation.md) for details.

## Task Gallery

38 tasks across 7+ capability categories:

| Capability | Example Tasks | Count |
|------------|---------------|-------|
| **Navigation** | GoToGoal, MazeNavigation, FogOfWar | 5 |
| **Memory** | KeyDoorPuzzle, SequenceMemory, DelayedGratification | 5 |
| **Reasoning** | SokobanPush, CausalChain, RuleInduction | 5 |
| **Skill Discovery** | ToolUse, RecipeAssembly, EmergentStrategy | 5 |
| **Control** | PreciseNavigation, TimingChallenge, ChaseEvade | 4 |
| **Combinatorial** | LightsOut, GraphColoring, PackingPuzzle | 4 |
| **Adversarial** | DeceptiveReward, DistributionShift, NoisyObservation | 3 |
| **Meta-Learning** | FewShotAdaptation, TaskInterference | 2 |
| **Multi-Agent** | CooperativeTransport, CompetitiveTag | 2 |

Browse all tasks: [Task Reference](concepts/tasks.md)

## Example Use Cases

### Train RL Agent
```python
from stable_baselines3 import PPO

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
```

See: [RL Training Examples](/examples/rl/)

### Evaluate LLM Agent
```python
from agentick.leaderboard.adapters import APIAgent

env = agentick.make("GoToGoal-v0", render_mode="language")
agent = APIAgent(provider="openai", model="gpt-4o")

obs, info = env.reset()
action = agent.act(env, obs, info)
```

See: [LLM Examples](/examples/llm/)

### Collect Training Data
```python
from agentick.data.collector import TrajectoryCollector
from agentick.agents.oracle import OracleAgent

env = agentick.make("GoToGoal-v0")
collector = TrajectoryCollector(env)
oracle = OracleAgent(env)

trajectory = collector.collect_episode(oracle)
trajectory.save("demo.json")
```

See: [Data Collection Examples](/examples/data/)

## Learn More

- [Getting Started](getting_started/quickstart.md) - 5-minute tutorial
- [Examples](/examples/README.md) - 40+ runnable examples
- [Tasks](concepts/tasks.md) - Browse all 38 tasks
- [CLI Reference](cli.md) - Command-line interface
- [Leaderboard](leaderboard/overview.md) - Submit your agent

## Contributing

Contributions welcome! See examples and documentation for patterns.

## License

MIT License - see LICENSE file for details.
