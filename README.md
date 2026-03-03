# Agentick

**Universal benchmark for evaluating AI agents across all paradigms**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Agentick provides 38 procedurally generated tasks for evaluating AI agents. Train and evaluate any agent type: RL, LLM, VLM, hybrid, or human.

## Key Features

- **38 Tasks** across navigation, planning, reasoning, memory, generalization, and multi-agent coordination
- **Multi-Modal Observations**: isometric pixel sprites, flat 2D grid, ASCII text, language, structured state
- **Training-First**: vectorized environments, trajectory export, SFT and RL fine-tuning
- **Universal Support**: RL, LLMs, VLMs, bots, humans
- **Capability Decomposition**: radar charts showing agent strengths/weaknesses
- **Experiment System**: pre-configured YAML configs, reproducible evaluation, leaderboard

## Quick Start

```python
import agentick

# Create environment (default: ASCII observation)
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
# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/agentick/agentick.git
cd agentick
uv sync --extra all

# Verify
uv run agentick --version
```

### Dependency Groups

```bash
uv sync --extra rl         # RL training (torch, wandb)
uv sync --extra llm        # LLM agents (openai, anthropic)
uv sync --extra viz        # Visualization (matplotlib, seaborn)
uv sync --extra all        # Everything
```

## Examples

Runnable examples in `examples/`:

```bash
# Basic usage
uv run python examples/basics/01_make_and_step.py

# RL training (flat 2D pixel observations)
uv run python examples/rl/sb3_ppo.py

# LLM agent (requires API key)
export OPENAI_API_KEY="your-key"
uv run python examples/llm/openai_text_agent.py

# Data collection
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py

# Run a full benchmark experiment
uv run python examples/experiments/run_single_benchmark.py examples/experiments/configs/random_agent.yaml
```

See [examples/README.md](examples/README.md) for the full list.

## Render Modes

| Mode | Description | Output |
|------|-------------|--------|
| `"ascii"` | ANSI-colored text grid (default) | `str` |
| `"language"` | Natural language description | `str` |
| `"language_structured"` | Structured dict with position, surroundings | `dict` |
| `"rgb_array"` | **Isometric pixel sprites** (512×512, Kenney assets) | `np.ndarray` |
| `"rgb_array_flat"` | Flat 2D top-down grid sprites | `np.ndarray` |
| `"state_dict"` | Numpy arrays for grid layers and agent state | `dict` |

```python
# Isometric rendering (default pixel mode)
env = agentick.make("MazeNavigation-v0", render_mode="rgb_array")

# Flat 2D grid (faster, good for RL training)
env = agentick.make("MazeNavigation-v0", render_mode="rgb_array_flat")
```

## Task Gallery

| Capability | Example Tasks | Count |
|------------|---------------|-------|
| Navigation | GoToGoal, MazeNavigation, CuriosityMaze, TimingChallenge | 8 |
| Planning | SokobanPush, KeyDoorPuzzle, PackingPuzzle, ToolUse | 9 |
| Reasoning | CausalChain, SwitchCircuit, GraphColoring, ProgramSynthesis | 9 |
| Memory | SequenceMemory, DelayedGratification, TreasureHunt, FogOfWar | 4 |
| Generalization | FewShotAdaptation, DistributionShift, NoisyObservation | 3 |
| Multi-Agent | CooperativeTransport, ChaseEvade, Herding, EmergentStrategy | 5 |

**Total: 38 tasks**, each with 4 difficulty levels (easy, medium, hard, expert).

## CLI

```bash
uv run agentick --version                     # Show version
uv run agentick list-tasks                     # List all tasks
uv run agentick list-suites                    # List benchmark suites
uv run agentick evaluate --submission X --suite Y  # Run evaluation
```

## Try It First

The fastest way to explore Agentick is the **interactive webapp** — play tasks yourself, watch oracle demos, and browse all observation modalities:

```bash
uv run agentick webapp          # Opens http://localhost:5000
```

## Documentation

- [Installation Guide](docs/getting_started/installation.md)
- [Quickstart Tutorial](docs/getting_started/quickstart.md)
- [Task Reference](docs/concepts/tasks.md)
- [RL Training Guide](docs/agents/rl_agents.md)
- [LLM Agent Guide](docs/agents/llm_agents.md)
- [API Reference](docs/api/index.md)

## Project Structure

```
agentick/
├── agentick/               # Core library
│   ├── core/              # Environment, grid, renderer, types
│   ├── tasks/             # 38 task implementations
│   ├── oracles/           # Optimal reference policies
│   ├── agents/            # LLM/VLM/RL agent harnesses
│   ├── leaderboard/       # Evaluation system and suites
│   ├── rendering/         # Isometric sprite renderer
│   ├── data/              # Trajectory collection
│   ├── training/          # SFT, BC, and RL trainers
│   └── human/             # Web showcase and human play
├── examples/              # Runnable examples
├── docs/                  # Documentation + showcase gallery
└── tests/                 # Test suite
```
## Citation

```bibtex
@software{agentick2025,
  title={Agentick: Universal Benchmark for AI Agents},
  author={Agentick Team},
  year={2025},
  url={https://github.com/agentick/agentick}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with [Gymnasium](https://gymnasium.farama.org/), inspired by research in agent evaluation and general intelligence.
