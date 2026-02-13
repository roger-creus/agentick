# Agentick

**Universal benchmark for evaluating AI agents across all paradigms**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Agentick provides 38 procedurally generated tasks for evaluating AI agents. Train and evaluate any agent type: RL, LLM, VLM, hybrid, or human.

## Key Features

- **38 Tasks** across navigation, memory, reasoning, skill discovery, control, and more
- **Multi-Modal Observations**: text, pixels, language, structured state
- **Training-First**: vectorized environments, trajectory export, curriculum learning
- **Universal Support**: RL, LLMs, VLMs, bots, humans
- **Capability Decomposition**: radar charts showing agent strengths/weaknesses
- **Experiment System**: pre-configured suites, reproducible evaluation, leaderboard

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

40+ runnable examples in `examples/`:

```bash
# Basic usage
uv run python examples/basics/01_make_and_step.py

# RL training
uv run python examples/rl/sb3_ppo.py

# LLM agent (requires API key)
export OPENAI_API_KEY="your-key"
uv run python examples/llm/openai_text_agent.py

# Data collection
uv run python examples/data/collect_oracle_trajectories.py
```

See [examples/README.md](examples/README.md) for full list.

## Task Gallery

| Capability | Example Tasks | Count |
|------------|---------------|-------|
| Navigation | GoToGoal, MazeNavigation, FogOfWar | 5 |
| Memory | KeyDoorPuzzle, SequenceMemory, DelayedGratification | 5 |
| Reasoning | SokobanPush, CausalChain, RuleInduction | 5 |
| Skill Discovery | ToolUse, RecipeAssembly, EmergentStrategy | 5 |
| Control | PreciseNavigation, TimingChallenge, ChaseEvade | 4 |
| Combinatorial | LightsOut, GraphColoring, PackingPuzzle | 4 |
| Adversarial | DeceptiveReward, DistributionShift | 3 |
| Meta-Learning | FewShotAdaptation, TaskInterference | 2 |
| Multi-Agent | CooperativeTransport, CompetitiveTag | 2 |

**Total: 38 tasks**, each with 4 difficulty levels (easy, medium, hard, expert).

## CLI

```bash
uv run agentick --version                     # Show version
uv run agentick list-tasks                     # List all tasks
uv run agentick list-suites                    # List benchmark suites
uv run agentick evaluate --submission X --suite Y  # Run evaluation
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
│   ├── tasks/             # Task implementations
│   ├── benchmark/         # Baseline agents
│   ├── leaderboard/       # Evaluation system
│   ├── data/              # Data collection
│   └── visualization/     # Plotting utilities
├── examples/              # 40+ runnable examples
├── experiments/           # Experiment configs
├── docs/                  # Documentation
└── tests/                 # Test suite
```

## Contributing

Contributions welcome! Areas of interest:

### New Tasks
- Implement tasks in `agentick/tasks/`
- Follow existing patterns (see `tasks/navigation/go_to_goal.py`)
- Register in `tasks/__init__.py`
- Add tests

### New Examples
- Add to `examples/` directory
- Include docstring with runtime estimate
- Make self-contained (no dependencies on other examples)
- Test that it runs: `uv run python examples/your_example.py`

### Bug Fixes
- Write a test that reproduces the bug
- Fix the issue
- Verify test passes

### Documentation
- Keep pages concise (<200 lines)
- Focus on library features, not general advice
- Include runnable code examples
- Update navigation in `mkdocs.yml`

## Development Setup

```bash
# Clone repository
git clone https://github.com/agentick/agentick.git
cd agentick

# Install dependencies
uv sync --extra all

# Install dev tools
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Run linters
uv run ruff check agentick/ tests/
uv run mypy agentick/

# Build docs
cd docs && uv run mkdocs serve
```

## Roadmap

- [ ] Additional task categories (physics, language grounding)
- [ ] Web-based human evaluation interface
- [ ] Integration with more RL libraries
- [ ] Curriculum learning utilities
- [ ] Multi-agent task suite expansion

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
