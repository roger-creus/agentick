# Full Benchmark Pipeline

This directory contains a complete benchmark pipeline for evaluating agents on AgentICK tasks.

## Quick Start

Run a single experiment:
```bash
uv run python examples/experiments/full_benchmark/run_single_benchmark.py \
    examples/experiments/full_benchmark/configs/random_agent.yaml
```

Run all benchmarks (takes several hours with LLM agents):
```bash
bash examples/experiments/full_benchmark/run_all_benchmarks.sh
```

## Available Benchmark Configs

### No Dependencies (Quick)
- `configs/random_agent.yaml` - Random baseline (~2 min)
- `configs/oracle_agent.yaml` - Greedy heuristic baseline (~5 min)

### Reinforcement Learning (Requires GPU, ~hours)
- `configs/ppo_pixels.yaml` - PPO trained on pixels

### LLM Agents (Requires API Keys, ~$10-20 per run)
- `configs/openai_text.yaml` - GPT-4 with text observations
- `configs/openai_vision.yaml` - GPT-4 with vision
- `configs/anthropic_text.yaml` - Claude Sonnet 4.5 with text
- `configs/anthropic_vision.yaml` - Claude Sonnet 4.5 with vision

## Pipeline Structure

```
full_benchmark/
├── configs/                    # Experiment configurations
│   ├── random_agent.yaml
│   ├── oracle_agent.yaml
│   ├── ppo_pixels.yaml
│   ├── openai_text.yaml
│   ├── openai_vision.yaml
│   ├── anthropic_text.yaml
│   └── anthropic_vision.yaml
├── run_single_benchmark.py     # Run one config
├── run_all_benchmarks.sh       # Run all configs in sequence
├── train_and_eval_ppo.py       # Train PPO then evaluate
├── plot_all_results.py         # Generate all plots
└── generate_report.py          # Generate HTML report with plots + videos
```

## Output Structure

Results are saved to `results/full_benchmark/<experiment_name>/`:
```
results/full_benchmark/random_baseline/
├── random_baseline_results.json   # Per-episode results
├── videos/                         # One video per task
│   ├── random_baseline_GoToGoal-v0-episode-0.mp4
│   └── ...
└── traces/                         # Text interaction logs
    ├── GoToGoal-v0.txt
    └── ...
```

## Estimated Runtimes and Costs

| Agent Type | Runtime | Cost | Requirements |
|-----------|---------|------|--------------|
| Random | 2 min | Free | None |
| Greedy | 5 min | Free | None |
| PPO | 2-4 hours | Free | GPU |
| OpenAI (text) | 30-60 min | $10-15 | API key |
| OpenAI (vision) | 30-60 min | $15-25 | API key |
| Anthropic (text) | 30-60 min | $8-12 | API key |
| Anthropic (vision) | 30-60 min | $12-20 | API key |

**Total for all agents: 4-6 hours, ~$50-80 in API costs**

## Usage Examples

### Run random baseline
```bash
uv run python examples/experiments/full_benchmark/run_single_benchmark.py \
    examples/experiments/full_benchmark/configs/random_agent.yaml
```

### Run with custom output directory
```bash
uv run python examples/experiments/full_benchmark/run_single_benchmark.py \
    examples/experiments/full_benchmark/configs/random_agent.yaml \
    --output-dir my_custom_results
```

### Generate plots from all results
```bash
uv run python examples/experiments/full_benchmark/plot_all_results.py
```

### Generate HTML report
```bash
uv run python examples/experiments/full_benchmark/generate_report.py
# Open results/full_benchmark/report.html in browser
```

## API Keys Setup

For LLM agents, create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

The scripts will automatically load these when running LLM agents.

## Customizing Experiments

Edit the YAML configs to customize:
- `tasks`: List of task IDs to evaluate on
- `num_seeds`: Number of random seeds per task
- `timeout`: Max steps per episode
- `render_mode`: "rgb_array" for vision, "text" for text-only

Example:
```yaml
name: my_custom_experiment
description: "My custom agent evaluation"
tasks:
  - GoToGoal-v0
  - MazeNavigation-v0
agent_type: random
num_seeds: 5
episodes_per_seed: 1
render_mode: rgb_array
timeout: 500
record_video: true
record_traces: true
```
